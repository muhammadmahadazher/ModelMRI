"""The browser viewer must produce exactly the numbers the tool produces.

    python scripts/build_frontend.py --viewer
    python tests/viewer_check.py

`frontend/src/viewer.ts` re-implements `modelmri/session.py` in TypeScript:
gunzip, base64, uint8 dequantisation, rounding. Two implementations of one
format drift — and a viewer that renders a *slightly* different matrix than
the tool is worse than no viewer, because nothing on screen would say so.

So this parses the same file both ways and compares every cell. It serves the
bundled viewer over http (module scripts do not load from file://) and drives
it with Playwright, reading the numbers back out of the running page rather
than trusting the source to be equivalent.

Needs: playwright (uv run playwright install chromium), and a built viewer.
"""

from __future__ import annotations

import asyncio
import functools
import http.server
import socketserver
import sys
import threading
import urllib.parse
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
VIEWER = ROOT / "modelmri" / "static" / "viewer"
FIXTURE = ROOT / "tests" / "fixtures" / "parity.mri"

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def build_fixture() -> bytes:
    """A session with structure worth disagreeing about: an attention sink,
    a causal mask, and values across the whole quantisation range."""
    sys.path.insert(0, str(ROOT))
    from modelmri import session

    n, layers, heads = 24, 4, 3
    matrices = {}
    for layer in range(layers):
        for head in range(heads):
            rows = []
            for r in range(n):
                raw = [
                    ((layer * 7 + head * 13 + r * 3 + c * 11) % 23) + 0.5
                    for c in range(r + 1)
                ]
                raw[0] += 40.0  # the sink every real head has
                raw += [0.0] * (n - r - 1)
                total = sum(raw)
                rows.append([v / total for v in raw])
            matrices[(layer, head)] = rows
    return session.build(
        model_id="parity/fixture",
        device="cpu",
        dtype="float32",
        n_params=1234,
        tokens=[f"t{i}" for i in range(n)],
        prompt="parity",
        generation="check",
        attention=matrices,
        n_layers=layers,
        n_heads=heads,
        note="every cell must match",
    )


def build_image_fixture() -> bytes:
    """A `.mri` carrying an image run, hostile in the four ways a stranger's
    can be — none of which this writer can produce.

    `session.build` validates through `session._image`, so the file is built
    legitimately and then EDITED: the point is a file that never went through
    this reader at all, which is what the viewer build is handed. It gets the
    raw section straight out of the gzip with only the provenance checked.
    """
    import base64
    import gzip
    import json

    sys.path.insert(0, str(ROOT))
    from modelmri import session

    png = (
        "data:image/png;base64,"
        + base64.b64encode(
            bytes.fromhex(
                "89504e470d0a1a0a0000000d494844520000000100000001080600000"
                "01f15c4890000000a49444154789c6360000002000100ffff0300000600"
                "0557bfabd40000000049454e44ae426082"
            )
        ).decode()
    )

    blob = session.build(
        model_id="stabilityai/sd-turbo",
        device="cpu",
        dtype="float32",
        n_params=None,
        tokens=[],
        prompt="",
        generation="",
        attention={},
        n_layers=0,
        n_heads=0,
        note="a file from a stranger",
        scope="one denoising run",
        image={
            "provenance": {
                "repo": "stabilityai/sd-turbo",
                "family": "diffusion",
                "architecture": "UNet2DConditionModel",
                "revision": "",
                "kind": "denoising",
            },
            "prompt": "an astronaut riding a horse",
            "seed": None,
            "scheduler": "Euler",
            "frames": [
                {
                    "step": 0,
                    "timestep": 999.0,
                    "png": png,
                    "size": [64, 64],
                    "downsampled": False,
                    "latent_rms": 1.25,
                }
            ],
            "steps_requested": 20,
            "steps_run": 20,
            "decoded_steps": [0],
            "skipped_steps": [],
            "steps_never_reached": [],
            "attention": {
                "tokens": ["an", "astronaut", "<pad>", "<pad>"],
                "steps": [
                    {
                        "step": 0,
                        "timestep": 999.0,
                        "per_token": [0.4, 0.3, 0.2, 0.1],
                        "blocks": 16,
                    }
                ],
                "padding_from": 2,
                "conditioning_width": 77,
                "columns_unlabelled": 0,
                "steps_requested": 20,
                "steps_measured": 1,
                "resolutions": [16],
                "means": "one step of twenty",
            },
            "means": "1 decoded frame of a 20-step run.",
        },
    )

    doc = json.loads(gzip.decompress(blob).decode("utf-8"))
    img = doc["image"]
    # 1. the boundary nobody measured -- absent, which is not 0
    img["attention"].pop("padding_from", None)
    # 2. the run length nobody stated -- absent, which is not a 0-step run
    img.pop("steps_requested", None)
    img.pop("steps_run", None)
    # 3. a frame that is a LINK. Opening the file must not tell whoever wrote
    #    it that you did.
    img["frames"].append(
        {
            "step": 1,
            "timestep": 500.0,
            "png": "https://beacon.invalid/1x1.png",
            "size": [64, 64],
            "downsampled": False,
            "latent_rms": None,
        }
    )
    # 4. arrays that are not there at all -- a TypeError with no error
    #    boundary above it is a white page where the recording used to be.
    for gone in ("skipped_steps", "steps_never_reached", "decoded_steps"):
        img.pop(gone, None)
    return gzip.compress(json.dumps(doc, separators=(",", ":")).encode(), 6)


def build_robot_fixture() -> bytes:
    """A `.mri` carrying a robot finding, hostile in the ways a stranger's can
    be — built legitimately and then EDITED, because the point is a file that
    never passed through `session._vla`.

    `/api/vla/share` wrote this section from the day the robot work landed and
    nothing served it back, so the recipient opened an empty text session. The
    reader was there the whole time; only the route and the panel were not.
    """
    import gzip
    import json

    sys.path.insert(0, str(ROOT))
    from modelmri import session

    blob = session.build(
        model_id="smolvla",
        device="cpu",
        dtype="float32",
        n_params=1,
        tokens=["a"],
        prompt="a",
        generation="",
        attention={},
        n_layers=1,
        n_heads=1,
        note="a robot finding from a stranger",
        vla={
            "provenance": {
                "policy": "lerobot/smolvla_base",
                "dataset": "lerobot/pusht",
                "camera": "observation.images.top",
                "revision": "abc123",
                "episode": 5,
                "timestep": 12,
            },
            "frame": "data:image/png;base64,AAAA",
            "frame_size": [96, 96],
            "occlusion": {
                "baseline": "episode_mean",
                "grid": [2, 2],
                "attention_agreement": -0.12,
                "blocks": [
                    {
                        "row": 0,
                        "col": 0,
                        "shift": 0.4,
                        "control_max": 0.1,
                        "clears_control": True,
                        "control_draws": 8,
                    },
                    {
                        "row": 0,
                        "col": 1,
                        "shift": 0.1,
                        "control_max": None,
                        "clears_control": None,
                        "control_draws": 0,
                    },
                ],
            },
        },
    )

    doc = json.loads(gzip.decompress(blob).decode("utf-8"))
    vla = doc["vla"]
    # A FRAME THAT IS A LINK. `session._vla` refuses one in as many words;
    # the viewer shim serves the section raw, so only the panel stands between
    # a hostile file and the recipient announcing that they opened it.
    vla["frame"] = "https://beacon.invalid/frame.png"
    return gzip.compress(json.dumps(doc, separators=(",", ":")).encode(), 6)


async def robot_side(port: int) -> dict:
    """Open that file in the real viewer and read what landed on screen."""
    from playwright.async_api import async_playwright

    async with async_playwright() as p:
        browser = await p.chromium.launch()
        page = await browser.new_page()
        fetched: list[str] = []
        page.on(
            "request",
            lambda r: fetched.append(r.url) if "beacon.invalid" in r.url else None,
        )
        await page.goto(f"http://127.0.0.1:{port}/", wait_until="networkidle")
        got = await page.evaluate(
            """async () => {
              const blob = await (await fetch('./robot.mri')).blob();
              const file = new File([blob], 'robot.mri');
              const input = document.querySelector('input[type=file][accept=".mri"]');
              const dt = new DataTransfer(); dt.items.add(file);
              input.files = dt.files;
              input.dispatchEvent(new Event('change', {bubbles: true}));
              await new Promise(r => setTimeout(r, 1500));

              const served = await (await fetch('/api/vla/replay')).json();
              const panel = [...document.querySelectorAll('.panel')]
                .find(p => p.innerText.includes('ROBOT FINDING'));
              return {
                served: served.available === true,
                mounted: !!panel,
                text: panel ? panel.innerText : '',
                linked: [...document.images]
                  .map(i => i.src)
                  .filter(s => !s.startsWith('data:')).length,
              };
            }"""
        )
        await browser.close()
        got["requested"] = fetched
        return got


async def image_side(port: int) -> dict:
    """Open that file in the real viewer and read what it put on screen.

    THIS EXISTS BECAUSE THE PANEL WAS NOT MOUNTED HERE AT ALL. `ImageRunReplay`
    sat inside App's `!VIEWER` gate under a comment explaining that it sits
    OUTSIDE it -- so the build A6 exists for, the recipient's, was the one
    build that never rendered a shared image run, while `/api/image/replay`
    answered `available: true` to nobody. No unit test could see that: the
    component was correct and nothing rendered it.
    """
    from playwright.async_api import async_playwright

    async with async_playwright() as p:
        browser = await p.chromium.launch()
        page = await browser.new_page()
        fetched: list[str] = []
        page.on(
            "request",
            lambda r: fetched.append(r.url) if "beacon.invalid" in r.url else None,
        )
        await page.goto(f"http://127.0.0.1:{port}/", wait_until="networkidle")
        got = await page.evaluate(
            """async () => {
              const blob = await (await fetch('./image.mri')).blob();
              const file = new File([blob], 'image.mri');
              const input = document.querySelector('input[type=file][accept=".mri"]');
              const dt = new DataTransfer(); dt.items.add(file);
              input.files = dt.files;
              input.dispatchEvent(new Event('change', {bubbles: true}));
              await new Promise(r => setTimeout(r, 1500));

              const served = await (await fetch('/api/image/replay')).json();
              const panel = [...document.querySelectorAll('.panel')]
                .find(p => p.innerText.includes('IMAGE RUN'));
              return {
                served: served.available === true,
                mounted: !!panel,
                text: panel ? panel.innerText : '',
                cells: document.querySelectorAll('.irr-cell').length,
                dimmed: document.querySelectorAll('.irr-cell.pad').length,
                linked: [...document.images]
                  .map(i => i.src)
                  .filter(s => !s.startsWith('data:')).length,
              };
            }"""
        )
        await browser.close()
        got["requested"] = fetched
        return got


def python_side(data: bytes) -> dict:
    sys.path.insert(0, str(ROOT))
    from modelmri import session

    parsed = session.parse(data)
    total, cells, worst = 0.0, 0, 0.0
    for key in sorted(parsed.attention):
        layer, head = (int(x) for x in key.split(":"))
        for row in parsed.attention_slice(layer, head)["matrix"]:
            rs = 0.0
            for v in row:
                total += v
                cells += 1
                rs += v
            worst = max(worst, abs(rs - 1))
    return {
        "slices": len(parsed.attention),
        "cells": cells,
        "checksum": round(total, 6),
        "worst": round(worst, 5),
        "tokens": parsed.tokens,
    }


def serve(directory: Path, port: int) -> socketserver.TCPServer:
    handler = functools.partial(
        http.server.SimpleHTTPRequestHandler, directory=str(directory)
    )
    httpd = socketserver.TCPServer(("127.0.0.1", port), handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    return httpd


# `?f=` names a file the local server is serving. Each of these tries to make
# the page fetch somewhere else; a backslash defeated the first version of the
# guard, which pattern-matched instead of resolving.
HOSTILE = [
    "https://evil.example/x.mri",
    "//evil.example/x.mri",
    "\\\\evil.example\\x.mri",
    "\\/evil.example/x.mri",
    "http://127.0.0.1:9/x.mri",
    "/etc/passwd",
    "../../../../etc/passwd",
    "..%2f..%2fpyproject.toml",
    "javascript:alert(1)",
    "data:text/plain,x",
]


# What the page says about itself after a probe. The file input is rendered
# unconditionally by SessionBar, so it is the marker for "the viewer mounted";
# `.panel.replay` only exists once a session is open.
_STATE = """() => {
  if (document.querySelector('.panel.replay')) return 'opened';
  if (document.querySelector('input[type=file][accept=".mri"]')) return 'idle';
  return 'absent';
}"""


async def hostile_side(port: int) -> dict:
    """Load the viewer with each hostile `?f=` and watch what it requests.

    Returns escaped probes AND probes that never ran. The second list is the
    point: this is a security check, and a probe whose navigation failed looks
    exactly like a probe that loaded and was correctly blocked — no off-origin
    request, no `.panel.replay`. Without it, a dead browser or a server that
    never came up reports ten clean probes having tested nothing.
    """
    from playwright.async_api import Error as PlaywrightError
    from playwright.async_api import async_playwright

    origin = f"http://127.0.0.1:{port}/"
    escaped: list[str] = []
    vacuous: list[str] = []
    async with async_playwright() as p:
        browser = await p.chromium.launch()
        page = await browser.new_page()
        seen: list[str] = []
        page.on("request", lambda r: seen.append(r.url))
        for probe in HOSTILE:
            seen.clear()
            url = f"{origin}?f={urllib.parse.quote(probe, safe='')}"
            failure = ""
            try:
                await page.goto(url, wait_until="networkidle")
            except PlaywrightError as err:
                # One class covers everything Playwright reports here: a
                # navigation the browser aborted, and the `networkidle`
                # timeout (TimeoutError subclasses Error). Neither is fatal to
                # the sweep — the state check below decides whether this probe
                # still tested anything — but neither is a pass on its own,
                # so the reason is kept rather than dropped.
                failure = str(err).splitlines()[0][:120]
            await page.wait_for_timeout(400)
            for requested in seen:
                # Anything off this origin, or reaching above the served
                # directory, means the guard let it through.
                if not requested.startswith(origin):
                    escaped.append(f"{probe} -> {requested}")
            try:
                state = await page.evaluate(_STATE)
            except PlaywrightError as err:
                state = "absent"
                failure = failure or str(err).splitlines()[0][:120]
            if state == "opened":
                escaped.append(f"{probe} -> opened a session")
            elif state == "absent":
                vacuous.append(
                    f"{probe} -> viewer never loaded: {failure or 'no page'}"
                )
        await browser.close()
    return {"escaped": escaped, "vacuous": vacuous, "probes": len(HOSTILE)}


async def browser_side(port: int) -> dict:
    from playwright.async_api import async_playwright

    async with async_playwright() as p:
        browser = await p.chromium.launch()
        page = await browser.new_page()
        await page.goto(f"http://127.0.0.1:{port}/", wait_until="networkidle")
        result = await page.evaluate(
            """async () => {
              const blob = await (await fetch('./parity.mri')).blob();
              const file = new File([blob], 'parity.mri');
              const input = document.querySelector('input[type=file][accept=".mri"]');
              const dt = new DataTransfer(); dt.items.add(file);
              input.files = dt.files;
              input.dispatchEvent(new Event('change', {bubbles: true}));
              await new Promise(r => setTimeout(r, 1200));

              const meta = await (await fetch('/api/attention/meta')).json();
              if (!meta.available) return {error: 'the viewer did not open the file'};
              let total = 0, cells = 0, worst = 0, tokens = null;
              for (let l = 0; l < meta.n_layers; l++) {
                for (let h = 0; h < meta.n_heads; h++) {
                  const d = await (await fetch(`/api/attention?layer=${l}&head=${h}`)).json();
                  if (d.error) return {error: d.error};
                  tokens = d.tokens;
                  for (const row of d.matrix) {
                    let s = 0;
                    for (const v of row) { total += v; cells++; s += v; }
                    worst = Math.max(worst, Math.abs(s - 1));
                  }
                }
              }
              return {
                slices: meta.n_layers * meta.n_heads,
                cells,
                checksum: Number(total.toFixed(6)),
                worst: Number(worst.toFixed(5)),
                tokens,
              };
            }"""
        )
        await browser.close()
        return result


def main() -> int:
    if not (VIEWER / "index.html").is_file():
        print(
            f"no viewer build at {VIEWER}\n  python scripts/build_frontend.py --viewer",
            file=sys.stderr,
        )
        return 1

    data = build_fixture()
    FIXTURE.parent.mkdir(parents=True, exist_ok=True)
    FIXTURE.write_bytes(data)
    (VIEWER / "parity.mri").write_bytes(data)
    (VIEWER / "image.mri").write_bytes(build_image_fixture())
    (VIEWER / "robot.mri").write_bytes(build_robot_fixture())

    expected = python_side(data)
    port = 5921
    httpd = serve(VIEWER, port)
    try:
        got = asyncio.run(browser_side(port))
        hostile = asyncio.run(hostile_side(port))
        shared = asyncio.run(image_side(port))
        robot = asyncio.run(robot_side(port))
    finally:
        httpd.shutdown()

    if got.get("error"):
        print(f"FAILED: {got['error']}", file=sys.stderr)
        return 1

    print(f"  fixture   {len(data) / 1024:.1f} KB, {expected['slices']} slices")
    cells_ok = True
    for key in ("slices", "cells", "checksum", "worst", "tokens"):
        same = got.get(key) == expected[key]
        cells_ok = cells_ok and same
        shown = key if key != "tokens" else "tokens"
        mark = "PASS" if same else "FAIL"
        detail = (
            f"{expected[key]}"
            if key != "tokens"
            else f"{len(expected[key])} identical"
            if same
            else f"python={expected[key][:4]} browser={str(got.get(key))[:60]}"
        )
        print(f"  [{mark}] {shown:9} {detail}")
        if not same and key != "tokens":
            print(f"         python={expected[key]}  browser={got.get(key)}")

    print()
    # A probe that did not run is not a probe that passed. Reported as its own
    # failure line rather than folded into "escaped", because the two mean
    # opposite things: one is a guard that leaked, the other is a guard nobody
    # tested.
    tested = hostile["probes"] - len(hostile["vacuous"])
    clean = not hostile["escaped"] and not hostile["vacuous"]
    ok = cells_ok and clean
    if hostile["escaped"]:
        print(f"  [FAIL] ?f=       escaped: {hostile['escaped'][:4]}")
    if hostile["vacuous"]:
        print(
            f"  [FAIL] ?f=       {len(hostile['vacuous'])} probe(s) tested "
            f"nothing: {hostile['vacuous'][:4]}"
        )
    if clean:
        print(
            f"  [PASS] ?f=       {tested} hostile values, all loaded the "
            f"viewer, none escaped the origin"
        )

    print()
    # A SHARED IMAGE RUN, IN THE BUILD IT WAS WRITTEN FOR. Each line is one
    # thing a file from a stranger could have made this page do.
    image_ok = True
    for label, passed, detail in (
        (
            "mounted",
            shared["served"] and shared["mounted"],
            "the panel is on the page"
            if shared["mounted"]
            else "the panel is NOT mounted in the viewer build"
            if shared["served"]
            else "the viewer did not serve the section at all",
        ),
        (
            "no beacon",
            not shared["requested"] and shared["linked"] == 0,
            f"{shared['linked']} linked <img>, "
            f"{len(shared['requested'])} request(s) off-origin",
        ),
        (
            "reported",
            "never fetches" in shared["text"],
            "the dropped frame is named, not silently missing",
        ),
        (
            "no padding",
            shared["cells"] > 0 and shared["dimmed"] == 0,
            f"{shared['dimmed']} of {shared['cells']} cells dimmed with no "
            f"measured boundary",
        ),
        (
            "unstated",
            "does not say how many steps" in shared["text"]
            and "of 0 step" not in shared["text"],
            "an unstated run length is SAID to be unstated, not printed as 0",
        ),
    ):
        mark = "PASS" if passed else "FAIL"
        print(f"  [{mark}] image     {label:10} — {detail}")
        image_ok = image_ok and passed
    ok = ok and image_ok

    print()
    # A SHARED ROBOT FINDING, in the build it was written for. `/api/vla/share`
    # wrote this section for months with no route and no panel behind it.
    robot_ok = True
    for label, passed, detail in (
        (
            "mounted",
            robot["served"] and robot["mounted"],
            "the panel is on the page"
            if robot["mounted"]
            else "the panel is NOT mounted — a shared robot finding is unreadable"
            if robot["served"]
            else "the viewer did not serve the section at all",
        ),
        (
            "no beacon",
            not robot["requested"] and robot["linked"] == 0,
            f"{robot['linked']} linked <img>, "
            f"{len(robot['requested'])} request(s) off-origin",
        ),
        (
            "control",
            "not yet a finding" in robot["text"],
            "an uncontrolled block is named as uncontrolled, not as a result",
        ),
        (
            "agreement",
            "-0.120" in robot["text"],
            "a negative attention/cause agreement keeps its sign",
        ),
    ):
        mark = "PASS" if passed else "FAIL"
        print(f"  [{mark}] robot     {label:10} — {detail}")
        robot_ok = robot_ok and passed
    ok = ok and robot_ok

    print()
    # Two different failures, and the last line has to name the right one:
    # "THE VIEWER DISAGREES WITH THE TOOL" about a run where every cell
    # matched and the browser simply never started would send the next reader
    # looking for a quantisation bug that is not there.
    if ok:
        print("the viewer and the tool agree on every cell")
    elif not cells_ok:
        print("THE VIEWER DISAGREES WITH THE TOOL")
    elif not image_ok:
        print("THE VIEWER MISHANDLES A SHARED IMAGE RUN — see above")
    elif not robot_ok:
        print("THE VIEWER MISHANDLES A SHARED ROBOT FINDING — see above")
    else:
        print("every cell matched, but the ?f= guard was not proven — see above")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
