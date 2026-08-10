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
SKIP: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> bool:
    (PASS if ok else FAIL).append(name)
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}{(' - ' + detail) if detail else ''}")
    return ok


def skip(section: str, why: str) -> None:
    """A section that could not run, recorded rather than only printed.

    Sections here are conditional on server state, and CI starts a fresh
    server with no model — so /api/model/load is in EXPENSIVE and must not
    fire, and the head-ranking and .mri round-trip sections quietly do
    nothing. Measured: this file reports "18 passed, 0 failed" against a bare
    server and "32 passed, 0 failed" once gpt2 is loaded and prompted, so 14
    checks — including all five over the head-ranking panel — never ran, and
    the exit code said nothing about it. Green now has to be read next to the
    skip list.
    """
    SKIP.append(f"{section}: {why}")
    print(f"  [SKIP] {section} — {why}")


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

        # These assertions are about what a FRESH server does. In CI it always
        # is one; locally it is whatever the last session left behind, and a
        # custom model loaded by some other probe made "the panel starts
        # inert" fail for a reason that had nothing to do with the panel.
        # Establish the precondition instead of assuming it.
        try:
            await page.request.post(BASE.rstrip("/") + "/api/custom/unload")
        except Exception:
            pass  # demo builds have no server to reset

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
        loaded = await page.evaluate(
            "async () => { try { const r = await fetch('/api/session');"
            " return (await r.json()).model.loaded; } catch { return false; } }"
        )
        if demo:
            pass
        elif loaded:
            # A model this server already had. Not a finding — the invariant
            # that matters ("the app never loads one by itself") is asserted
            # above by the network check, which would have caught an
            # /api/model/load fired on mount.
            print("    (a model is already loaded on this server — pill check n/a)")
        else:
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

        print("\nthe UI offers whatever the server can actually answer")
        # `epoch` was a client-side counter, so a reload dropped it to 0 and
        # unmounted the attention and feature panels while the server still
        # held attention for the last generation. You generated 141 tokens,
        # refreshed, and your analysis was gone with nothing saying why.
        meta = await page.evaluate(
            "async () => { try { const r = await fetch('/api/attention/meta');"
            " return await r.json(); } catch { return null; } }"
        )
        if meta and meta.get("available"):
            panel = await page.query_selector(".panel:has(.h-attn)")
            check(
                "attention is available, so the panel is on the page",
                panel is not None,
                f"{meta.get('n_tokens')} tokens, {meta.get('n_layers')} layers",
            )
            toks = await page.query_selector_all(".tok")
            check("the token strip is populated", len(toks) > 1, f"{len(toks)} tokens")
        else:
            print("    (no generation on this server yet — nothing to restore)")

        print("\nerrors are sentences, not envelopes")
        # The API answers failures as {"error": "..."} and those sentences are
        # the good part. The fetch helper used to throw the whole envelope, so
        # what reached the screen was:
        #     Error: 422: {"error":"SAE d_in=768 does not match model …"}
        # Every panel showed errors that way, because every panel goes through
        # that helper. Provoke one and read what a person would see.
        # An ENABLED button. The first control in the features panel is a
        # "Load" that is correctly disabled when no SAE exists for the loaded
        # model — clicking it hung this check for thirty seconds and reported
        # a product failure for a panel that was behaving properly.
        sae_btn = None
        for candidate in await page.query_selector_all(".panel:has(.h-feat) button"):
            if await candidate.is_enabled():
                sae_btn = candidate
                break
        if sae_btn:
            await sae_btn.click()
            await page.wait_for_timeout(4000)
            shown = await page.evaluate(
                """() => [...document.querySelectorAll('.hint, .hint.err, .err')]
                     .map(e => e.innerText.trim()).filter(Boolean).join(' | ')"""
            )
            leaked = [
                m
                for m in ('{"error"', '{"detail"', "Error: 4", "Error: 5")
                if m in shown
            ]
            check(
                "no raw JSON or status code reaches the user",
                not leaked,
                f"leaked {leaked} in: {shown[:90]}" if leaked else "checked live text",
            )
        else:
            skip("features panel refusals", "no enabled control on this server")

        print("\nthe model picker does not resize under you")
        # Its list arrives async. A content-sized sheet opened ~200px tall
        # around "scanning…" and snapped to 78vh when results landed — a 266px
        # jump, which moves whatever row is under the cursor. Sample the height
        # across the load and allow only the entrance animation's scale.
        picker = await page.query_selector(".model-btn")
        if picker:
            await picker.click()
            await page.wait_for_selector(".sheet", timeout=30_000)
            heights = []
            for _ in range(18):
                h = await page.evaluate(
                    "() => { const s = document.querySelector('.sheet');"
                    " return s ? Math.round(s.getBoundingClientRect().height) : 0; }"
                )
                if h:
                    heights.append(h)
                await page.wait_for_timeout(180)
            spread = max(heights) - min(heights) if heights else 0
            check(
                "the picker keeps one height while its list loads",
                spread <= 20,
                f"{spread}px spread over {len(heights)} samples "
                f"({min(heights)}-{max(heights)})",
            )
            # The invariant is that a fixed-height sheet never shows a blank
            # rectangle — not that it always has rows. A CI runner has no
            # models cached, so "Nothing found under …" is the correct and
            # helpful answer, and asserting rows failed the product for
            # behaving properly.
            body = await page.evaluate(
                """() => {
                  const s = document.querySelector('.sheet');
                  if (!s) return null;
                  const rows = s.querySelectorAll('.model-row, .skel-row').length;
                  // text below the tab strip, i.e. the list area's own words
                  const head = s.querySelector('.sheet-head');
                  const all = s.innerText.trim();
                  const chrome = head ? head.innerText.trim() : '';
                  return {rows, said: all.replace(chrome, '').trim().length};
                }"""
            )
            check(
                "the picker always says something, even with nothing to list",
                bool(body) and (body["rows"] > 0 or body["said"] > 12),
                f"{body['rows']} rows, {body['said']} chars of explanation"
                if body
                else "no sheet",
            )
            await page.keyboard.press("Escape")
            await page.wait_for_timeout(400)

        print("\nkeyboard focus is visible")
        # Real Tab presses, not element.focus(). Chromium only sets
        # :focus-visible from actual keyboard interaction, so a sweep built on
        # .focus() reports every button as ringless — a fact about the probe,
        # not the page. That false alarm nearly got "fixed" here.
        #
        # What it did catch, once measured properly: `:focus-visible` and
        # `.model-row` share specificity (0,1,0), so every `all: unset` below
        # it in the file won on source order and left outline-style: none
        # while :focus-visible matched. 19 of 20 controls in the picker moved
        # focus invisibly.
        await page.evaluate("() => document.body.focus()")
        seen, ringless = set(), []
        for _ in range(45):
            await page.keyboard.press("Tab")
            spot = await page.evaluate(
                """() => {
                  const e = document.activeElement;
                  if (!e || e === document.body) return null;
                  const s = getComputedStyle(e);
                  return {
                    id: (typeof e.className === 'string' ? e.className : '')
                        + '|' + (e.innerText || e.value || '').trim().slice(0, 20),
                    ring: (s.outlineStyle !== 'none' && parseFloat(s.outlineWidth) > 0)
                          || s.boxShadow !== 'none',
                  };
                }"""
            )
            if not spot or spot["id"] in seen:
                continue
            seen.add(spot["id"])
            if not spot["ring"]:
                ringless.append(spot["id"][:40])
        check(
            "every control shows a focus ring when tabbed to",
            not ringless,
            f"{len(seen)} controls swept"
            if not ringless
            else f"no ring on: {ringless[:6]}",
        )

        print("\nhead ranking")
        # A leaderboard is read as truth. These check that the numbers arrive
        # with what they do not mean attached, because two of the three ways
        # this feature can lie are about interpretation rather than arithmetic.
        rank_btn = page.locator("button", has_text="Rank heads")
        if await rank_btn.count() == 0:
            skip("head ranking", "no model loaded on this server")
        else:
            await rank_btn.first.click()
            try:
                await page.wait_for_selector(".ranking", timeout=120_000)
            except Exception:
                check("ranking returns a result", False, "timed out")
            else:
                text = await page.locator(".ranking").inner_text()
                check("the ranking says what it measured", "forward passes" in text)
                check("the baseline is named", "ablation" in text, text[:60])
                check(
                    "the scores are not presented as shares",
                    "do not add up" in text,
                    "the caveat is missing",
                )
                check(
                    "the baseline's effect on the order is stated",
                    "different order" in text,
                )
                labelled = await page.evaluate(
                    """() => {
                      const sel = document.querySelectorAll('.panel.attn select')[1];
                      return [...sel.options].slice(0, 3).map(o => o.text);
                    }"""
                )
                check(
                    "the head dropdown carries the ranking",
                    all("KL" in o for o in labelled),
                    str(labelled),
                )

        print("\nshared sessions (.mri)")
        # Round-trips a session through the page the way a person does: export
        # from the panel, hand the bytes to the file input, read what appears.
        # The API tests cannot see whether the banner is legible or whether the
        # panels that cannot work in replay are still on screen offering to.
        await page.set_viewport_size({"width": 1440, "height": 900})
        await page.wait_for_timeout(200)
        opener = page.locator("button", has_text="Open a shared analysis")
        check(
            "an .mri can be opened before anything is loaded", await opener.count() == 1
        )

        can_export = await page.evaluate(
            "async () => (await fetch('/api/session/export')).status === 200"
        )
        if not can_export:
            skip(".mri round trip", "no generation on this server")
        else:
            result = await page.evaluate(
                """async () => {
                  const r = await fetch('/api/session/export?layer=0&head=0&note=probe');
                  const file = new File([await r.blob()], 'probe.mri');
                  const input = document.querySelector('input[type=file][accept=".mri"]');
                  const dt = new DataTransfer(); dt.items.add(file);
                  input.files = dt.files;
                  input.dispatchEvent(new Event('change', {bubbles: true}));
                  await new Promise(res => setTimeout(res, 900));
                  const panel = document.querySelector('.panel.replay');
                  const heads = [...document.querySelectorAll('.panel h2')]
                                  .map(h => h.textContent);
                  return {
                    banner: !!panel,
                    note: panel ? /probe/.test(panel.innerText) : false,
                    pill: [...document.querySelectorAll('.pill')]
                            .some(p => /replay/.test(p.textContent)),
                    attention: heads.some(h => /ATTENTION/.test(h)),
                    features: heads.some(h => /FEATURES/.test(h)),
                    share: [...document.querySelectorAll('button')]
                             .some(b => /Share this view/.test(b.textContent)),
                  };
                }"""
            )
            check("opening one shows the shared-session banner", result["banner"])
            check("the sender's note is shown", result["note"])
            check("the status pill says replay, not the model name", result["pill"])
            check("attention still works from the recording", result["attention"])
            # The two that would otherwise offer controls with nothing behind
            # them: features need activations, and re-exporting a recording as
            # your own run would be a lie.
            check("the features panel is not offered in replay", not result["features"])
            check("you cannot re-share a session you are viewing", not result["share"])

            closed = await page.evaluate(
                """async () => {
                  [...document.querySelectorAll('button')]
                    .find(b => b.textContent.trim() === 'Close').click();
                  await new Promise(res => setTimeout(res, 800));
                  return !document.querySelector('.panel.replay');
                }"""
            )
            check("closing it returns to the live model", closed)

        refused = await page.evaluate(
            """async () => {
              const junk = new File([new Blob(['not a session'])], 'x.mri');
              const input = document.querySelector('input[type=file][accept=".mri"]');
              const dt = new DataTransfer(); dt.items.add(junk);
              input.files = dt.files;
              input.dispatchEvent(new Event('change', {bubbles: true}));
              await new Promise(res => setTimeout(res, 700));
              const msg = document.querySelector('.session-open-row .hint.err');
              return {
                text: msg ? msg.textContent : '',
                intact: !document.querySelector('.panel.replay'),
              };
            }"""
        )
        check(
            "a file that is not a session is refused in words",
            "not a ModelMRI session" in refused["text"] and refused["intact"],
            refused["text"][:70],
        )
        check(
            "the refusal names no exception class",
            "Error" not in refused["text"],
            refused["text"][:70],
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

    print(f"\n{len(PASS)} passed, {len(FAIL)} failed, {len(SKIP)} sections skipped")
    for f in FAIL:
        print(f"  FAILED: {f}")
    # Printed last, and named, so a green run cannot be read as "the
    # head-ranking panel was checked" when nothing on this server could have
    # checked it. Load a model and prompt once before running this if you want
    # those sections: gpt2 takes about 11 seconds and lights up 14 more.
    for s in SKIP:
        print(f"  NOT CHECKED: {s}")
    return 1 if FAIL else 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
