# Working log

## 2026-08-09 — 0.6.0–0.6.2: sharing a finding, and a 1.5 TB near-miss

**The `.mri` format.** `*.mri` had been sitting in `.gitignore` since week one
and was never implemented. It's the obvious missing thing: you find the head
that moves the subject token, and the only way to show anyone is a screenshot
they cannot explore. So a file that holds the observation and not the model —
tokens, attention, the generation, and a note.

Size was the whole design. A 24-layer, 14-head, 141-token attention tensor is
6.7 million numbers; as JSON at four decimals that's tens of megabytes for
something meant to be attached to a message. uint8 against each matrix's own
maximum, then gzip: a 29-token gpt2 run with **all 144 attention maps is
54 KB**. Measured against the live model, worst absolute error 0.002 and the
strongest attention in every row survives. The file states its own precision,
because a number that has quietly lost some is exactly what this project
exists to catch.

The implementation trick that made it cheap: replay is served through
`runtime.attention()`, the same method a live model uses. Nothing in any panel
changed.

**Then someone clicked GLM-5.2.** 753 billion parameters. It began downloading
**1,506.7 GB** onto a laptop with an 8.6 GB GPU and 88 GB of free disk. No
size shown, no warning, and no way to stop it except killing the server
process.

Three separate failures, all mine:

- The picker queried the Hub with `full=true`, which does **not** return
  `safetensors` — so every row came back with no size at all. Switching to
  `expand[]=safetensors` gives per-dtype parameter counts, which is how the
  same 753B model correctly reads 1.5 TB in BF16 and 756 GB in FP8. One
  number would have been a lie for the other.
- No capacity check. There is one now, shared by the HuggingFace and Ollama
  paths so they cannot drift, checked against real free space on the volume
  that download would land on. Disk refusals cannot be overridden; "too big
  for your GPU" can, with a second deliberate click. Enforced server-side —
  a check the browser performs is a check the browser can skip.
- No cancel, and this one was interesting. `from_pretrained` downloads inside
  the calling thread, and Python cannot interrupt a thread blocked in a socket
  read. So the fetch happens in a **child process** now, precisely so it can
  be terminated.

Which promptly produced its own bug: I spawned that child with `stderr=PIPE`
and nothing draining it. `huggingface_hub` writes progress to stderr, the
~64 KB pipe buffer filled, and the child blocked forever. The UI sat at
"551 MB / 551 MB · 234s · reading from local cache" with the weights fully
downloaded. Both streams go to DEVNULL now, and two tests hold it there —
verified red, one of which takes 30 seconds to fail because that is what a
deadlock does.

**Then: reading it shouldn't need the tool.** `modelmri open` worked, but it
imported torch and transformers first — **26 seconds** — to display a 54 KB
recording that needs neither. The first person to run it pressed ctrl-c
partway through, which is the correct response.

Two fixes. A browser viewer at `/viewer/`: the same React app with the API
answered from a file you drop, so a recipient reads a shared analysis with
nothing installed and nothing uploaded. And `modelmri open` now serves that
same bundle from the standard library — **0.26s warm, 0.69s from a cold fresh
install**. The split is finally clean: `modelmri serve` is the tool,
`modelmri open` is a file reader.

The format now has two implementations, so I stopped assuming they agree and
started checking: `tests/viewer_check.py` parses one file both ways and
compares every cell. 6,912 cells, identical checksums. A viewer that renders a
*slightly* different matrix would be worse than no viewer, because nothing on
screen would say so.

**Seventeen path bugs**, from an audit of code I had already tested and
shipped. They share one shape — a location computed correctly in one module
and approximately in another. `import modelmri` died outright on a container
with no resolvable home, *before* `MODELMRI_HOME` (the documented fix for
exactly that) could be read. The HuggingFace token was created world-readable
and narrowed a moment later, which on a shared host is a window. `HF_HUB_CACHE`
was ignored in four places, so the robot panel called a cached checkpoint
missing and suggested a download that landed in the directory it wasn't
reading.

The test that would have caught most of them now exists: run the whole app
inside a synthetic home and fail if any absolute path in any API response
points outside it. **It found a real bug on its first run** — `MODELMRI_HOME`
promised to relocate everything, but a surviving `~/.modelmri/traces.sqlite`
still won, so identical commands produced different storage depending on
upgrade history.

**Lesson of the week**, again: I am not a reliable auditor of code I just
wrote. Four CI failures went out before I started running CI's exact command
locally — including a POSIX-only test that skipped on my machine and failed in
CI *for the code being correct*, because it watched `hub.json` when the atomic
write opens `hub.json.<pid>.tmp`.

## 2026-08-08 (later) — bring your own model, and four bugs that were invisible

**Custom models.** Every other panel is transformer-shaped, so the honest
answer to "does this work on the model I trained?" was no, unless you'd saved
it as a HuggingFace causal LM. Now: an adapter (`def load(): return model`) or
TorchScript, and you get a layer map of one real forward pass — shapes,
activation ranges, dead units, saturation, timing, and the first layer where a
nan appears. A `state_dict` alone is refused with the reason: it's weights
without an architecture, and guessing one would produce a map that looks
authoritative and describes a network nobody trained.

Statistics exclude non-finite values on purpose. One nan propagates through
mean/std/min/max, so the naive version prints nan for every layer downstream
and hides where it started.

`tests/mutation_check.py` breaks `custom.py` twelve ways and asserts the named
test notices. 12/12. Two were survivors when first written, and both were the
test's fault: one asserted a downstream proxy that a leaked hook doesn't
disturb (the leaked hook closes over the *previous* rows list, so it appends
out of sight and the count stays right), and one mutation didn't break what it
claimed to.

**Four bugs, all invisible, all shipped for weeks.**

1. `--glass-fill: var(--glass-fill)` — a self-referential custom property is a
   cycle, so it computes to nothing and takes the whole `background`
   declaration with it. Every liquid-glass surface was fully transparent. The
   model picker was blur with no frost and the hero headline read straight
   through the model list. The owner reported it as "the background isn't
   blurred enough"; it was worse than that. Third time a var() has silently
   voided a declaration here, so there's a test for the class now.

2. The scrim behind that sheet was `blur(3px)` and was doing *all* the work —
   an element with `backdrop-filter` becomes a backdrop root for its children,
   so `.sheet`'s own `blur(40px)` only ever sampled the scrim's flat tint.

3. Keyboard focus was invisible on half the app. `:focus-visible` and
   `.model-row` are both specificity (0,1,0), so the eight `all: unset` rules
   below it won on source order; `.theme-seg button` at (0,1,1) won outright.
   19 of 20 controls in the picker moved focus with no ring — while
   `:focus-visible` matched and `outline-style` computed to `none`.

   The first probe for this reported *every* button as ringless, because
   `element.focus()` doesn't set `:focus-visible` in Chromium. That's a fact
   about the probe. I nearly "fixed" it.

4. A page reload discarded your analysis. The attention and feature panels
   were gated on a client-side counter, so refreshing unmounted them while the
   server still held attention for 141 tokens and would have served it.

Also: errors reached the screen as `Error: 422: {"error":"…"}` on all 19 paths;
the picker resized 266px under the cursor when its list landed; the footer read
`MRI-0.3` through the whole 0.4 line; the hosted demo's "On this machine" tab —
the feature whose entire point is finding your models — said "Nothing found".

**Repo hygiene.** Contributing guide, code of conduct, security policy,
support, issue/PR templates, CITATION.cff, CODEOWNERS, Dependabot, CodeQL,
changelog. SECURITY.md states the trust model plainly: local single-user, no
auth, and loading any model executes code.

**Verification.** `tests/ui_check.py` (17 browser assertions, in CI) plus
`gen_api_docs.py --check` so the API reference can't drift. The unstyled-button
check injects the bug into the live page each run and fails if it isn't
detected — a check that can't rot into a no-op.

0.5.0 built and verified from a clean install into an empty venv. Not
published yet.


## 2026-08-08 — design v5, and dark mode

Researched how Google and Apple actually build colour, motion and type
systems (Material 3 tonal palettes, HIG spring params, OKLCH ramps, and what
Linear/Vercel/Stripe do in CSS), then rebuilt the foundation rather than
adjusting hex codes. Spec: `Blueprint/08-design-system-v5.md`.

**Colour.** Six hues on one lightness schedule, chroma bounded by the real
sRGB gamut. `--color-cobalt` and `--color-attn` were 7 degrees apart -- a
duplicate, not a distinction -- so attention now IS the primary and violet
moved +9.5 to open a 32-degree gap. Neutrals disagreed (warm ground, cool
ink); light is one warm hue now, dark one cool graphite, and that temperature
flip is the theme boundary rather than an accident.

Two AA failures caught by measuring, not looking: amber at 3.76 under a
10.5px heading (now 7.22), and crimson at 4.49 which I introduced during the
rebuild (now 5.9). Every text role passes AA against every surface in both
themes.

**Dark mode.** Three states, because a binary toggle strands you off the OS
setting with no way back. The failure this was always going to have is
canvases -- CSS re-cascades free, rasterised pixels do not. Verified the hero
repaints on switch: mean pixel 530 -> 356 -> 532.

**Signature moves, each carrying data.** The section divider is a measurement
rule with ticks on the token strip's own 8px pitch. Feature rows arrive
ranked by activation (28.4 at 0ms, 27.1 at 22ms, monotonic down) because
every row is the same violet and rank is the only channel left. Numbers use
tabular figures so a streaming count does not shiver. And a 640ms specular
scan crosses a panel only when genuinely new data lands, replacing an
entrance animation that fired on every mount and therefore meant nothing.

**Two bugs testing caught that reasoning did not.** FeaturesPanel has two
return paths rendering `.panel feat`; my scan ref attached to whichever came
first in the file, which was the EMPTY state, so the panel that shows data
never scanned. And a MutationObserver silently watched stale nodes across a
remount and reported a false negative -- polling the live DOM showed both
panels firing correctly.

Also: nested backdrop-filter double-blurring the same pixels, disabled
buttons dissolving to 0.35 so the ground showed through, a steering slider
Firefox never drew, all seven `transition: all` (worst on ~256 token chips),
hover transforms latching on touch, and the `!important` reduced-motion nuke
that breaks transitionend.

Colour literals in the rules: 89 -> 0. 52 unit tests, 42 e2e checks.

## 2026-08-07 (night, last) — Gemma runs; the gate check was wrong twice

Chasing the one model explicitly asked for turned up two bugs in a row, both
mine, both the same failure mode: *the observable result matched what I
expected, so I stopped looking.*

**Gemma-3-270m-it, verified on GPU:** 268M params, bf16, cuda:0, cold load
151.5s (575 MB download), generation `"The Eiffel Tower is located in Paris,
France."`, attention **18L x 4H**, 31 tokens, rows sum to 1.000, causal mask
holds. SAE correctly declined (GPT-2 only). That is 5 of the 6 current open
models now proven end-to-end.

**Bug 1 — a token is not a licence.** `hub.search` computed
`usable = (not gated) or bool(token)`. Gating is *per-repo* acceptance, so
every Gemma and Llama build was shown as available to an account that had
accepted neither. The loader refused them with a good error, but the picker
had already promised them. I had reported this as a working feature earlier
in the night — "search llama returns gated models showing gated OK with 0
locked rows" — which was the bug rendering, read as a success.

**Bug 2 — the fix for bug 1.** `_has_access` routed through `_api()`, which
does `json.load()`. The auth-check endpoint answers **200 with an empty
body**, so it raised, the `except` swallowed it, and the function returned
False for *every* repo. It passed a live test only because every gated repo
on hand was one this account genuinely could not reach — right answer, wrong
reason. Anyone who *had* accepted a licence would have seen their model
marked locked. Caught by probing the endpoint directly and noticing `gpt2`
answers HTTP 200 while the function said False.

The final proof is the discrimination, not the pass: on one fresh server,
`google/gemma-3-270m-it` reports gated **and usable** (and does run), while
`meta-llama/Llama-3.2-1B` reports gated **and not usable** (and 403s). Two
gated repos, opposite answers, both correct.

Locked rows now open the model's Hub page instead of doing nothing, and say
which step is missing: "sign in" when signed out, "accept licence" when
signed in.

**Test lesson worth keeping:** the first test could never have caught bug 2 —
it monkeypatched `_has_access`, the very function under test. The replacement
drives the real function against a fake empty 200. Both new tests were run
against reverted code first to confirm they fail without the fix. A test that
passes either way is not evidence.

**Still blocked, honestly:** OLMo-2-1B stalls at exactly 134 MB of 2.98 GB
across four attempts, on both HF transports and to a local disk. Not our bug;
the stall detector is what proved that rather than guessed it.

47 tests, 42 e2e checks.

## 2026-08-07 (night, later) — an audit against the field, and the bar that painted nothing

Ran 4 research agents over Apple's Liquid Glass spec, the interaction craft of
Linear/Raycast/Vercel/Stripe, the WAI-ARIA dialog pattern, and inspector UIs
(DevTools, Perfetto, BertViz). Then audited this frontend against that standard
across 5 dimensions, with an adversarial pass whose job was to *refute* each
finding. 13 survived, 12 were killed. Only the survivors were acted on.

**The worst finding was mine, from three hours earlier.** `--color-accent` does
not exist — the palette name is `--color-cobalt`. An undefined `var()`
invalidates the whole declaration, so the load-progress fill fell back to
transparent. I had verified that bar by reading its *width*. Width was never
the question. Fixed, and confirmed this time by reading computed
`backgroundImage`.

The same bug was already in the codebase: `--model` doesn't exist either, and
`ArcCanvas` read it for the arc stroke. Canvas ignores an unparseable
`strokeStyle`, so the attention arcs had been drawing in **default black**
instead of the attention blue since they were written. Now measured: 6,115
pixels of `rgb(26,96,209)`.

**Correctness**
- `ws/generate` had `try/finally` with no `except`. A generation that raised
  died in the worker thread, the `finally` posted the sentinel, and the browser
  was told `"done"` — CUDA OOM and unsupported architectures arrived as
  successful *empty answers*. The new test was run against the reverted code
  first to confirm it actually fails without the fix.
- A failed Hub search left the picker on "searching…" forever (`models === null`
  is the loading sentinel and the catch never cleared it).
- The attention panel had no loading or failure state, despite a first fetch
  that runs a full `output_attentions` pass.

**Access** — keyboard focus was invisible app-wide: `all: unset` on `.model-row`
resets `outline-style` in the *author* origin, which outranks the UA focus ring,
and nothing defined a replacement. Two component `outline: none` rules at
(0,2,0) were then quietly beating the new global rule; they now suppress the
ring for pointer focus only. The picker sheet was a plain div — now a real
dialog (role, `aria-modal`, initial focus, Tab trap, scroll lock, focus restored
to the opener). Attention and feature chips were pointer-only spans, so the arc
view and the whole SAE workflow had **no keyboard path at all**.

**Craft** — `ArcCanvas` drew at 1 device pixel per CSS pixel; thin arcs are the
panel's entire payload and were being upscaled on every retina screen. The
feature panel discarded the `argmax` the API already returns, so in a 256-token
strip the one chip worth seeing was unfindable. "sign out" composited to 3.32:1
through `opacity`, under AA, and it is the only way to sign out.

**Two bugs in my own fix**, found by testing rather than reasoning: React's
`autoFocus` fires during commit, so the effect captured the *search input* as
"the opener" and Esc restored focus to a dead node (body). And `onClose` is an
inline arrow, so depending on it re-ran the modal effect on every parent render
and would yank focus mid-keystroke.

**And one found by simply looking at the panel** rather than its numbers: the
attention strip could grow but never shrink. `.attn-inner` is
`width: max-content` and the canvas is its widest child, so measuring
`row.scrollWidth` while the canvas still held the previous generation's width
just returned that width again. A 23-token generation was rendering into a
12,645px box. Reproduced the sequence, fixed, re-verified: 267 tokens
(11,207px) → 5 tokens (741px), arcs still painting.

44 unit tests, 42 e2e checks, every visual claim checked against computed
styles or canvas pixels.

## 2026-08-07 (night) — the model picker meets real models

Ran five current open models through the actual HTTP API on the 4060, and the
run found more than the code review did.

**Verified on GPU (bf16, cuda:0), not predicted:**

| model | params | load | shape | attention rows | causal |
|---|---|---|---|---|---|
| Qwen/Qwen3-0.6B | 596M | 10.0s | 28L x 16H | 1.001 | yes |
| Qwen/Qwen2.5-0.5B-Instruct | 494M | 7.9s | 24L x 14H | 1.000 | yes |
| HuggingFaceTB/SmolLM2-360M-Instruct | 362M | 6.1s | 32L x 15H | 1.002 | yes |
| gpt2 | 124M | — | 12L x 12H | — | yes (+ SAE, steering) |

OLMo-2-1B is untested: its download stalls from this network on both the xet
and plain transports, and a probe straight to a local disk stalls identically.
Not our bug — but the stall detector below is what proved that, instead of me
guessing.

**What the run surfaced:**

- **Silent loads.** Minutes of nothing but the word "loading". `/api/model/progress`
  now reports stage + real bytes. Determinate bar when the size is known,
  indeterminate sweep when it isn't — a fake percentage is worse than none.
- **Byte counting is not obvious.** Three cache layouts exist and I hit all
  three; take the max of `blobs/` and `snapshots/`, never the sum. And size the
  download from a whitelist of what `from_pretrained` fetches — blacklisting odd
  formats left a fully-cached gpt2 reporting 26%, because gpt2 ships tflite,
  rust, h5 and flax copies of itself.
- **Dead downloads don't raise, they just stop.** Watched one sit at 128 MB of
  3.0 GB indefinitely. Called out after 45s now.
- **Qwen3 leaks `<think>` into the output.** Reasoning models stream a
  scratchpad. It gets its own collapsible block — on an introspection tool the
  model's working is the point, so hiding it would be the wrong fix.
- **The page scrolled sideways to 7859px.** `main` is a grid; grid items default
  to `min-width:auto` (= min-content), so a panel holding a 194-token attention
  strip grew to 7813px and dragged the whole layout with it. `min-width:0`.
  Every generation of any length hit this.
- **The picker forgot which model was loaded** across a reload, so Generate
  silently swapped models. It adopts the live one now.

**Environment, the hard way:** DriveFS truncated `typescript/package.json` to
zero bytes and refuses junctions, so `node_modules` cannot live beside the
source here. `scripts/build_frontend.py --work C:/build/modelmri` builds off the
synced drive — `npm ci` is 3s there against minutes on J:.

Two e2e checks "failed" until I found the cause: a uvicorn started before the
edit was still holding :5900, and my kill filter had missed it. Kill by port,
then confirm the new route answers, before believing any e2e result.

42 unit tests, 42 e2e checks, browser-confirmed at 194 tokens.

## 2026-08-07 (late) — v0.4 verification + hosted demo
- **`tests/e2e_check.py`**: exercises every feature against a live server (real models, real SAEs, real robot frames). **40/40 pass** — session/static/no-cache header/bundle, model discovery incl. Ollama-off path, load+generate, attention (rows sum to 1.000) + 422s, SAE load/features/steer/restore-exactly, traces import/list/get/404/422, VLA episodes/frame/load/analyse/heatmaps + "sharpens with depth" assertion. Run before every release.
- **Hosted demo shipped**: `scripts/bake_demo.py` captures real responses from a live server into `frontend/public/demo/*.json` (70 KB total); `VITE_DEMO=1` builds a static bundle whose `fetch` is patched once in main.tsx to serve those payloads, so every call site is identical to the real app and the demo can't drift. WS streaming is replayed word-by-word. `.github/workflows/pages.yml` deploys to GitHub Pages.
- **Bug caught by testing the demo rather than assuming:** the steering A/B rendered baseline text in BOTH cards — the demo prompt handler ignored steering state. Fixed by mirroring `steerActive` in demo.ts. Would have shipped a demo whose headline feature visibly does nothing.
- Demo verified in a browser against the static bundle: generation streams, 23 token chips, 2589 arc pixels, feature rows (#974 @ 60.9), A/B now differs (Paris -> San Diego), VLA frame + 153-px heatmap, layer slider changes the map (13k -> 34k at L9), scrubber returns correct frames (ask 54 -> t=54 @ 5.4s).
- **Perf finding:** cold gpt2 load measured 523s, reload 3.7s. Cause is the Drive-backed cache: J: is Google Drive File Stream, so a 1.1 GB model is *streamed from the cloud* on first touch, then cached locally. Not a product defect (a normal user's HF cache is on local disk) but a real consequence of the storage move — worth knowing before demoing cold.

## 2026-08-07 (late) — v0.4 VLA MODE: inside a real robot policy
- Recon workflow (4 parallel scouts) verified everything empirically before a line was written: AV1 decode 0.02-0.05s, PushT parquet 1.4MB/0.74s, and the blocker — **lerobot pins torch<2.12/numpy<2.3 but the venv runs torch 2.12.1/numpy 2.5.1**, so installing it would downgrade the working LLM path.
- Decision: **read the dataset directly** (pyarrow + pyav, no lerobot) and **lift the vision tower straight out of `lerobot/smolvla_base/model.safetensors`** — 197 tensors under `model.vlm_with_expert.vlm.model.vision_model.`, loaded into `SmolVLMVisionTransformer` with 0 missing. These are the policy's real weights, not a stand-in.
- Gotchas paid for: PushT's cache ref is `v3.0` (assuming `main` breaks discovery); dataset lives under `$HF_HOME/lerobot/hub`, models under `$HF_HOME/hub`; sdpa silently returns `attentions=None` → must force `_attn_implementation="eager"`; raw attention is [1,12,1024,1024] ≈50MB/layer → reduce to per-head 32×32 inside `no_grad`.
- Shipped: `vla_data.py` (episodes/state/action/frames), `vla.py` (VLAHandle), 7 endpoints, `VLAPanel` + `FrameCanvas` (scrubber + heat overlay + layer slider + stale badge), `vla-lite` extra. 25 tests.
- **Verified numbers:** 206 episodes · frame decode 60ms · tower load 5.2s · analysis 1.7s · heatmap paints 156 samples in the UI · **attention concentration rises with depth (top-5% mass 27% → 56% → 60% across layers 0/6/11)** — the expected diffuse→focused pattern, measured on real robot frames.
- Honest scope: this is the perception half. The action expert needs lerobot in a separate venv (`full` mode, designed, not built). The UI says so.

## 2026-08-07 (night) — blank-page bug + verification lesson
- Owner reported the whole app blank (only the header pill rendered). Cause: `AsciiField` measured its PARENT then wrote its OWN style size — canvas grows → hero grows → ResizeObserver refires → unbounded loop. Page inflated to thousands of px of empty canvas; all panels pushed off-screen.
- Fix: CSS owns the canvas box; JS only syncs the pixel buffer (no style writes, no-op when unchanged), observes the canvas itself, repaints on resize.
- **Verification lesson (permanent):** DOM-presence checks are not visibility checks. From now on, UI verification must assert LAYOUT — document.body.scrollHeight is sane, key elements' getBoundingClientRect().top is inside the first viewport, canvas box has a fixed expected height. That is how a blank page slipped past "all panels present".
- Verified after fix: scrollHeight 1283 (was runaway), canvas 200px stable, 900 glyphs painted, headline at y=319, Generate at y=687.

## 2026-08-07 (evening) — v4 "VANTAGE PAPER" + one-click Generate + any-model support
- **Root-caused the "Generate isn't working" report:** after any server restart the model unloads, leaving a silently disabled CTA. Fixed properly: Generate now auto-loads the selected model first (status shown), then streams. Verified from a cold server: one click → auto-load → 1,073 chars streamed.
- **Design v4 "Vantage Paper"** — LIGHT theme, straight from the owner's saved poistudio VANTAGE recipe: warm paper #f6f4ee, cobalt #2743e0, white hairline plates, centered editorial headline, the ASCII field recolored cobalt-on-paper, and the actual **Switzer** variable font the recipe names (self-hosted from Fontshare, license bundled).
- **Any model, two sources:** HF combo input with datalist fed by curated + a new local-cache scanner (GET /api/models/local — found all 10 cached models incl. VGGT/SAM3/SmolVLA) — plus **Ollama mode** (GET /api/ollama, load source="ollama", NDJSON streaming via stdlib): run any open model as text; UI states clearly that internals need HF. Graceful when Ollama is off.
- **release.yml workflow:** tag → build frontend into wheel → verify assets inside → attach to GH release.
- 19 tests. Django/Postgres formally dropped per owner (concept fit).

## 2026-08-07 (later) — CATCH-UP SPRINT: design v3 + Agent Mode v0.3
- **Design v3 "editorial scanner"**, grounded in the owner's actual X bookmarks (viewed via his Chrome): poistudio ASCII-dither art, brrranding condensed wordmarks, magenta-on-black pixel craft, Swiss spec labels. Tailwind CSS v4 (CSS-first), bundled Archivo Black, flat hairline plates, one electric magenta, and the signature `AsciiField` — a live ASCII-dither canvas (10fps, frame-1 sync paint, reduced-motion static). Stack note: kept FastAPI+SQLite over requested Django/Postgres — local-first pip install is the product; documented rationale.
- **Agent Mode (v0.3)**: `modelmri/traces.py` (SQLite WAL store, trace/step schema, ~/.modelmri/traces.sqlite), `modelmri.record` subpackage (trace ctx, nesting steps via contextvars, instrument_anthropic monkeypatch, POST-or-file delivery, never crashes host), endpoints (import/list/get), AgentsPanel UI (trace list, lane timeline colored by kind, error glow, step inspector w/ IN/OUT). `examples/record_demo.py` ships a realistic 17.1s failing run.
- Tests 15/15. Browser-verified: 10 blocks / 2 lanes render; clicking the error block shows "git push · step 8 · FAILED · Permission denied (publickey)" — kill-demo 3 is real.
- To extract at release: `modelmri-record` as its own PyPI dist (reserve name!).

## 2026-08-07 — Back after a month. Full feature audit, two hangs made impossible, lockfile added.
- Owner reported "no generated answer visible, only attention stats." Root causes found and fixed:
  1. `index.html` served with no cache headers while each deploy PURGES old hashed bundles → a stale cached page half-breaks. Fixed: `Cache-Control: no-cache, must-revalidate` on `/` (verified in response headers).
  2. A WS generation observed hanging forever with zero tokens (streamer blocks if the generate worker dies). Two-sided hardening: `TextIteratorStreamer(timeout=180)` server-side; 90s no-token watchdog + onclose handling client-side — the UI can no longer spin forever.
- **Lockfile lesson:** a month of dependency drift happened silently because `uv.lock` was gitignored (bad week-0 call). Reversed: `uv.lock` committed (78 packages pinned).
- Environment survived the break: registry env vars now inherited by new sessions (UV_PROJECT_ENVIRONMENT, HF_HOME), HF symlink intact, venv rebuilt in one `uv sync`.
- FULL browser audit, all green: generation output visible (257 pieces · 9.1s · 1,055 chars), attention arcs paint (4,305 px on pin), SAE loads (24,576 feats), token→features works (·Paris → **#974 @ 60.9 — the same feature as July**), heat view (267 chips), steering A/B reproduces the kill demo (" Paris, France." → " San Diego…" at #974 @ -40, reversible).

## 2026-07-08 — Week 1, day 6: STORAGE MIGRATION + design system v2 "scanner glass"
- Everything moved off C: per owner request: repo now at `J:\My Drive\Claude_Experiments\special\ModelMRI`, HF model cache (21.8 GB incl. other projects' models) at `special\models\huggingface`. **C: freed 20.7 GB** (72.6 → 93.3).
- DriveFS lessons (now standing knowledge): junctions/symlinks CANNOT be created on DriveFS; npm .bin shims break on it → venv lives at `C:\venvs\modelmri` (UV_PROJECT_ENVIRONMENT, registry-persisted), frontend builds in `C:\venvs\mri-build` temp and deploys back to J:. Old default HF path symlinked → J: so every process resolves models without env vars.
- Verified end-to-end from new locations: 11/11 tests, gpt2 loads from J: cache in 9.8s (no re-download), server serves the app.
- Design system v2: aurora ground (drifting radial washes + masked dot grid), orbiting scanner mark, gradient-ink wordmark, gradient-border glass panels (per-section tint), segmented model picker, shine-sweep buttons, custom violet slider, feature bars with grow-in, section headers with glow dots + fading rules. All motion on one easing; reduced-motion kills everything.
- Verified: computed styles confirm glass (blur 22px sat 1.5) + mark live; page interactive. (Preview screenshotter times out on infinite animations — page itself healthy; owner should eyeball localhost:5900.)

## 2026-07-08 — Week 1, day 5: FEATURES UI + liquid-glass design system. "We moved the Eiffel Tower to Berlin."
- Design system v1: liquid glass (backdrop-blur panels, layered radial-glow background, inset borders), per-section palettes (teal brand / blue attention / violet features), 200-300ms eased micro-animations, reduced-motion respected. Bar: design.google / Apple.
- FeaturesPanel shipped: model picker (Qwen chat / GPT-2 SAE) → generate → load SAE → click any token → its top-8 features with bars → click a feature → per-token heat view → steering slider → one-click deterministic A/B with side-by-side glass cards (always leaves steering cleared).
- Browser-verified end to end: GPT-2 sampled an Eiffel-in-Berlin hallucination; clicked "·Berlin", top feature #12884 (51.0); steered +40 → baseline " Paris, France." vs steered " Berlin, Germany." — amplifying the Berlin concept relocates the tower. Screenshot taken.
- Standing rules recorded (Blueprint/06): always share the localhost URL; Chrome posting on request; Gemini Pro (Nano Banana Pro / Veo) for premium assets; premium design bar.
- Next: GIF-ready polish + agent mode (v0.3), or Gemini-generated brand assets for README.

## 2026-07-08 — Week 1, day 4: SAE FEATURES + STEERING (backend). We turned off "Paris".
- New `modelmri/saes.py`: loads SAELens-format SAEs straight from HF (cfg.json + safetensors) — no sae-lens dependency chain. Default: jbloom/GPT2-Small-SAEs-Reformatted @ blocks.8.hook_resid_pre (24,576 features).
- Runtime: chat-template fallback for base models (GPT-2 has none), residual capture via forward_pre_hook, per-token feature computation (cached), single-feature steering (adds scale × unit decoder direction to the residual stream during generation, hook removed in finally).
- Endpoints: POST /api/sae/load, GET /api/sae, GET /api/features/summary, GET /api/features/{id}, POST/GET /api/steer. 11 tests green.
- VERIFIED END-TO-END (all real numbers):
  - Features are consistent: feature 1066 fires on both " Tower" occurrences, 19941 on both " E"s, 974 on " Paris" (60.9), 7310 on " France" (56.0).
  - THE steering A/B: baseline greedy → " Paris, France." · steer 974 at -40 → " San Diego, and is located in the San Diego State University" · clear → byte-identical " Paris, France." Deterministic, reversible, mechanistic.
- Next: FeaturesPanel in the React frontend (token → top features → steering slider → side-by-side steered output).

## 2026-07-08 — Week 1, day 3 (later): v0.1.0 RELEASE PREP + Day-3 post live
- Day-3 X post published (the "Paris attends to capital/France" find).
- Version bumped 0.1.0a1 → 0.1.0. README gains the pip install path.
- RELEASE-KILLING BUG caught by verification: the wheel had ZERO frontend assets — hatchling skips VCS-ignored files and `modelmri/static/app/` is gitignored. Fixed with `force-include` on both wheel and sdist (sdist matters: uv builds the wheel FROM it). Anyone pip-installing would have gotten a backend with no UI. Verify-before-ship pays again.
- Full release gate passed: wheel contains index.html + JS + CSS; clean-venv install → `modelmri 0.1.0` → server up → root serves the React app → assets 200.
- Tagged v0.1.0 + GitHub release. PyPI publish awaits the token (user action).

## 2026-07-08 — Week 1, day 3: REACT FRONTEND
- Real frontend shipped: React 18 + Vite + TypeScript (strict), no component libs, no state libs — 5 components, 150KB JS (49KB gz), builds in <1s.
- `npm run build` emits into `modelmri/static/app/`; FastAPI serves it at `/` (falls back to the legacy single-file page when no build exists). Built assets are NOT committed — built at release time.
- Dev loop: `npm run dev` on :5173 proxies /api + /ws to the Python backend on :5900.
- Verified in a real browser (automated): loaded model → streamed 51 pieces in 3.9s → attention panel appeared (24L × 14H × 90 tokens, head fetch 0.26s) → pinned the generated "·blue" token → arcs rendered (thick short-range + long-range sweep to early context). Screenshot taken.
- Divergence from blueprint: skipped tailwind + zustand for now — plain CSS on the established palette and lifted useState are simpler at this scale. Revisit when Agent Mode adds cross-cutting state.
- Next: WebGL grid/fabric view OR v0.1.0 PyPI release + GIF. Release first — ship what works.

## 2026-07-08 — Week 1, day 2: ATTENTION IS VISIBLE
- Attention capture shipped: model now loads with `attn_implementation="eager"` (SDPA/flash never materializes attention weights — the day's big lesson).
- After any generation, one full forward pass with `output_attentions=True` caches all layers (fp16, CPU); `GET /api/attention?layer=&head=` serves any head's S×S matrix instantly.
- Playground grew an attention inspector: token chips, layer/head selectors, hover → Canvas2D arcs to attended tokens (thickness = weight), click to pin. WebGL comes with the React frontend when we render full head grids.
- Verified with real numbers: 24 layers × 14 heads × 45 tokens, softmax rows sum to 1.000; at L12/H7 the generated " Paris" token attends to " capital" (0.098) and " France" (0.064 + 0.044) — the fig-3 demo moment, real.
- Also observed: massive attention sink on <|im_start|>/<|im_end|> — classic, and now *visible*.
- Next: record the GIF, then pip-installable v0.1 polish.

## 2026-07-08 — Week 1, day 1
- Backend v0.1 skeleton LIVE: `modelmri serve` → FastAPI on :5900.
- `ModelRuntime`: loads any HF causal LM (default Qwen2.5-0.5B-Instruct, 494M params), streams generation via `TextIteratorStreamer` in a worker thread.
- REST: `/api/session`, `/api/model/load`, `/api/model/prompt`. WS: `/ws/generate` (verified: 41 pieces streamed end-to-end).
- Built-in dark playground page at `/` (temporary until React frontend).
- Bug found by smoke test: Windows cp1252 console can't print `→` — crashed the CLI banner. Fixed to ASCII. (Good Day-2 post material.)
- Note: PyPI torch on Windows is CPU-only; 0.5B is snappy anyway (full generation in 2.5s). GPU via cu124 index or WSL2 when needed.
- Next: PyTorch forward hooks on attention layers → stream weights → WebGL arcs.

## 2026-07-08 — Week 0
- Name decided: **ModelMRI** ("an MRI machine for AI models"). Verified free on PyPI + npm.
- Repo created, skeleton committed: README with public roadmap, MIT license, CI (ruff + pytest + frontend build), Python package stub, npm stub.
- Dev environment confirmed: Python 3.12, uv, node 26, WSL2, RTX 4060.
- Next: publish name-reserving stubs to PyPI + npm, Substack setup, essay #1, Day-1 X post.
