# Working log

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
