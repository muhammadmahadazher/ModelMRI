# Working log

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
