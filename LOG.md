# Working log

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
