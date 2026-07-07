# ModelMRI

**Chrome DevTools for AI models and agents.** Load any local model — LLM, VLM, or robot policy — and see inside it while it runs: attention, features, circuits, agent steps. Local-first. MIT.

> 🚧 **Under active construction, in public.** v0.1 (live attention playground) ships in ~2 weeks.
> Follow the build: [Substack](https://modelmri.substack.com)

## Why

When a model gives a wrong answer, you can't see why. When an agent fails at step 47, you get a wall of logs. When a robot policy drops the object, you get nothing. The research tools that *can* see inside (SAEs, attention analysis, circuit tracing) live in notebooks only specialists can drive.

ModelMRI packages that research into a tool with the ergonomics of browser DevTools: one-line install, open `localhost:5900`, look inside.

## Planned (the 12-week public roadmap)

- **v0.1** — load a HuggingFace LLM, watch attention flow live at 60fps (WebGL)
- **v0.2** — SAE feature browser: see the *concepts* inside the model, steer them
- **v0.3** — agent mode: record any Anthropic-SDK/Claude-Code agent run, replay it, find the failing step
- **v0.4** — the first interactive tool for looking inside a robot policy (SmolVLA + LeRobot)
- **v0.5** — polish, zero-install hosted demo, launch

## Status

| Piece | State |
|---|---|
| Backend (FastAPI, activation hooks) | 🏗️ in progress |
| Frontend (React + WebGL) | 🏗️ in progress |
| `pip install modelmri` | placeholder published |

MIT © Muhammad Mahad Azher
