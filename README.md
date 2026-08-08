# ModelMRI

**Chrome DevTools for AI models and agents.** Load any local model — LLM, VLM, or robot policy — and see inside it while it runs: what it attended to, which concepts fired, what happens when you turn one off, and exactly where your agent went wrong.

Local-first. No cloud, no account, no telemetry. MIT.

**[▶ Try the live demo](https://muhammadmahadazher.github.io/ModelMRI/)** — no install, no GPU, real recorded output.
**[📖 Docs](https://muhammadmahadazher.github.io/ModelMRI/docs/)**

<p align="center">
  <img src="docs/media/attention.gif" alt="Hovering tokens; attention arcs follow the cursor across the strip" width="800">
</p>

<p align="center"><em>Hover any token — arcs show what it attended to. Every layer, every head.</em></p>

```bash
pip install modelmri
modelmri serve          # open http://localhost:5900
```

<p align="center">
  <img src="docs/media/picker.gif" alt="The model picker listing models already on disk" width="800">
</p>

<p align="center"><em>It finds the models you already have — HF cache, plain folders, GGUF — before asking you to type anything.</em></p>

---

## What you can actually do with it

### 1. See what a token attended to

Type a prompt, watch it stream, then hover any token — arcs show which earlier tokens it looked at, scaled by attention weight, for any layer and head.

> On GPT-2, the generated token `" Paris"` attends back to `" capital"` and `" France"`. The information was always there. Nobody was looking.

### 2. Find a concept and turn it off

Load a sparse autoencoder and ModelMRI shows the human-interpretable features firing on every token. Click one, drag the slider, and run a deterministic A/B:

```
prompt   The Eiffel Tower is located in the city of
baseline  Paris, France.
feature #974 @ -40   San Diego, and is located in the San Diego State University
```

Same prompt, greedy decoding, no prompt tricks. We reached into layer 8 and turned the concept down. Clearing the steer restores the baseline byte-for-byte.

### 3. Find the step where your agent died

Two lines of `modelmri.record` around any agent run gives you a timeline: LLM calls, tool calls, subagents, each as a block. The failure glows. Click it for the exact input, output, tokens, and error.

```python
from modelmri.record import trace, step

with trace("fix-failing-tests"):
    step("llm_call", name="plan", input=prompt, output=answer, tokens_in=912)
    with step("subagent", name="auth-fixer"):
        step("tool_call", name="pytest", output="17 passed")
```

Or instrument automatically: `modelmri.record.instrument_anthropic()`.

### 4. Look inside a robot policy

This is the part nobody else ships. ModelMRI loads the **vision tower of the real SmolVLA checkpoint** and runs actual robot-camera frames through it, painting each image patch's attention back onto the frame. Scrub an episode, run the policy, drag the layer slider.

Measured on PushT frames — share of attention mass in the top 5% of patches:

| vision layer | concentration |
|---|---|
| 0 | 27% |
| 6 | 56% |
| 11 | 60% |

Early layers look everywhere; deep layers lock on. No robot hardware required — it reads public LeRobot datasets straight from disk.

---

## Install

```bash
pip install modelmri              # core: playground, attention, features, steering, agents
pip install "modelmri[vla-lite]"  # + robot datasets (av, pyarrow, pillow)
pip install modelmri-record       # just the agent recorder — stdlib only, 7 KiB
modelmri serve
```

From source:

```bash
git clone https://github.com/muhammadmahadazher/ModelMRI && cd ModelMRI
cd frontend && npm ci && npm run build && cd ..
uv sync && uv run modelmri serve
```

**Models.** Type any HuggingFace id, pick from what's already cached on your machine, or switch to **Ollama** to run any open model you've pulled. (Ollama gives you text only — internals need a HuggingFace model, and ModelMRI says so rather than pretending.)

Everything runs on CPU. A 0.5B model streams in a couple of seconds on a laptop.

## API

The UI is a client of a plain HTTP API — script against it directly.

| | |
|---|---|
| `POST /api/model/load` | `{hf_id, source}` — `"hf"` or `"ollama"` |
| `WS /ws/generate` | stream tokens |
| `GET /api/attention` | `?layer=&head=` → tokens + attention matrix |
| `POST /api/sae/load` · `GET /api/features/summary` | SAE features per token |
| `POST /api/steer` | `{feature_id, scale}` — clamp a concept during generation |
| `POST /api/traces/import` · `GET /api/traces/{id}` | agent traces |
| `POST /api/vla/analyse` · `GET /api/vla/attention` | robot-policy attention |

## Status

| | |
|---|---|
| Playground · streaming · any local model · Ollama | ✅ |
| Attention inspector | ✅ |
| SAE feature browser + activation steering | ✅ |
| Agent trace timeline + step inspector | ✅ |
| Robot policy (VLA) attention over real episodes | ✅ perception |
| VLA action expert (needs `lerobot`, separate env) | 🏗️ |
| Hosted zero-install demo | 🏗️ |

## Honest limits

- **Attention needs eager attention.** SDPA and FlashAttention never materialize the weights, so ModelMRI loads models with `attn_implementation="eager"`. Slower, but it's the only way to see anything.
- **SAE features need a matching SAE.** Ships pointed at the public GPT-2 SAEs; other models need their own.
- **VLA mode is the perception half.** SmolVLA's vision tower is real and loaded from the real checkpoint; the action expert needs `lerobot`, whose torch/numpy pins conflict with the core runtime, so it lives behind an opt-in extra rather than degrading everyone's install.

## Built in public

Notes, mistakes, and what broke: [modelmri.substack.com](https://modelmri.substack.com)

MIT © Muhammad Mahad Azher
