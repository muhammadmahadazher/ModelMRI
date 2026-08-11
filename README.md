<h1 align="center">ModelMRI</h1>

<p align="center"><strong>Chrome DevTools for AI models and agents.</strong></p>

<p align="center">
  <a href="https://pypi.org/project/modelmri/"><img src="https://img.shields.io/pypi/v/modelmri?color=2563eb&label=pypi" alt="PyPI version"></a>
  <a href="https://pypi.org/project/modelmri/"><img src="https://img.shields.io/pypi/dm/modelmri?color=2563eb&label=downloads" alt="PyPI downloads"></a>
  <a href="https://pypi.org/project/modelmri/"><img src="https://img.shields.io/pypi/pyversions/modelmri" alt="Python versions"></a>
  <a href="https://github.com/muhammadmahadazher/ModelMRI/actions/workflows/ci.yml"><img src="https://github.com/muhammadmahadazher/ModelMRI/actions/workflows/ci.yml/badge.svg" alt="CI"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-MIT-green" alt="MIT licence"></a>
</p>

<p align="center">
  <a href="https://muhammadmahadazher.github.io/ModelMRI/"><b>▶ Live demo</b></a> ·
  <a href="https://muhammadmahadazher.github.io/ModelMRI/viewer/"><b>Open a .mri</b></a> ·
  <a href="https://muhammadmahadazher.github.io/ModelMRI/docs/"><b>Docs</b></a> ·
  <a href="https://modelmri.substack.com"><b>Build log</b></a>
</p>

Load any local model — LLM, VLM, or robot policy — and see inside it while it runs: what it attended to, which concepts fired, what happens when you turn one off, and exactly where your agent went wrong.

**ModelMRI is an open-source, local-first interpretability and debugging tool
for transformer language models, vision-language models, robot policies and
LLM agents.** It visualizes per-layer, per-head attention weights from a live
forward pass, ranks attention heads by causal ablation scored with KL
divergence, decomposes the residual stream with sparse autoencoders, steers
generation along a feature direction, maps activations in any custom
`nn.Module`, and records agent runs as an inspectable timeline. It runs on
your own machine — no cloud, no account, no telemetry — and writes findings
to a `.mri` file a colleague can open in a browser with nothing installed.

Python 3.10+, Windows / macOS / Linux, MIT licensed.

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

### 2. Ask which heads actually mattered

144 heat maps and no reason to open any of them is a browsing tool. **Rank heads** zeroes each head in a layer, runs the model again, and measures how far the answer moves — so the dropdown arrives ordered and the top head is already selected.

```
gpt2 · "The capital of France is" · zero-ablation · bf16

Rank heads → L0 H7  KL 0.898   p(" the") 0.098 → 0.057
             L0 H10 KL 0.535
             L0 H9  KL 0.412
```

The setup line is not decoration. The same three heads on the same model score
0.784 / 0.543 / 0.415 in fp32 and 0.825 / 0.559 / 0.469 over a 261-token
generation — a KL depends on the prompt, the dtype and the sequence, so a
figure quoted without them cannot be checked by anyone.

A ranking costs `n_heads + 2` forward passes; the whole model costs `n_layers × n_heads + 2`. That is the part that is portable — gpt2 is 146 passes, Qwen3-0.6B is 450. What a pass costs on *your* machine is not: measured on one RTX 4060 across sessions it moved between 12 and 71 ms for the same model, so the panel measures a layer on your machine and extrapolates from that rather than quoting a number from mine. One layer by default; the whole model only when told, with the estimate shown first.

Then ask **what changes?** on any ranked head and the panel subtracts the two runs — arcs in one colour where the model attends *more* without that head, another where it attends *less*. It opens at layer L+1, because removing a head cannot change its own layer's attention (that layer's input is unchanged), and a zero result says so rather than showing you an empty canvas.

Both sides are forward passes over the **same token sequence**, never two generations — sampling diverges, and chat templates insert 0, 8 or 29 leading tokens depending on the model, so subtracting two generations would align token 5 of one against token 5 of a different sentence.

It reports what it measured and nothing more. These are **not** each head's share of the prediction — on gpt2 layer 0 the twelve per-head scores sum to 1.995 while ablating the whole layer gives 0.208 — and the ranking depends on what a removed head is replaced with, so the baseline is named on screen and both are offered. `head_dim` is read from the model rather than computed as `hidden_size // n_heads`, which is wrong by 2× on Qwen3 and would rank half-heads confidently.

### 3. Ask where in the model the answer is decided

Ablation says *what mattered*. It cannot say *where the thing is*. **Patching** takes two prompts that differ in one fact, moves an activation from the run that knows the answer into the run that does not, at every (layer, position), and reports how much of the difference comes back.

<p align="center">
  <img src="docs/media/patching.gif" alt="The patching grid filling in row by row, then three tabs — residual stream, attention, MLP — each showing a different map of the same prompt" width="800">
</p>

<p align="center"><em>Blue recovered the clean answer, red pushed it further away. Ringed cells were tested against chance.</em></p>

```
clean    "The Eiffel Tower is located in the city of"   ->  " Paris"
corrupt  "The Colosseum is located in the city of"      ->  " Rome"
```

Three grids, because *where* and *through what* are different questions — and they disagree. Measured on the same pair across three architectures:

| model | residual | attention | MLP |
|---|---|---|---|
| `gpt2` | +0.844 · L11 · `of` | +0.232 · L9 · `of` | **+0.365 · L0 · `um`** |
| `Qwen2.5-0.5B-Instruct` | +0.999 · L23 · `of` | +0.478 · L21 · `of` | **+0.721 · L0 · `os`** |
| `gemma-3-270m-it` | +1.010 · L17 · `of` | +0.736 · L12 · `of` | **+0.483 · L3 · `osseum`** |

The MLP peak sits on a **subject** token in an early layer in all three — `um`, `os` and `osseum` are all pieces of "Colosseum" — while the attention peak sits on the **last** token, late. Early MLP writes the fact; late attention carries it to where the prediction is made. The residual grid contains both and shows only the destination.

The score is **signed**, and it is the one ranking here that is not a KL: a patch can push the answer further away, and 5 of 132 sites did. It is also not capped at 1.0 — a single site can overshoot, and `gemma-3-270m-it` reads 1.010.

Each of the strongest sites is run again against **eight** same-norm random draws at the same site, not one, because one is a coin flip: at a single site the draws ran from −2.038 to +0.616 against a real recovery of +0.427.

Most casually-written pairs are refused, and both failures are invisible unless you are told — the prompts must tokenize to the same length (2 of 8 natural minimal pairs did not) and must predict different tokens (2 of 3 did not, making the denominator exactly 0). Both refusals print what to change.

### 4. Find a concept and turn it off

Load a sparse autoencoder and ModelMRI shows the human-interpretable features firing on every token. Click one, drag the slider, and run a deterministic A/B:

```
gpt2 · layer 8 · jbloom/GPT2-Small-SAEs-Reformatted · FVU 0.0010 · 60.5 features/token

prompt                The Eiffel Tower is located in the city of
baseline               Paris, France.
feature #5856 @ -40    London's central London borough.
```

\#5856 is the **top-firing** feature on the final prompt token — activation
35.55, the one the panel already has selected — so this example is reachable by
following the instructions above rather than by knowing which number to type.
Same prompt, greedy decoding, no prompt tricks. We reached into layer 8 and
turned the concept down. Clearing the steer restores the baseline byte for byte.

**The SAE checks itself before it shows you anything.** An SAE fed the wrong
activation convention does not error — it returns features, in the right shape,
with plausible magnitudes, for a vector it never saw. So ModelMRI measures which
convention actually reconstructs (centered along `d_model` or not, `b_dec`
subtracted from the input or not), reports the fraction of variance unexplained,
and refuses to plot anything when no convention reconstructs. On the default SAE
that is the difference between **60.5** features firing per token and **7,491**,
and between an FVU of **0.0010** and **13,579**.

### 5. Find the step where your agent died

Two lines of `modelmri.record` around any agent run gives you a timeline: LLM calls, tool calls, subagents, each as a block. The failure glows. Click it for the exact input, output, tokens, and error.

```python
from modelmri.record import trace, step

with trace("fix-failing-tests"):
    step("llm_call", name="plan", input=prompt, output=answer, tokens_in=912)
    with step("subagent", name="auth-fixer"):
        step("tool_call", name="pytest", output="17 passed")
```

Or instrument automatically: `modelmri.record.instrument_anthropic()`.

### 6. Look inside a robot policy

This is the part nobody else ships. ModelMRI loads the **vision tower of the real SmolVLA checkpoint** and runs actual robot-camera frames through it, painting each image patch's attention back onto the frame. Scrub an episode, run the policy, drag the layer slider.

Measured on PushT frames — share of attention mass in the top 5% of patches:

| vision layer | concentration |
|---|---|
| 0 | 27% |
| 6 | 56% |
| 11 | 60% |

Early layers look everywhere; deep layers lock on. No robot hardware required — it reads public LeRobot datasets straight from disk.

### 7. Debug a model you trained yourself

Everything above is transformer-shaped. This isn't. Point ModelMRI at your own `nn.Module` — an MLP, a small CNN, whatever you're training — and get a layer-by-layer map of one real forward pass.

```python
# my_net_adapter.py — the whole contract
def load():
    model = MyNet()
    model.load_state_dict(torch.load("checkpoints/best.pt", map_location="cpu"))
    return model
```

| layer | type | output | activation | |
|---|---|---|---|---|
| `fc1` | Linear | 8×64 | −31.20 ± 24.21 | |
| `act1` | ReLU | 8×64 | 0.10 ± 0.26 | **80% dead** |
| `fc2` | Linear | 8×32 | −1.02 ± 4.74 | |
| `act2` | Tanh | 8×32 | −0.12 ± 0.90 | **55% saturated** |
| `head` | Linear | 8×3 | −0.13 ± 0.44 | |

Dead units, saturated activations, and **the first layer where a `nan` appears** — statistics exclude non-finite values on purpose, so one bad number can't turn every row below it into `nan` and hide where it started.

A `state_dict` alone is refused, with the reason: it's weights without an architecture, and guessing one would produce a map that looks authoritative and describes a network you never trained.

### 8. Send someone the finding, not the model

You found the head. Now show a colleague — who does not have your GPU, your prompt, or 8 GB of spare disk.

<p align="center">
  <img src="docs/media/share.png" alt="The attention panel's share control, with a note reading 'L8 H3 copies the subject token'" width="800">
</p>

That writes **one 54 KB file** holding the tokens, the attention, the generation and your note. No weights — it's an observation, not a checkpoint.

<p align="center">
  <img src="docs/media/viewer.png" alt="The same analysis open in the browser viewer: replay banner, attention arcs from 'Amsterdam' back through the prompt" width="800">
</p>

<p align="center"><em>The recipient opens it at <a href="https://muhammadmahadazher.github.io/ModelMRI/viewer/">the viewer</a> — nothing installed, nothing uploaded, the file is read in their browser.</em></p>

Locally it's the same page, served from the package by the standard library:

```bash
modelmri open gpt2.mri     # ~0.3s — no torch, no model, no GPU
```

Every panel reads a recording through the same calls it uses for a live model, so the arcs, the layer/head dials and the token strip all behave normally. The status pill says `replay` and the footer says *recorded, not live*, so it can never be mistaken for your own run.

The browser viewer and the Python tool are checked cell-for-cell against the same file on every change ([tests/viewer_check.py](tests/viewer_check.py)) — a viewer that renders a *slightly* different matrix would be worse than no viewer, because nothing on screen would say so.

---

## Install

```bash
pip install modelmri              # core: playground, attention, features, steering, agents
pip install "modelmri[vla-lite]"  # + robot datasets (av, pyarrow, pillow)
pip install modelmri-record       # just the agent recorder — stdlib only, an 8.9 KiB wheel
modelmri doctor                   # what this machine can and cannot run, measured
modelmri serve
```

**Will it run here?** `modelmri doctor` measures your machine and says so — OS,
cores, RAM, free disk, the torch build, the accelerator it found and its
precision, and roughly how large a model fits. It exits non-zero when something
would stop a load, so it is scriptable. `modelmri serve` prints the one-line
version at startup.

```
  accelerator NVIDIA GeForce RTX 4060 Laptop GPU (cuda)   vram 8.6 GB   bfloat16
  Models up to roughly 3.2B parameters should fit.
```

Every figure is read off the machine at the moment you ask, and a number that
cannot be determined says "could not measure" rather than being invented. Note
this is a *first-run* check rather than an install-time one, deliberately: a
wheel is an archive and pip does not execute code from it, so there is nowhere
honest to put a check during `pip install`. It is also the better place for it,
because the same machine can be perfectly able to open a shared `.mri` and
unable to load a 7B model — and those are different questions.

From source:

```bash
git clone https://github.com/muhammadmahadazher/ModelMRI && cd ModelMRI
cd frontend && npm ci && npm run build && cd ..
uv sync && uv run modelmri serve
```

**Models.** Search HuggingFace, pick from what's already cached on your machine, or switch to **Ollama** and pull any model by name. (Ollama gives you text only — internals need a HuggingFace model, and ModelMRI says so rather than pretending.)

**Nothing downloads by surprise.** Every row shows its size before you click, and a download that cannot fit your disk is refused with both numbers rather than started. One that dwarfs your GPU asks first. Whatever is running, **Stop** actually stops it — the fetch happens in a child process precisely so it can be killed, and the half-written blobs are cleaned up. This exists because a click once began fetching 1.5 TB onto an 8 GB laptop with no way out but killing the server.

**GPU when you have one.** NVIDIA, AMD, Intel and Apple silicon are detected automatically and the badge explains its choice — including the common case where torch was installed as a CPU-only build, where it prints the exact command to fix it. CPU works fine too; a 0.5B model streams in a couple of seconds.

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
| `POST /api/custom/load` · `POST /api/custom/run` | inspect a model you trained yourself |

## Status

| | |
|---|---|
| Playground · streaming · any local model · Ollama | ✅ |
| Attention inspector | ✅ |
| Head ranking by ablation | ✅ |
| Compare two runs (signed attention diff) | ✅ |
| SAE feature browser + activation steering | ✅ |
| Agent trace timeline + step inspector | ✅ |
| Robot policy (VLA) attention over real episodes | ✅ perception |
| Custom models — adapters, TorchScript, layer map | ✅ |
| Shareable `.mri` sessions + zero-install browser viewer | ✅ |
| Download size guard + a Stop button that works | ✅ |
| VLA action expert (needs `lerobot`, separate env) | 🏗️ |
| Hosted zero-install demo | ✅ |

## Honest limits

- **Attention needs eager attention.** SDPA and FlashAttention never materialize the weights, so ModelMRI loads models with `attn_implementation="eager"`. Slower, but it's the only way to see anything.
- **SAE features need an SAE that exists.** They are trained per model, and public ones exist for only a handful — this build knows of four repositories — so there is none for most of what you will load, and no amount of code makes one appear. ModelMRI offers the one that matches your model, says plainly when there is none, and falls back to a logit lens, which needs nothing but the model.
- **Custom models get a layer map, not attention.** Attention and SAE features need a transformer; for an arbitrary `nn.Module` ModelMRI shows shapes, activation statistics and pathologies. Loading an adapter runs your Python — see [SECURITY.md](SECURITY.md).
- **VLA mode is the perception half.** SmolVLA's vision tower is real and loaded from the real checkpoint; the action expert needs `lerobot`, whose torch/numpy pins conflict with the core runtime, so it lives behind an opt-in extra rather than degrading everyone's install.

## How it compares

The parts of ModelMRI are not new. Attention visualization, causal ablation,
sparse autoencoders and agent tracing all have good tools already, and most of
them do their one thing better than ModelMRI does. What is unusual here is the
combination — model internals and agent traces, in one GUI, on your hardware,
with no notebook in between.

| | what it is | where ModelMRI differs |
|---|---|---|
| [BertViz](https://github.com/jessevig/bertviz) | attention visualization in a notebook | ModelMRI is a standalone app, and ranks heads by ablating them rather than only drawing them |
| [TransformerLens](https://github.com/TransformerLensOrg/TransformerLens) | a mechanistic-interpretability library: hooks, caching, patching | TransformerLens is more capable and more precise; you write Python. ModelMRI is a UI for the common questions, with no code |
| [nnsight](https://github.com/ndif-team/nnsight) | library access to internals, including remote large models | nnsight reaches models too big for your GPU; ModelMRI only runs what fits on your machine |
| [Neuronpedia](https://www.neuronpedia.org/) | hosted browser for SAE features | Neuronpedia has far richer feature data for the models it covers; ModelMRI runs an SAE against *your* prompt, locally, and steers with it |
| [SAELens](https://github.com/jbloomAus/SAELens) | training and analysing sparse autoencoders | ModelMRI consumes SAEs, it does not train them |
| [Langfuse](https://langfuse.com/) · [Phoenix](https://github.com/Arize-ai/phoenix) · [LangSmith](https://www.langsmith.com/) | LLM application observability — traces, prompts, cost | These are production observability platforms and much stronger at it. ModelMRI records a run so you can open it next to the model's internals |

Use TransformerLens if you are doing research and want precision. Use Langfuse
or Phoenix if you are running an agent in production and need dashboards,
retention and alerting. Reach for ModelMRI when you have a model on your
machine that is doing something you do not understand, and you want to look at
it now.

## Questions people ask

### What is ModelMRI?

An open-source, local-first tool for inspecting what a model is doing
internally while it runs: attention per layer and head, which heads carry the
prediction, which interpretable features fire, what changes when you turn one
off — plus a recorder that makes an agent run inspectable step by step.
`pip install modelmri && modelmri serve`, then open `http://localhost:5900`.

### Does it work with GPT-4, Claude, or Gemini?

Not for internals, and no tool can — closed API models do not expose attention
weights or activations to anyone outside the provider. ModelMRI needs weights
it can run, so internals mean a local HuggingFace model.

The **agent recorder is a different matter**: it wraps whatever your agent
calls, including hosted APIs, so you can record and inspect a run driven
entirely by Claude or GPT-4. `modelmri.record.instrument_anthropic()` does it
in one line.

### Do I need a GPU?

No. CPU works — a 0.5B model streams in a couple of seconds. NVIDIA, AMD,
Intel and Apple silicon are detected automatically when present, and if torch
was installed as a CPU-only build the badge says so and prints the command to
fix it.

### Which models work?

Any causal LM transformers can load with eager attention, from the HuggingFace
cache you already have, a plain folder on disk, or a search-and-download in
the app. The ones with a recorded end-to-end result are **GPT-2, Qwen3-0.6B,
Qwen2.5-0.5B-Instruct, SmolLM2-360M-Instruct and Gemma-3-270m-it** — the
[verified table](https://muhammadmahadazher.github.io/ModelMRI/docs/#verified-not-asserted)
lists what each one actually measured. Others should work and are not claimed
to have been checked: Llama-3.2-1B is deliberately absent because the
`meta-llama` repos are gated and returned 403 for the account used, and an
untested model in a list headed "supported" is just a guess in bold. Ollama
models work for text generation; internals need a HuggingFace model, and
ModelMRI tells you that instead of quietly showing you nothing. Non-transformer
models you trained yourself get a layer map with activation statistics.

### Is any of my data sent anywhere?

No. There is no cloud, no account and no telemetry. Models are downloaded from
HuggingFace if you ask for one you don't have; beyond that, nothing leaves the
machine. A `.mri` file you choose to share contains tokens, attention and your
note — never weights — and the browser viewer reads it client-side without
uploading it.

### What is a `.mri` file?

One ~54 KB file holding the tokens, the attention matrix, the generation and a
note you wrote. It exists so you can send a colleague the *finding* without
sending them 8 GB of weights or asking them to install anything — they open it
at [the viewer](https://muhammadmahadazher.github.io/ModelMRI/viewer/) in a
browser. `modelmri open file.mri` does the same locally in about 0.3s, with no
torch and no GPU.

### How is this different from TransformerLens or BertViz?

BertViz draws attention; ModelMRI also measures which heads matter by removing
them. TransformerLens is a library you write code against and is more precise
and more flexible than this; ModelMRI is a UI for the questions people ask
most, and adds SAE steering, agent traces and robot policies in the same
window. See [How it compares](#how-it-compares).

### Is it production-ready?

No, and the package says so — it is classified alpha. It is a debugging and
research tool, not infrastructure. The measurements it reports are tested, but
the API surface still moves between minor versions; see the
[changelog](CHANGELOG.md).

### How do I cite it?

[CITATION.cff](CITATION.cff) is in the repository root — GitHub's "Cite this
repository" button reads it. Please include the version, because the measured
figures in this README belong to the release that produced them.

## Contributing

Issues and pull requests are welcome. One rule runs the whole repository:
**don't ship a measurement you haven't verified.** A visualization that looks
plausible and is wrong is worse than none, because interpretability is exactly
the domain where nobody has an independent way to notice.

- [Contributing guide](CONTRIBUTING.md) — setup, quality gates, and the three
  bugs that made that rule
- [Code of conduct](CODE_OF_CONDUCT.md)
- [Security policy](SECURITY.md) — trust model, credential handling, and what
  loading a model actually executes
- [Support](SUPPORT.md) · [Changelog](CHANGELOG.md)

## Built in public

Notes, mistakes, and what broke: [modelmri.substack.com](https://modelmri.substack.com)

MIT © Muhammad Mahad Azher
