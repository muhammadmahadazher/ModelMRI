---
description: "Answers to common questions about ModelMRI — which models work, whether it runs without a GPU, what head ranking actually measures, how it compares to TransformerLens and BertViz, and where it stores files."
---

# FAQ

## What is ModelMRI?

ModelMRI is an open-source, local-first interpretability and debugging tool for
transformer language models, vision-language models, robot policies and LLM
agents. It shows per-layer, per-head attention from a live forward pass, ranks
attention heads by causal ablation, decomposes the residual stream with sparse
autoencoders, steers generation along a feature direction, maps activations
inside a custom `nn.Module`, and records agent runs as an inspectable timeline.

```bash
pip install modelmri
modelmri serve
```

Then open <http://127.0.0.1:5900>. Licence: MIT.

## Who is it for?

Someone who has a model or an agent on their own machine that is behaving
oddly, and wants to look at it now rather than write a notebook first. It is a
debugging tool that happens to use interpretability methods, not a research
framework — if you are doing interpretability research and need precision and
control, [TransformerLens](https://github.com/TransformerLensOrg/TransformerLens)
is the better instrument and this page says so below.

## Can I use it with GPT-4, Claude or Gemini?

Not for internals. Closed API models do not expose attention weights or
activations to anyone outside the provider, so no tool — this one included —
can show you inside them. Internals require weights ModelMRI can run locally.

The agent recorder is different. It wraps whatever your agent calls, hosted
APIs included, so you can record and inspect a run driven entirely by Claude or
GPT-4:

```python
import modelmri.record
modelmri.record.instrument_anthropic()
```

See [Recording agents](guides/agents.md).

## Do I need a GPU?

No. On CPU a 0.5B model streams in a couple of seconds. NVIDIA, AMD, Intel and
Apple silicon are detected automatically when present. If PyTorch was installed
as a CPU-only build — which happens quietly and is the most common reason for
an unexpectedly slow session — the device badge says so and prints the exact
command that fixes it.

## Which models are supported?

Any causal language model `transformers` can load with eager attention, found
in the HuggingFace cache you already have, a folder on disk, or searched for
and downloaded in the app.

The models with a **recorded end-to-end result** are GPT-2, Qwen3-0.6B,
Qwen2.5-0.5B-Instruct, SmolLM2-360M-Instruct and Gemma-3-270m-it; the
[verified table](index.md#verified-not-asserted) lists what each measured.
Others should work. They are not listed as supported, because a model nobody
ran is a guess, and Llama-3.2-1B is a live example — it runs in principle, but
the `meta-llama` repositories are gated and returned 403 for the account used
here, so there is no result to report.

Models served through **Ollama** generate text but expose no internals; that is
a property of Ollama, and ModelMRI says so in the UI rather than showing an
empty panel. Non-transformer networks you trained yourself get a
[layer map](guides/custom-models.md) instead of attention.

## Why does attention need `attn_implementation="eager"`?

SDPA and FlashAttention are faster precisely because they never materialise the
full attention matrix — it is fused away inside the kernel. There is nothing to
read afterwards. ModelMRI therefore loads models with eager attention, which is
slower and is the only implementation from which attention weights can be
recovered at all.

## What does "rank heads" actually measure?

It zeroes one attention head, runs the model again on the same tokens, and
measures the KL divergence between the new output distribution and the
original. A larger number means removing that head moved the answer further.

A ranking costs `n_heads + 2` forward passes for one layer, or
`n_layers × n_heads + 2` for a whole model — 146 passes for GPT-2, 450 for
Qwen3-0.6B. That cost is a property of the algorithm and is portable. The
wall-clock time is not, which is why ModelMRI measures one layer on your
machine and extrapolates instead of printing a figure from someone else's.

## Are per-head scores each head's contribution to the answer?

No, and treating them that way is the most likely way to misread this panel.
They are not additive. On GPT-2 layer 0 the twelve per-head scores sum to
**1.995**, while ablating the entire layer at once gives **0.208** — the parts
"sum" to roughly ten times the whole. Heads are redundant: remove one and
others compensate, remove all twelve and nothing does.

The score also depends on what a removed head is replaced with, so ModelMRI
names the baseline on screen and offers more than one. And a KL depends on the
prompt, the dtype and the sequence length: the same head scoring 0.898 in bf16
scores 0.784 in fp32 and 0.825 over a 261-token generation. A figure quoted
without those conditions cannot be checked by anyone.

## Why is there no sparse autoencoder for my model?

Because SAEs are trained per model, and public ones exist for only a handful.
This build knows of four repositories. For most models you load there is no SAE
in existence, and nothing in the software can conjure one.

When that happens ModelMRI says so plainly and offers a **logit lens** instead,
which needs nothing but the model: it projects intermediate layers through the
output head to show what the model would predict if it stopped there. Less
interpretable than features, but real. See
[Features and steering](guides/features.md).

## What is a `.mri` file?

A single file of roughly 54 KB holding the tokens, the attention matrix, the
generation and a note you wrote — and no weights, because it records an
observation rather than a checkpoint. It exists so you can send a colleague the
finding without sending 8 GB of model or asking them to install anything.

They open it at [the viewer](https://muhammadmahadazher.github.io/ModelMRI/viewer/)
in a browser, where it is read client-side and never uploaded. Locally,
`modelmri open analysis.mri` serves the same page from the package in about
0.3 s with no torch, no model and no GPU. A replayed file always says `replay`
on the status pill, so it cannot be mistaken for a live run.

## Does ModelMRI send my data anywhere?

No. There is no cloud, no account and no telemetry. Prompts, weights, traces
and credentials stay on the machine. The only outbound traffic is to
huggingface.co, and only when you ask for a model or an SAE you do not already
have.

## Where does it store files, and how do I remove it?

```bash
modelmri where       # print every directory it reads or writes
modelmri uninstall   # remove everything it has written to this machine
```

Locations follow each platform's own convention rather than scattering
dotfiles: `%LOCALAPPDATA%\ModelMRI` and `%APPDATA%\ModelMRI` on Windows,
`~/Library/Application Support/ModelMRI` on macOS, and `$XDG_DATA_HOME/modelmri`
with `$XDG_CONFIG_HOME/modelmri` on Linux.

`uninstall` leaves your HuggingFace cache alone by default — those models were
probably not downloaded by ModelMRI and may be shared with other tools. Pass
`--models` to include them, and it reports the space actually reclaimed rather
than the space it intended to reclaim.

## How does it compare to TransformerLens, BertViz or Langfuse?

Honestly: each of those does its own job better.

| | what it is | where ModelMRI differs |
|---|---|---|
| [BertViz](https://github.com/jessevig/bertviz) | attention visualization in a notebook | a standalone app, and it ranks heads by ablating them rather than only drawing them |
| [TransformerLens](https://github.com/TransformerLensOrg/TransformerLens) | mechanistic-interpretability library — hooks, caching, patching | more capable and more precise, and you write Python for it; ModelMRI is a UI for the common questions |
| [nnsight](https://github.com/ndif-team/nnsight) | library access to internals, including remote models | reaches models too large for your GPU; ModelMRI runs only what fits locally |
| [Neuronpedia](https://www.neuronpedia.org/) | hosted SAE feature browser | far richer feature data for the models it covers; ModelMRI runs an SAE against *your* prompt and steers with it |
| [SAELens](https://github.com/jbloomAus/SAELens) | training and analysing SAEs | ModelMRI consumes SAEs, it does not train them |
| [Langfuse](https://langfuse.com/) · [Phoenix](https://github.com/Arize-ai/phoenix) · [LangSmith](https://www.langsmith.com/) | LLM application observability | production platforms with retention, dashboards and alerting; ModelMRI records a run so you can open it beside the model's internals |

The combination is the unusual part — model internals and agent traces in one
local GUI — not any individual capability.

## Is it production-ready?

No, and the package is classified alpha to say so. It is a debugging and
research tool rather than infrastructure: measurements are tested, but the API
surface still moves between minor releases. See the
[changelog](https://github.com/muhammadmahadazher/ModelMRI/blob/main/CHANGELOG.md).

## How do I cite it?

[`CITATION.cff`](https://github.com/muhammadmahadazher/ModelMRI/blob/main/CITATION.cff)
is in the repository root and GitHub's "Cite this repository" button reads it.
Please include the version — the measured figures throughout these docs belong
to the release that produced them.
