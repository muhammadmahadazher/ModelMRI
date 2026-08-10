---
description: "Fixes for real ModelMRI problems: stalled HuggingFace downloads, CPU-only PyTorch builds, gated repositories, missing sparse autoencoders, and models that will not load."
---

# Troubleshooting

Most of these are things that actually happened during development, with what
the tool now does about them.

## A load sits there for minutes

Normal for a cold model — the weights are downloading. The progress bar shows
the stage and real bytes, so you can tell *downloading* from *stuck*.

**If it says "no new data for Ns — the download may have stalled":** believe
it. A dead HuggingFace transfer doesn't raise; it just stops moving, and the
load would otherwise wait forever. Cancel, and either retry or pick a smaller
model. This is a network condition, not something ModelMRI can retry its way
out of.

## The badge says CPU and I have a GPU

Hover it — the accelerator always explains its own choice. Common causes:

- **PyTorch installed without CUDA.** `pip install torch` gives you a CPU build
  on some platforms. Install from the CUDA index for your driver.
- **The model didn't fit.** ModelMRI falls back to CPU rather than dying, and
  says `fell back to CPU: OutOfMemoryError`. A 7B model in bf16 needs ~14 GB.
- **Both failed.** You get an explicit "does not fit" error rather than a
  half-loaded model.

## "Gated model" but I'm signed in

Being signed in is not the same as having accepted that model's licence — it's
per repository. Rows you can't use are marked, and clicking one opens the page
where you accept.

After accepting, search again. The access check is live, not cached from
sign-in.

## The attention panel says the model changed

You loaded a different model after that generation. A generation belongs to the
weights that produced it, and running new weights over old token ids yields
numbers that look fine and mean nothing. Generate again.

## Ollama models have no attention or features

Ollama serves text over HTTP; the internals never leave its process. For
attention, features and steering, load the model through HuggingFace instead.

## The SAE won't load

- **Wrong model.** An SAE only fits the model it was trained on. `d_in` must
  equal the model's `hidden_size`.
- **Unsupported hook point.** ModelMRI reads the residual stream, so
  `hook_resid_pre` and `hook_resid_post` work and others are refused — better
  than feeding the SAE a tensor it never saw in training.

## No models in "On this machine"

The scan starts from the directory you launched from. Either start the server
where your models live, or point it explicitly:

```bash
export MODELMRI_MODELS_DIR=/mnt/big-disk/models
```

If it says the scan stopped early, the tree was too large to walk in its time
budget — set the variable rather than letting it guess.

## Traces aren't showing up

- The recorder POSTs to `127.0.0.1:5900`. If nothing is listening it writes
  the trace to ModelMRI's data directory instead — run `modelmri where` and
  look at `undelivered_traces`. Set `MODELMRI_TRACE_DIR` to put them elsewhere.
- Import a file by POSTing it to `/api/traces/import`.
- Recording is best-effort by design: it will never raise to tell you it
  failed, because a tracing library that can take down your app is one nobody
  leaves switched on.

## A secret appeared in a trace

It shouldn't — redaction is on by default and covers the common credential
shapes, including in payloads the recorder truncated. If you find a shape that
slips through, that's a bug worth reporting, and you can cover it immediately:

```python
from modelmri_record.redact import make_redactor
red = make_redactor([r"your-token-shape-[0-9a-f]{32}"])
```

## Something else

Open an issue with what you ran and what you saw:
<https://github.com/muhammadmahadazher/ModelMRI/issues>
