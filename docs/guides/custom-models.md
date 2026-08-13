---
description: "Inspect a PyTorch nn.Module you trained yourself. ModelMRI maps one real forward pass layer by layer: output shapes, activation statistics, dead units, saturation, and the first layer where a nan appears."
---

# Your own models

Everything else in ModelMRI is transformer-shaped — attention heads, residual
streams, sparse autoencoders. None of it applies to the networks most people
actually train: an MLP on tabular data, a small CNN, a two-layer regressor.

This panel is for those. It gives you a layer-by-layer map of **one real
forward pass**: what shape comes out of every module, what the activations look
like, how many units are dead, whether anything has gone non-finite, and where
the time goes.

## The short version

```bash
cp examples/adapter_template.py my_net_adapter.py
# edit load() to build your model and load your weights
modelmri serve
```

Open **CUSTOM MODEL**, click **Find models here**, pick your adapter, click
**Run forward pass**.

## What ModelMRI accepts

| you have | works? | what to do |
|---|---|---|
| a HuggingFace-format folder (`config.json` + weights) | ✅ | use the normal model picker — it's already found |
| a Python file that builds your `nn.Module` | ✅ | that's an adapter; see below |
| TorchScript (`torch.jit.save`) | ✅ | pick the `.pt` directly |
| a GGUF (`.gguf`) | ✅ | pick it — the header reads instantly, and the **load for introspection** button turns it into a full model; see [below](#gguf-and-what-loading-one-costs) |
| a `state_dict` (`torch.save(model.state_dict(), ...)`) | ❌ | write an adapter — see [why](#why-a-state_dict-alone-is-refused) |
| an ONNX file, a scikit-learn pickle, a Keras model | ❌ | not yet |

## Adapters

An adapter is a Python file with one required function:

```python
def load():
    model = MyNet()
    model.load_state_dict(torch.load("checkpoints/best.pt", map_location="cpu"))
    return model
```

Two optional extras, both worth adding:

```python
def example_input():
    return torch.randn(8, 20)      # one realistic batch

LABELS = ["negative", "neutral", "positive"]   # names your output classes
```

Return the module itself — not a `state_dict`, not a path, not a
`(model, optimizer)` tuple. Each of those is refused by name rather than
producing a confusing failure later.

!!! tip "Run it yourself first"
    If `python my_net_adapter.py` works, ModelMRI will. The template ends with
    a `__main__` block that does exactly that.

## Reading the layer map

| column | what it tells you |
|---|---|
| **layer** | the attribute name you gave it, so it matches your code |
| **type** | the `nn.Module` class |
| **output** | the shape that actually came out — not the shape you intended |
| **params** | parameters owned by that module alone, not its children |
| **activation** | the range the values occupied, with the mean marked, and `mean ± std` |
| **ms** | wall time in that module, with a bar relative to the slowest |

Rows are highlighted when something is worth looking at:

- **`n% dead`** — that fraction of the activation's outputs were *exactly*
  zero. Normal for a ReLU at around half; 90% means most of your layer is
  doing nothing, and no gradient flows back through it.
- **`n% saturated`** — that fraction sat within 1% of the activation's own
  bound. Only reported for bounded activations (`Tanh`, `Sigmoid`, `Softmax`
  and friends), because a large `ReLU` output is not saturation, it's just a
  large output.
- **`n nan/inf`** — non-finite values, and *which layer they first appear in*.
  This is the one that saves an evening.

!!! note "Statistics exclude nan and inf, deliberately"
    A single `nan` propagates through `mean`, `std`, `min` and `max`, so the
    naive version reports `nan` for every column of every layer downstream —
    which hides exactly the thing you're looking for. ModelMRI counts the
    non-finite values, reports the count, then computes the statistics from
    what's left. The first row with a non-zero count is where it started.

## The input shape

ModelMRI never runs a forward pass on a shape you haven't seen.

- If your adapter has `example_input()`, that's what runs, and the panel says
  so.
- Otherwise ModelMRI infers a shape from your first `Linear`, `Conv` or
  `Embedding` and puts it in the field **marked as inferred**, with the part
  it guessed named explicitly — a `Conv2d` fixes the channel count but not the
  height and width, and the panel says that rather than pretending.
- If there's nothing to infer from, it refuses and asks you to type one.

You can always overwrite the field. A wrong shape gives you the real
exception, prefixed with the observation that the shape is the usual cause.

## Why a `state_dict` alone is refused

`torch.save(model.state_dict(), "weights.pth")` saves numbers. It does not save
the class that produced them, the order the layers run in, or the forward
method. Nothing can reconstruct your architecture from it — PyTorch itself
can't, which is why `load_state_dict` requires you to build the model first.

ModelMRI says that, tells you how many tensors it found and names a few, and
points you at the template. The alternative — guessing an architecture that
fits the tensor shapes — would produce a layer map that looks authoritative
and describes a network you never trained.

## GGUF, and what loading one costs

Every other local runner shows a GGUF as a quantisation label and a file size.
Open one here and you get where the bits actually went, computed per tensor
from the file's own table — measured on `Qwen3-0.6B-Q4_K_M.gguf`, a file
labelled Q4_K reads **5.245 bits per weight effective**. Almost all of that
0.745-bit lift above Q4_K's 4.5 comes from the 29 Q6_K tensors; the 113 F32
tensors are real but tiny, totalling 65,536 elements and 0.003 bits of it.

Then there is a button that loads it. Pressing it gives you the lens, the head
sweep, the patching grid and attention on a file that used to be readable and
not runnable. What it does **not** give you is a 4-bit model in memory:

| | Qwen3-0.6B-Q4_K_M | SmolLM2-135M-Q4_K_M | Gemma 4 E2B Q4_0 † |
|---|---|---|---|
| file on disk | 0.397 GB | 0.105 GB | 2.83 GB |
| parameters | 596,049,920 | 134,515,008 | 4,628,569,635 |
| resident at bfloat16 | **1.192 GB** (3.00×) | **0.269 GB** (2.55×) | **9.26 GB** (3.27×) |
| …predicted vs weighed | error 0.000000 | error 0.000000 | — |
| peak host RAM, predicted | 2.384 GB | 0.538 GB | 18.51 GB |
| …sampled RSS delta | 2.30 GB (−3.5%) | 0.585 GB (+8.6%) | — |

† The Gemma column is a **projection, not a measurement**: that load was
refused (see below), so nothing was ever weighed. The other two columns come
from `python scripts/measure_docs.py --gguf FILE` on an RTX 4060 with
transformers 5.13, and running it prints exactly these numbers back.

The resident figure is exact — `parameters × dtype bytes`, error 0.000000 on
both files that loaded. The peak is a projection accurate to about ten
percent, and note the errors have **opposite signs**: process RSS also carries
the tokeniser and the allocator's own release timing, which land differently
at 135M than at 596M. There is no correction factor to apply, so the tool
reports both figures and their disagreement rather than picking one.

Transformers has no kernels for these quantised types, so it dequantises every
tensor on the way in — and it materialises the whole checkpoint as float32
before casting, which is why asking for bfloat16 still transits through
`parameters × 4`. The resident figure is `parameters × dtype bytes` and it is
exact: 1,192,099,840 bytes predicted from the header, 1,192,099,840 weighed
from the built module.

Both numbers come from the header, which is a few hundred kilobytes of a
multi-gigabyte file, so the panel shows them **before** you press anything.
That is the point of the feature. Gemma 4 E2B is 4.63 *billion* raw parameters
behind an "E2B" name; on a 16.94 GB machine the answer is "will not fit", and
it says *total* RAM rather than free, because closing other programs cannot
change it.

If what you want is to *run* a large GGUF rather than look inside it, use the
Ollama backend — same file, real bit width, much faster, and no introspection.
That trade is the whole reason both exist.

Two things the panel will refuse rather than guess:

- **A directory holding several quantisations.** Repos ship Q4_K_M beside Q8_0
  beside BF16. Which one loads is which one your measurements describe, so you
  name it.
- **The companions.** `mmproj-*` is a vision projector, `mtp-*` is a
  speculative-decoding head, and `*-00001-of-*` is one shard of a split file.
  None is a language model.

And a standing caveat on everything measured afterwards: a loaded GGUF is the
*quantised* weights, dequantised. It is not the original model. To see how far
apart they are, point `quantdiff` at both — it is a library module
(`modelmri/quantdiff.py`), not a route, so it is used from Python rather than
from the UI.

## What this is not

- **It is not a training monitor.** One forward pass, on demand. For loss
  curves over time, use TensorBoard or Weights & Biases; they answer a
  different question and answer it well.
- **It is not attention or feature analysis.** Those panels need a
  transformer. If your custom model *is* one, load it through the normal
  picker in HuggingFace format and you get all of it.
- **It runs on CPU.** These are small models and one pass; the device
  plumbing isn't wired through yet.
- **Gradients are not shown.** The pass runs under `torch.no_grad()`. Dead
  units and non-finite activations are visible; vanishing gradients are not.

## Security

Loading an adapter **imports and runs your Python file**. That is the point —
only your code knows how to build your model — but it means an adapter is
exactly as trustworthy as its author.

ModelMRI limits the blast radius:

- It only imports a path you explicitly chose.
- That path must be under the directory you launched in, or one you named in
  `MODELMRI_MODELS_DIR`. Anything else is refused, by name.
- It never fetches an adapter from the network.
- **Discovery never imports anything.** Finding candidates reads the first 4 KB
  of each file as text and looks for a module-level `def load(`. A file that
  would crash the process on import is listed safely, not executed.

Treat an adapter you didn't write the way you'd treat any other script someone
sent you. See [SECURITY.md](https://github.com/muhammadmahadazher/ModelMRI/blob/main/SECURITY.md).

## API

Everything the panel does is available over HTTP — see the
[API reference](../reference/api.md).

```bash
curl -s localhost:5900/api/custom/candidates | jq '.adapters[].path'
curl -s -XPOST localhost:5900/api/custom/load \
     -H 'content-type: application/json' \
     -d '{"path": "my_net_adapter.py"}'
curl -s -XPOST localhost:5900/api/custom/run \
     -H 'content-type: application/json' -d '{"shape": [8, 20]}' | jq '.layers[]'
```
