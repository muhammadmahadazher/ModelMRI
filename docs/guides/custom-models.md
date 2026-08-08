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
