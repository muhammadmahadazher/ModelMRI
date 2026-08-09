# Attention

After any generation, ModelMRI can show you which tokens each token looked at.

![Attention arcs following the cursor across the token strip](../media/attention.gif)

## How to read it

Every token in the sequence becomes a chip. Hover one — or Tab to it and the
same thing happens — and arcs sweep back to the tokens it attended to. Arc
thickness is the attention weight. Click, or press Enter, to pin a token so you
can change layer and head while the arcs stay put.

The `layer` and `head` selectors move you through the model. This is where the
interesting behaviour lives:

- **Early layers** attend broadly and locally. Top-5% attention mass around
  27% is typical — the model is still smearing information around.
- **Late layers** are sharp. The same measure runs past 59%: specific heads
  have found specific things to look at.

If you generate `The Eiffel Tower is located in the city of` with GPT-2 and
walk to the later layers, you can watch the token that produces " Paris" attend
back to " capital" and " France".

## What is actually computed

Attention weights are not a by-product of generation — they're thrown away
unless you ask for them. So:

1. Generation finishes and the full token sequence is retained.
2. On the first attention request, ModelMRI runs **one more forward pass** over
   that sequence with `output_attentions=True`.
3. Every layer is cached as fp16 on CPU, so changing head or layer afterwards
   is instant.

That first request costs a real forward pass — seconds on CPU for a 0.5B model.
The panel says `computing…` while it happens rather than showing you an empty
box.

!!! note "Why models load with eager attention"
    SDPA and FlashAttention never materialise the attention matrix — that's
    most of why they're fast. ModelMRI therefore loads models with
    `attn_implementation="eager"`. It's slower, and it's the only way to see
    the numbers at all.

## Reading the numbers honestly

Every row of the matrix is a softmax, so it sums to 1.0. ModelMRI's own tests
assert that, along with the causal mask holding — no token attends to a token
after it. If those ever stopped being true, the picture would be decorative
rather than informative.

You will also see a large **attention sink** on the first token or a chat
template's opening tag. That's real and well documented, not a bug: heads that
have nothing useful to do park their mass somewhere harmless.

## Sharing what you found

A screenshot of a heat map cannot be explored, and "download this 8 GB model,
use my exact prompt, and look at layer 14 head 3" is not a thing you can ask
of a reader. **Share this view** writes a `.mri` instead: the tokens, the
attention, the generation, the decode settings, and a one-line note saying
what you think you found.

```
Share this view → "L8 H3 copies the subject token" → gpt2.mri (54 KB)
```

The person you send it to needs one command, and no model:

```bash
pip install modelmri && modelmri open gpt2.mri
```

Or, if they already have ModelMRI running: click **Open a shared analysis**,
or drop the file anywhere on the page. It works with **no model loaded** — the panels read the recording
through exactly the same calls they use for a live model, so the arcs, the
token strip and the layer/head dials all behave normally. The status pill says
`replay` and the panel footer says *recorded, not live*, so there is never a
moment where you could mistake one for the other.

What a `.mri` deliberately does **not** contain: weights. It is an
observation, not a checkpoint. That is also why the features panel disappears
in replay — SAE features need activations, which means it needs the model.

!!! note "Attention is stored lossily, and the file says so"
    Attention values are quantised to one byte against each matrix's own
    maximum, then gzipped — that is what turns tens of megabytes into tens of
    kilobytes. Worst measured error on a real gpt2 run is 0.002, and the
    strongest attention in every row survives, so the picture you send is the
    picture they see. But if you plan to do arithmetic on the numbers rather
    than look at them, read `meta.precision` in the file first.

Loading a model, or generating your own run, closes an open session. Your
output above someone else's heat map would be a discrepancy nothing on screen
could explain.

## Limits worth knowing

- **Ollama models have no attention view.** Ollama serves text over HTTP; the
  internals never leave its process. The panel says so instead of pretending.
- **Attention is not explanation.** A head attending to a token tells you where
  information moved, not why the model produced what it did. Treat it as one
  instrument among several — which is why features and steering exist.
- Changing the model invalidates the view. A generation belongs to the weights
  that produced it, and ModelMRI refuses to show one model's tokens through
  another model's forward pass.
