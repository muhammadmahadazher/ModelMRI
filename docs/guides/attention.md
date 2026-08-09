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

## Which heads actually mattered

144 heat maps and no reason to open any particular one is a browsing tool.
**Rank heads** turns it into a question: zero one head, run the model again,
and measure how far the next-token distribution moves.

```
Rank heads → L0 H7  KL 0.825   p(" the") 0.082 → 0.065
             L0 H10 KL 0.559
             L0 H9  KL 0.469
```

The head dropdown comes back ordered, with each head's score on it, and the
top head is selected for you. **all N layers** ranks the whole model.

That button only appears after you have ranked one layer, and that is
deliberate — it quotes what the sweep will cost, and it cannot quote a number
it has not measured. The cost is dominated by the forward-pass count
(`n_layers × n_heads + 2`), and the per-pass time is stable enough to
extrapolate from: on an RTX 4060, gpt2 runs 71 ms/pass and one layer predicts
the full sweep at 10.4 s against 10.28 s actual; Qwen3-0.6B's 307 ms/pass
predicts 138 s against 137.2 s. Both within 1%.

!!! warning "Three ways this number can be a confident lie"
    - **`head_dim` is not `hidden_size // n_heads`.** That quotient is right
      for gpt2 (768/12) and Qwen2.5-0.5B (896/14), and wrong on Qwen3-0.6B
      (64 against a real 128) and gemma-3-270m-it (160 against 256). It is
      read off the output projection's own width instead, and a mismatch is
      refused rather than guessed — the quotient would ablate half of one head
      plus half of the next and rank the result confidently.
    - **KL, not a logit difference.** Softmax is shift-invariant, and ablation
      moves whole logit vectors. Zeroing gpt2's L0H0 on "The capital of France
      is" moves the top token's logit by −0.258, but the whole vocabulary
      moves by −0.145 — so the honest residual is −0.113, and a raw logit
      difference would call that head 2.3× more important than it is.
    - **The baseline is part of the answer.** Zeroing a head and replacing it
      with its own mean over positions are different questions. On gpt2 layer
      0 they give different answers: zero ranks heads 7, 10, 9; mean ranks 3,
      1, 10, dropping head 7 to sixth. Both are offered and the one you used
      is named on screen.

And the claim it refuses to make: these are **not** each head's share of the
prediction. On gpt2 layer 0 the twelve per-head scores sum to 1.995 while
zeroing the whole layer gives 0.208 — eight times too much. On
gemma-3-270m-it layer 0 it inverts: four per-head scores sum to 0.0007
against 6.57 for the layer, so every head looks irrelevant alone while the
layer is load-bearing. Each score says one thing only: removing *this* head,
on its own, moves the answer this much.

Scores at or below the measured noise floor are greyed rather than ranked.
The floor is one extra forward pass with nothing ablated, and it measures
exactly 0.0 on CPU and CUDA in fp32, bf16 and fp16 — the pass is what
establishes that rather than assuming it.

## What changes when a head is gone

Next to any ranked head, **what changes?** subtracts two runs of the same
sequence: the model as it is, and the model with that head removed. Arcs
appear in one colour where attention *increased* without the head and another
where it decreased, scaled against the difference's own peak.

Both sides are forward passes over one token sequence, never two generations.
Sampling diverges, and chat templates insert anywhere from 0 to 29 leading
tokens, so subtracting two generations would line up token 5 of one sentence
against token 5 of a different one and draw a smooth picture of nothing.

!!! note "Why it opens at the next layer"
    Ablation removes a head's *output*. The layer that head lives in is
    computed from an unchanged input, so its attention is bit-identical every
    time — comparing there is guaranteed to show zero. The first layer that
    can differ is the next one, so that is where it opens. If a difference is
    genuinely zero, the panel says so instead of showing you an empty canvas.

## Sharing what you found

A screenshot of a heat map cannot be explored, and "download this 8 GB model,
use my exact prompt, and look at layer 14 head 3" is not a thing you can ask
of a reader. **Share this view** writes a `.mri` instead: the tokens, the
attention, the generation, the decode settings, and a one-line note saying
what you think you found.

```
Share this view → "L8 H3 copies the subject token" → gpt2.mri (54 KB)
```

The person you send it to needs **nothing at all**: the
[hosted viewer](https://muhammadmahadazher.github.io/ModelMRI/viewer/) reads
the file in their browser. Nothing is uploaded — there is no server behind
that page to upload to.

If they do have ModelMRI installed, the same page is bundled with it:

```bash
modelmri open gpt2.mri     # ~0.3s, no model, no torch
```

That serves the viewer from the standard library on the loopback interface
and hands it the one file. It is deliberately *not* `modelmri serve`: reading
a 54 KB recording should not import torch, which used to cost 26 seconds.

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
