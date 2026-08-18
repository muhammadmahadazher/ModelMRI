---
description: "Browse sparse-autoencoder features over a language model's residual stream, then steer generation by clamping a feature direction and run a deterministic A/B against the baseline."
---

# Features and steering

Attention shows you where information moved. Features show you *what* the model
is representing — and steering lets you change it and watch the output move.

## Sparse autoencoders, briefly

A model's residual stream is dense: hundreds to thousands of numbers per token,
each of which participates in many unrelated things at once. That superposition is why
you can't read a neuron and learn anything.

A sparse autoencoder is trained to re-express that stream in a much wider basis
— 24,576 features here — under a sparsity penalty, so only a handful fire for
any given token. Those features are far more likely to correspond to something
you can name.

ModelMRI reads two SAE layouts: SAELens (`cfg.json` beside
`sae_weights.safetensors`) and Gemma Scope (one `params.npz` per layer,
dictionary width and average L0).

There is no default repo. Loading with none named asks
`modelmri/sae_registry.py` which release belongs to the model you have
resident — `google/gemma-2-2b` resolves to `google/gemma-scope-2b-pt-res`,
verified end to end on this project's own hardware. Gemma Scope publishes many
releases per layer, so a width and sparsity you did not name are CHOSEN by
rule and the answer says which rule, in `release.chosen_by`.

!!! warning "Most models have no SAE at all"
    They are trained per model, and public ones exist for a handful. A model
    with no registered release is refused BY NAME rather than being handed
    somebody else's SAE — the `d_in` has to equal the model's `hidden_size`
    and the layer has to exist, so a mismatched one produces numbers that look
    fine and mean nothing. The logit lens works on every model and needs
    nothing extra.

## Reading features

1. **Load the SAE** in the features panel.
2. **Click a token.** You get its top firing features, ranked by activation and
   rendered strongest-first — the ordering is the information, since every row
   is the same colour.
3. **Click a feature.** The whole token strip shades by how strongly that one
   feature fires across the sequence, and the peak token is ringed and scrolled
   into view. In a 256-token generation that one chip is the thing you were
   looking for.

## Steering

Once you know which feature you care about, you can add its decoder direction
straight into the residual stream and regenerate:

```
activations += scale * W_dec[feature_id]
```

The **Run steering A/B** button does exactly that, twice: once clean, once
steered, same prompt and greedy decoding both times. What differs is the
intervention.

Negative scales suppress; positive amplify. The default is −40, which is large
enough to see. Small values often do nothing visible, which is itself worth
knowing.

!!! note "The A/B does not disturb your analysis"
    Those two completions run with `commit=false`, so they don't become the
    sequence the panels are describing. An earlier version let them, and the
    token strip quietly ended up describing a different generation than the
    heat map above it.

    Generate is also locked out while steering is installed — the hook lives on
    the runtime, not the panel, so a generation started mid-A/B would come back
    steered with nothing on screen saying so.

## Hook points

SAEs are trained at a specific point in the block, and ModelMRI honours it:

| hook | what the SAE sees |
|---|---|
| `blocks.N.hook_resid_pre` | the stream **entering** block N |
| `blocks.N.hook_resid_post` | the stream **leaving** block N |

Anything else is rejected. Feeding a `resid_post` SAE the pre-block stream
doesn't error — it produces confident features describing activations the SAE
was never trained on, which is worse.

## What this is and isn't

A feature that fires on Paris-related tokens is evidence, not proof, that the
model represents "Paris". Feature interpretation is an open research problem,
and an SAE is a lossy re-description of the stream, not a decoding of it.

What steering gives you that inspection alone doesn't is a **causal** test: if
pushing that direction changes the output in the way the label predicts, the
label is doing some real work.
