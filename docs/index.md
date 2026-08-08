# ModelMRI

**See inside the model.** Attention, concepts and steering for any local model — plus a
flight recorder for your agents. One `pip install`, everything on your machine.

```bash
pip install modelmri
modelmri serve
```

Then open <http://127.0.0.1:5900>.

!!! tip "Just want to trace an agent?"
    You don't need the viewer's dependencies. `pip install modelmri-record` is
    stdlib only — 7 KiB, no torch. See [Recording agents](guides/agents.md).

---

## What it does

<div class="grid cards" markdown>

-   **Attention — where each token looked**

    ---

    Generate, then hover any token to see the arcs it attended to. Every layer,
    every head, computed on demand from a real forward pass with
    `output_attentions=True`.

    [Read more →](guides/attention.md)

-   **Features — the concepts inside**

    ---

    A sparse autoencoder over the residual stream: 24,576 interpretable
    features. Click a token to see what fired, click a feature to see where it
    fires across the sequence.

    [Read more →](guides/features.md)

-   **Steering — change the answer**

    ---

    Add a feature's direction to the residual stream and regenerate. Same
    prompt, same seed, different output — an A/B you can actually run.

    [Read more →](guides/features.md#steering)

-   **Agents — a flight recorder**

    ---

    Record LLM calls, tool calls and subagents from your own code, then read
    the run as a timeline instead of scrolling logs.

    [Read more →](guides/agents.md)

-   **Your own models**

    ---

    Point it at a network you trained yourself and get a layer map of one real
    forward pass — shapes, activation ranges, dead units, and the first layer
    where a `nan` appears.

    [Read more →](guides/custom-models.md)

</div>

---

## Why local-first

Everything runs on your machine. No account, no upload, no telemetry. The model
weights, the prompts, the traces and the credentials all stay where they are —
which is the only arrangement under which you can point this at real work.

The trade is honest: you need the hardware for whatever model you load. A 0.5B
model is comfortable on a laptop GPU; a 7B one is not.

## Verified, not asserted

Every model in the table below was run end to end on an RTX 4060 laptop in
bfloat16, with attention rows summing to 1.000 and the causal mask intact.

| model | params | shape |
|---|---|---|
| Qwen3-0.6B | 596M | 28 layers × 16 heads |
| Qwen2.5-0.5B-Instruct | 494M | 24 × 14 |
| SmolLM2-360M-Instruct | 362M | 32 × 15 |
| Llama-3.2-1B-Instruct | 1236M | 16 × 32 |
| Gemma-3-270m-it | 268M | 18 × 4 |
| GPT-2 | 124M | 12 × 12 |

Any causal LM on the Hub should work; those are the ones actually tested.

## Licence

MIT.
