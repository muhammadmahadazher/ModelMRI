---
description: "Causal tracing with activation patching — take two prompts that differ in one fact, move an activation from the run that knows the answer into the run that does not, and read off which layer and token carry it."
---

# Patching: where the answer is decided

Every other panel in ModelMRI takes one prompt and removes something from it —
a head, a token, a feature — and reports how far the answer moved. That
answers *what mattered*. It does not answer *where the thing is*.

Patching answers that, by using two prompts instead of one:

```
clean    "The Eiffel Tower is located in the city of"   ->  " Paris"
corrupt  "The Colosseum is located in the city of"      ->  " Rome"
```

Run both. Then run the corrupted prompt again with **one activation replaced**
by the clean run's, at one layer and one token position. If the answer comes
back toward " Paris", that site was carrying the fact. Repeat for every
(layer, position) and you have a map.

## Reading the grid

The score is the share of the gap between the two answers that the patch
restores. **1.0** is the clean answer, **0.0** the corrupted one.

Two things about that number surprise people, and both are real:

- **It is signed.** A patch can push the answer *further* from the clean run
  than doing nothing would. On the reference pair 5 of 132 sites did, the worst
  at −0.157. This is the only ranking in ModelMRI that is not a KL, and that is
  why: KL is unsigned and would report those the same as a site that did
  nothing.
- **It is not capped at 1.0.** A single site can overshoot. `gemma-3-270m-it`
  reads **1.010** at its last layer on the reference pair.

## Three grids, because "where" and "through what" are different questions

The residual stream at a layer is everything that has happened so far. Patching
it tells you the fact is present at that point — not how it got there. So the
panel also patches the two sublayer **outputs**, and they do not agree:

| grid | what it claims |
|---|---|
| `residual stream` | the fact is readable here, at the input to this block |
| `attention` | attention moved it in here |
| `MLP` | the MLP wrote it here |

Measured on the reference pair, float32, across three architectures:

| model | layers | residual peak | attention peak | MLP peak |
|---|---|---|---|---|
| `gpt2` | 12 | +0.844 · L11 · `of` | +0.232 · L9 · `of` | **+0.365 · L0 · `um`** |
| `Qwen/Qwen2.5-0.5B-Instruct` | 24 | +0.999 · L23 · `of` | +0.478 · L21 · `of` | **+0.721 · L0 · `os`** |
| `google/gemma-3-270m-it` | 18 | +1.010 · L17 · `of` | +0.736 · L12 · `of` | **+0.483 · L3 · `osseum`** |

The pattern is the same in all three and it is the standard causal-tracing
result: **the MLP peak sits on a subject token in an early layer** — `um`,
`os`, `osseum` are all pieces of "Colosseum" — **while the attention peak sits
on the last token, late** (75%, 87% and 67% of the way through the stack). Early
MLP writes the fact; late attention moves it to where the prediction is made.
The residual grid contains both and shows you only the destination.

## What the controls mean

A grid of numbers is not evidence on its own. Editing *anything* at a site
moves the answer somewhat, so each of the strongest sites — eight per grid — is
run again against two controls:

- **a random direction of the same size at the same site**, eight draws. One
  draw would not be enough: at a single site the draws ran from −2.038 to
  +0.616 against a real recovery of +0.427, and the gate moves from 76 of 132
  sites passing on one draw to 20 on all eight.
- **this layer's activation from the next token over**, which asks whether it
  is *this position* or merely this layer.

Ringed cells cleared both. A site that does not clear them is not distinguished
from an edit of that size at that layer, and the panel says so rather than
letting a large number speak for itself.

## Choosing a pair

Most casually-written pairs cannot be used, and both failure modes are
invisible if nothing tells you — each prompt runs fine on its own.

**The two prompts must tokenize to the same length.** Position 3 of one run
only corresponds to position 3 of the other if they do. Of eight natural
minimal pairs, two did not: `Michael Jordan` is 2 tokens where
`Serena Williams` is 4. ModelMRI refuses these and prints both tokenizations so
you can see which word split differently.

**The two prompts must predict different tokens.** The score divides by the gap
between the two answers, and of three casually-written pairs, two produced the
*same* next token — a denominator of exactly 0.000000. `The capital of France
is` and `The capital of Germany is` both answer `" the"` on GPT-2.

Pairs that work well are minimal and concrete: one name, one number, one place
changed, everything else identical.

## Cost

`n_layers × n_positions × 3` forward passes for the grids, plus nine more for
each of the 24 controlled sites. Measured on an RTX 4060, float32:

| model | passes | time |
|---|---|---|
| `gpt2` (12 layers, 11 tokens) | 614 | 5.5 s |
| `google/gemma-3-270m-it` (18 layers, 10 tokens) | 758 | 29.5 s |
| `Qwen/Qwen2.5-0.5B-Instruct` (24 layers, 11 tokens) | 1010 | 31.5 s |

Seconds measured on one machine do not transfer; `passes` and `seconds` both
come back in the response so you can derive a rate on yours.

## Precision

Unlike the SAE feature ranking, patching does **not** require float32. Replacing
a tensor with itself does no arithmetic, so the identity check — patching the
corrupted run with its own activation — comes back at exactly 0.0 in float32,
bfloat16 and float16 alike.

What precision does change is the *reference*. In bfloat16 GPT-2 answers `" T"`
to the corrupted prompt where float32 answers `" P"`, and the gap is exactly
4.000 rather than 4.467 — which also quantises the scores into steps of an
eighth. Compare within a dtype, not across one. The dtype and both reference
tokens come back with every result.

## API

```bash
curl -s localhost:5900/api/patch -H 'Content-Type: application/json' \
  -d '{"clean":"The Eiffel Tower is located in the city of",
       "corrupt":"The Colosseum is located in the city of"}'
```

`422` when the pair cannot be compared — different token lengths, the same
answer, or a gap too small to divide by. Each refusal names what to change.
`409` when there is no live HuggingFace model to re-run: patching needs weights,
so it is not available for a `.mri` recording or an Ollama-served model.
