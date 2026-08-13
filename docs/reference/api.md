---
description: "The ModelMRI HTTP API reference: load models, stream generation over a websocket, fetch attention matrices, load sparse autoencoders, steer features, and import agent traces."
---

# HTTP API

Everything the UI does goes through this API, so anything you can see you
can script. Generated from the app's own OpenAPI schema by
`scripts/gen_api_docs.py` — run it after adding a route.

Base URL: `http://127.0.0.1:5900`. Interactive docs: `/docs`.


## Shared sessions (.mri)

| method | path | notes |
|---|---|---|
| `GET` | `/api/session/export` | Session Export |
| `GET` | `/api/session/state` | Session State |
| `POST` | `/api/session/close` | Session Close |
| `POST` | `/api/session/open` | Session Open |


## Session

| method | path | notes |
|---|---|---|
| `GET` | `/` | Index |
| `GET` | `/api/session` | Session |


## Model

| method | path | notes |
|---|---|---|
| `GET` | `/api/accelerator` | Accelerator |
| `GET` | `/api/hub/models` | Hub Models |
| `GET` | `/api/model/progress` | Load Progress |
| `GET` | `/api/models/discovered` | Models Discovered |
| `GET` | `/api/models/local` | Models Local |
| `POST` | `/api/model/cancel` | Cancel Load |
| `POST` | `/api/model/load` | Load Model |
| `POST` | `/api/model/prompt` | Prompt |
| `POST` | `/api/model/unload` | Unload Model |


## Discovery

| method | path | notes |
|---|---|---|
| `GET` | `/api/hub/auth` | Hub Auth |
| `GET` | `/api/ollama` | Ollama Status |
| `GET` | `/api/ollama/resolve` | Ollama Resolve |
| `GET` | `/api/ollama/size` | Ollama Size |
| `GET` | `/api/ollama/suggested` | Ollama Suggested |
| `POST` | `/api/hub/signin` | Hub Signin |
| `POST` | `/api/hub/signout` | Hub Signout |
| `POST` | `/api/ollama/pull` | Ollama Pull |


## Attention

| method | path | notes |
|---|---|---|
| `GET` | `/api/attention` | Attention |
| `GET` | `/api/attention/ablate` | Ablate Heads |
| `GET` | `/api/attention/ablate/estimate` | Estimate Ablation |
| `GET` | `/api/attention/attribute` | Attribute Tokens |
| `GET` | `/api/attention/baselines` | Compare Baselines |
| `GET` | `/api/attention/control` | Control Ranking |
| `GET` | `/api/attention/diff` | Attention Diff |
| `GET` | `/api/attention/meta` | Attention Meta |
| `GET` | `/api/vla/attention` | Vla Attention |
| `GET` | `/api/vla/attention/meta` | Vla Attention Meta |


## Features

| method | path | notes |
|---|---|---|
| `GET` | `/api/features/ablate` | Ablate Features |
| `GET` | `/api/features/summary` | Features Summary |
| `GET` | `/api/features/{feature_id}` | Feature Detail |
| `GET` | `/api/sae` | Sae Status |
| `GET` | `/api/sae/available` | Sae Available |
| `GET` | `/api/steer` | Steer Status |
| `POST` | `/api/sae/load` | Sae Load |
| `POST` | `/api/steer` | Steer |


## Custom models

| method | path | notes |
|---|---|---|
| `GET` | `/api/custom` | Custom Status |
| `GET` | `/api/custom/candidates` | Custom Candidates |
| `POST` | `/api/custom/load` | Custom Load |
| `POST` | `/api/custom/run` | Custom Run |
| `POST` | `/api/custom/scan` | Custom Scan |
| `POST` | `/api/custom/unload` | Custom Unload |


## Robot policy

| method | path | notes |
|---|---|---|
| `GET` | `/api/vla` | Vla Status |
| `GET` | `/api/vla/datasets` | Vla Datasets |
| `GET` | `/api/vla/episodes` | Vla Episodes |
| `GET` | `/api/vla/frame` | Vla Frame |
| `POST` | `/api/vla/analyse` | Vla Analyse |
| `POST` | `/api/vla/dataset` | Vla Set Dataset |
| `POST` | `/api/vla/load` | Vla Load |


## Agents

| method | path | notes |
|---|---|---|
| `DELETE` | `/api/traces` | Traces Clear |
| `DELETE` | `/api/traces/{trace_id}` | Trace Delete |
| `GET` | `/api/traces` | Traces List |
| `GET` | `/api/traces/search` | Search Traces |
| `GET` | `/api/traces/{trace_id}` | Trace Get |
| `POST` | `/api/traces/import` | Traces Import |
| `POST` | `/api/traces/{trace_id}/steps/{step_id}/adopt` | Adopt Step |


## Other

| method | path | notes |
|---|---|---|
| `GET` | `/api/gguf` | Read Gguf |
| `GET` | `/api/gguf/plan` | Plan Gguf |
| `GET` | `/api/graph` | Graph |
| `GET` | `/api/lens` | Lens |
| `GET` | `/api/paths` | Where |
| `GET` | `/api/pull/progress` | Pull Progress |
| `GET` | `/api/telemetry` | Read Telemetry |
| `POST` | `/api/gguf/load` | Load Gguf |
| `POST` | `/api/otel/v1/traces` | Otel Ingest |
| `POST` | `/api/patch` | Patch Trace |
| `POST` | `/api/quantdiff/behaviour` | Quantdiff Behaviour |

## Head type labels

`GET /api/attention/types?seq_len=24&n_sequences=6&seed=0`

Labels each head induction / previous-token / duplicate-token / sink, or — for
most of them — `label: null`, which is "no type detected" and a result rather
than a gap.

**A label needs all three gates**, and each exists because the previous ones
were measured and found insufficient:

| gate | what it rules out |
|---|---|
| `margin` ≥ 3σ above the head's own null | the score being the null |
| `times_chance` ≥ 1 | significance without effect size — a null with no spread makes any score clear 3σ |
| the offset is the head's `peak` | a habit the head merely has, rather than what it does |

**Two nulls**, and `null_kind` says which was used. Induction and
duplicate-token are gated on matched non-repeating sequences, which is right
for offsets that are only special because the sequence repeats. A
previous-token head attends to i−1 whether or not anything repeats and a sink
attends to position 0 always, so those are gated on chance under the causal
mask instead — their non-repeating null is the same number again.

These are **behaviour on repeated random tokens, not claims about real text**,
and a label must never be read as explaining the ablation ranking: a head can
be labelled and irrelevant, or unlabelled and load-bearing. A byte-level
tokenizer is refused rather than measured badly.

When one label lands on most of a model's heads the response says so — that is
a fact about the model rather than a distinction between its heads.

## Direct logit attribution

`GET /api/attention/direct?position=&top_k=40`

Of the logit the model gave the token it predicted, how much came straight from
each head and MLP down the residual stream. Sited inside the ablation panel
because the two **disagree**: the ranking says what breaks when a head is
removed, this says what a head contributed directly, and a head can be near
zero here and still decide the answer by feeding a later head.

**The reconstruction residual is not optional.** Direct attribution is exact
only if the final normalisation is linear, and it is not. TransformerLens makes
it exact by folding LayerNorm into the weights — which changes the model you
are studying, and once folded nothing in the output says what the folding cost.
Here the model is untouched, the normalisation is frozen at the scale a hook
recorded from the real pass, and the gap is measured and returned: on gpt2
predicting ` Paris`, the components sum to 14.974 against a real 15.141, a
residual of **1.11%**. A chart without that number is claiming a decomposition
it does not have.

**The residual is also the floor.** A component contributing less than the
reconstruction error cannot be told from the reconstruction error, so those are
flagged `unreadable: true` rather than rendered as small. That is not a claim
that the component does not matter.

Every contribution is **shift-corrected** against its own vocabulary mean:
softmax ignores a constant added to every logit, so a component that lifts the
whole vocabulary equally reports zero.

Contributions are **signed** — a component can push against the token the model
chose, and folding that into a magnitude would hide half the mechanism.

The affine form is **verified before anything is reported**: the reconstruction
is compared against the model's own normalisation, with the tolerance derived
from the model dtype's representable step rather than a chosen epsilon. A model
whose norm is something else — a learned gate, a different centring — is
refused rather than attributed through the wrong transform.

## The two lenses

`GET /api/lens?top_k=5&kind=plain|tuned|both`

`layers` is **always the plain reading**, on every `kind`. A tuned reading
arrives beside it in `tuned`, never in its place — a translator fitted to
minimise disagreement with the final distribution will reduce disagreement with
the final distribution, so a caller handed translated rows where it expected
plain ones would have no way to tell the model from the fit.

**Align the two by `layer`, not by index.** The plain lens has one row more:
the model's own final state, which has no translator because it is the answer
rather than a guess at it.

| route | does |
|---|---|
| `GET` `/api/lens/tuned` | whether a translator has been fitted for the loaded model, and what to |
| `POST` `/api/lens/tune` | fit one. `{"texts": [...]}` or `{"file": "corpus.txt"}`, plus optional `steps` |

**Nothing is downloaded.** Pretrained lenses exist on the Hub and fetching one
would break the offline promise the rest of this package keeps, so the corpus
comes from the caller and training happens on this machine.

The response reports **held-out KL per layer** — measured on sequences the
translator never saw — beside the plain KL for the same layer. Training loss is
not reported anywhere, because a translator's training loss is a statement
about the translator. A layer the translator made *worse* shows a negative
gain rather than being clamped to zero.

`caution` is non-empty when the corpus is small relative to the fit: a
translator is `d_model² + d_model` parameters per layer, so a few thousand
tokens leaves it orders of magnitude under-determined. The held-out numbers are
still real; what they are about is text like the training text.

A saved lens is refused if it was fitted to a different model or a different
dtype. Loading one across either would produce a confident, plausible, entirely
wrong reading.

## Receipts

Every measurement route returns a `receipt` alongside its numbers: what
produced them, in a shape a machine can read. `/api/attention/ablate`,
`/api/attention/attribute`, `/api/attention/baselines`, `/api/features/ablate`,
`/api/lens` and `/api/patch` all carry one, and `session/export` writes the
set of them into the `.mri`.

```json
{
  "op": "ablate_heads",
  "request": {"layer": 0, "baseline": "zero", "position": 4},
  "tool_version": "0.10.1",
  "model": "gpt2",
  "revision": "607a30d783dfa663caf39e06633721c8d4cfcd7e",
  "revision_note": "the commit `refs/main` resolves to in the local cache",
  "dtype": "bfloat16",
  "device": "cuda:0",
  "attn_implementation": "eager",
  "seed": null,
  "tokenizer_sha256": "11e818f948f43497",
  "tokenizer_note": "the full fast-tokenizer definition",
  "prompt_sha256": "bbaff4d2ecd5892d",
  "n_prompt_tokens": 9,
  "measured_at": "2026-08-13T16:27:55+00:00"
}
```

Three fields can genuinely fail to resolve, and each answers `null` **with a
note saying why** rather than a plausible default — a receipt that quietly
reports the wrong revision is worse than one that reports no revision, because
the first is trusted and the second is questioned:

- `revision` is read from the local cache, never the network, so it works
  air-gapped. `refs/main` is consulted first; if several revisions are cached
  and no ref says which was loaded, the answer is `null` and `revision_note`
  says naming one would be a guess.
- `tokenizer_sha256` covers the full fast-tokenizer definition where there is
  one — vocabulary, merges, normaliser, pre-tokeniser. Where there is not,
  `tokenizer_note` says the hash is vocabulary-only, because two tokenizers
  with the same vocabulary and different normalisers produce different token
  ids and the two hashes must not be compared.
- `seed` is `null` when the measurement was not seeded. That is not seed `0`.

Receipts carry **no filesystem paths and no usernames**: the model name is
reduced to its basename when it was loaded from a folder, and any absolute
path in `request` is reduced the same way. `tests/test_no_machine_leaks.py`
enforces it.

## Streaming

`GET /ws/generate` (WebSocket). Send `{"prompt": "..."}` and receive
`{"type":"token","text":"..."}` frames, then one `{"type":"done"}`.
A generation that fails mid-stream sends `{"type":"error","message":...}` —
it does not silently close as a success.

## Status codes

| code | meaning |
|---|---|
| 200 | fine |
| 409 | you asked for something in the wrong order, or a dependency is down. The body has an actionable message. |
| 413 | the upload was larger than the limit — a `.mri` is not that big, so it is probably not one. |
| 422 | the request was malformed, the model is gated and you have not accepted its licence, a custom adapter could not be loaded, or a session file could not be read. |

There is deliberately no 500 path for ordinary failures: an unreachable
Ollama, a stalled download, an out-of-memory load and a user's adapter
raising on import all return a status that says what happened.
