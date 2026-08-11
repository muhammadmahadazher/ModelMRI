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
| `GET` | `/api/attention/attribute` | Attribute Tokens |
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
| `GET` | `/api/traces/{trace_id}` | Trace Get |
| `POST` | `/api/traces/import` | Traces Import |


## Other

| method | path | notes |
|---|---|---|
| `GET` | `/api/lens` | Lens |
| `GET` | `/api/paths` | Where |
| `GET` | `/api/pull/progress` | Pull Progress |
| `POST` | `/api/patch` | Patch Trace |

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
