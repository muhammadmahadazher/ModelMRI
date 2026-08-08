# HTTP API

Everything the UI does goes through this API, so anything you can see you
can script. Generated from the running app's OpenAPI schema.

Base URL: `http://127.0.0.1:5900`. Interactive docs: `/docs`.

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
| `POST` | `/api/model/load` | Load Model |
| `GET` | `/api/model/progress` | Load Progress |
| `POST` | `/api/model/prompt` | Prompt |
| `GET` | `/api/models/discovered` | Models Discovered |
| `GET` | `/api/models/local` | Models Local |

## Discovery

| method | path | notes |
|---|---|---|
| `GET` | `/api/hub/auth` | Hub Auth |
| `POST` | `/api/hub/signin` | Hub Signin |
| `POST` | `/api/hub/signout` | Hub Signout |
| `GET` | `/api/ollama` | Ollama Status |
| `POST` | `/api/ollama/pull` | Ollama Pull |

## Attention

| method | path | notes |
|---|---|---|
| `GET` | `/api/attention` | Attention |
| `GET` | `/api/attention/meta` | Attention Meta |
| `GET` | `/api/vla/attention` | Vla Attention |
| `GET` | `/api/vla/attention/meta` | Vla Attention Meta |

## Features

| method | path | notes |
|---|---|---|
| `GET` | `/api/features/summary` | Features Summary |
| `GET` | `/api/features/{feature_id}` | Feature Detail |
| `GET` | `/api/sae` | Sae Status |
| `POST` | `/api/sae/load` | Sae Load |
| `GET` | `/api/steer` | Steer Status |
| `POST` | `/api/steer` | Steer |

## Robot policy

| method | path | notes |
|---|---|---|
| `GET` | `/api/vla` | Vla Status |
| `POST` | `/api/vla/analyse` | Vla Analyse |
| `GET` | `/api/vla/episodes` | Vla Episodes |
| `GET` | `/api/vla/frame` | Vla Frame |
| `POST` | `/api/vla/load` | Vla Load |

## Agents

| method | path | notes |
|---|---|---|
| `GET` | `/api/traces` | Traces List |
| `POST` | `/api/traces/import` | Traces Import |
| `GET` | `/api/traces/{trace_id}` | Trace Get |

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
| 422 | the request was malformed, or the model is gated and you have not accepted its licence. |

There is deliberately no 500 path for ordinary failures: an unreachable
Ollama, a stalled download and an out-of-memory load all return a 409 that
says what happened.
