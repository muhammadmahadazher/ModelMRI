# modelmri-record

```bash
pip install modelmri-record
```

Stdlib only. See [Recording agents](../guides/agents.md) for the guide; this is
the surface.

## `trace(name, endpoint=..., redact=True)`

Context manager. Everything recorded inside becomes one trace, delivered on
exit.

| argument | default | meaning |
|---|---|---|
| `name` | — | what the run is called in the viewer |
| `endpoint` | `http://127.0.0.1:5900/api/traces/import` | where to POST |
| `redact` | `True` | `True` for the default scrubber, a `str -> str` callable, or `False` for verbatim |

An exception raised inside is recorded as an `error` step and then re-raised —
tracing never swallows your errors.

## `step(kind, name="", ...)`

Records one step. Usable bare, or as a context manager to nest everything
inside it underneath.

| argument | type | notes |
|---|---|---|
| `kind` | str | `llm_call`, `tool_call`, `subagent`, `mcp_call`, `user_turn`, `error` |
| `name` | str | free text; shown on the timeline |
| `input` / `output` | any | non-strings are JSON-encoded, falling back to `repr` |
| `duration_ms` | int | measured for you when used as a context manager |
| `tokens_in` / `tokens_out` | int \| None | shown on the step |
| `error` | bool | marks the step failed |

Outside a `trace()` it returns a falsy no-op that still supports `with`, so
instrumentation left in library code costs nothing for callers who never opted
in — including callers on worker threads, where contextvars do not reach.

## `instrument_anthropic()`

Wraps `anthropic.resources.messages.Messages.create` so every call becomes an
`llm_call` step with model, prompt preview, response and token counts. Returns
`False` if the `anthropic` package isn't installed; idempotent.

## `redact.make_redactor(extra, include_defaults=True)`

Builds a redactor from your own patterns plus the built-ins.

```python
from modelmri_record.redact import make_redactor
red = make_redactor([r"ACME-[0-9]{6}"])
```

`redact.default_redactor(text)` is the built-in scrubber if you want to call it
directly.

## Guarantees

- **It never raises into your app.** Unreachable endpoint, read-only disk,
  unserialisable payload, cyclic object graph — all degrade quietly.
- **Credentials are removed before anything leaves the process**, including
  from payloads the recorder itself truncated.
- **Delivery is idempotent.** The shutdown flush and the normal exit path
  cannot double-write a run.
- **Parentage is per-task.** Concurrent asyncio tasks each get their own view
  of the ancestry, so parallel agents produce a correct tree.

## Trace format

```json
{
  "id": "9f2a1c4de8b7",
  "name": "fix-failing-tests",
  "started_at": "2026-08-08T09:14:22Z",
  "meta": { "recorder": "modelmri-record/0.1.2" },
  "steps": [
    { "id": "…", "parent_id": null, "kind": "llm_call", "name": "plan",
      "started_ms": 0, "duration_ms": 1200, "input": "…", "output": "…",
      "tokens_in": 900, "tokens_out": 200, "error": false }
  ]
}
```

POST one of these to `/api/traces/import` to load a trace recorded elsewhere.
