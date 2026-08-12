---
description: "Record LLM calls, tool calls and subagents from your own code with modelmri-record, then read the agent run as a timeline of blocks instead of scrolling flattened logs."
---

# Recording agents

An agent run is a tree — the model calls a tool, the tool returns, a subagent
spawns and does the same thing three levels down. Logs flatten that tree into a
scroll. This puts it back.

## Install

```bash
pip install modelmri-record
```

Stdlib only. No torch, no SDK pins, 10.9 KiB. Instrumenting an agent should not
cost you a deep-learning install.

## Record

```python
from modelmri_record import trace, step

with trace("fix-failing-tests"):
    step("llm_call", name="plan",
         input=prompt, output=answer,
         duration_ms=1200, tokens_in=900, tokens_out=200)

    with step("subagent", name="test-runner"):
        step("tool_call", name="pytest", input="-q", output="3 failed")
        step("tool_call", name="ruff", output="clean")
```

A `step` used as a context manager becomes the parent of everything recorded
inside it, and its duration is measured for you. Used bare, it records a single
event.

`kind` is one of `llm_call`, `tool_call`, `subagent`, `mcp_call`, `user_turn`,
`error`. The viewer colours and groups by it.

### Auto-instrument an SDK

```python
from modelmri_record import instrument_anthropic
instrument_anthropic()
```

Every `Messages.create` becomes an `llm_call` step with model name, prompt
preview, response and token counts. Returns `False` rather than raising if the
`anthropic` package isn't installed.

## Where traces go

POSTed to a viewer on `http://127.0.0.1:5900`. If nothing is listening they are
written to `./modelmri-traces/*.json`, so you can record on a headless box and
read them somewhere else:

```python
with trace("nightly", endpoint="http://gpu-box.local:5900/api/traces/import"):
    ...
```

To read them: `pip install modelmri && modelmri serve`, then open the
**Agents** panel. Files can be imported by POSTing them to
`/api/traces/import`.

## Credentials are redacted by default

!!! warning "This is on unless you switch it off"
    Agent prompts routinely contain the key the agent was handed — people put
    credentials in system prompts and tools echo their own config. A recorder
    that writes those to disk verbatim is a liability dressed as an
    observability feature.

```python
with trace("run")                      # default scrubber
with trace("run", redact=my_function)  # your own str -> str
with trace("run", redact=False)        # verbatim, deliberately
```

Covered out of the box: `sk-…` (Anthropic, OpenAI), `hf_…`, `pypi-…`,
`ghp_…`/`gho_…`/`github_pat_…`, `xoxb-…`, Google `AIza…`, AWS access key ids,
`Bearer …` headers, and whole PEM private-key blocks.

Patterns are deliberately narrow — known credential shapes, not "anything that
looks high-entropy". A redactor that eats commit hashes and UUIDs makes traces
useless, and a useless trace gets the whole feature turned off, which protects
nobody.

Add your own shapes:

```python
from modelmri_record.redact import make_redactor

red = make_redactor([r"ACME-[0-9]{6}", r"internal_tok_\w{20}"])
with trace("run", redact=red):
    ...
```

Structural fields — names, ids, timings, token counts — are never redacted.
Removing them would destroy the trace without protecting anything.

## It will not take down your app

Recording is best-effort **by contract**:

- Viewer unreachable → falls back to a file.
- Disk read-only → gives up silently.
- Payload won't serialise → coerced with `default=str`.
- `step()` called outside a `trace()` → no-op, returns `None`.

A tracing library that can raise is one nobody leaves switched on in
production.

Traces still open when the process exits are flushed by an `atexit` hook. A
crash or a `SIGTERM` is exactly the run you most wanted to look at, and
delivery is idempotent so the hook and the normal exit path can't double-write.

## What it does not do yet

- No streaming-response capture for the Anthropic wrapper — only the completed
  call is recorded.
- No OpenAI auto-instrumentation. Use `step()` directly.
- No sampling. Every trace in a `with` block is recorded in full.
