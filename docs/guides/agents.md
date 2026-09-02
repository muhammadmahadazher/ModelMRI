---
description: "Record LLM calls, tool calls and subagents from your own code with modelmri-record, then read the agent run as a timeline of blocks instead of scrolling flattened logs."
---

# Recording agents

An agent run is a tree — the model calls a tool, the tool returns, a subagent
spawns and does the same thing three levels down. Logs flatten that tree into a
scroll. This puts it back.

## Two things fill the panel

The **Agents** panel shows recorded runs from two places, and you need the
library for only one of them:

-   **Generations you make in ModelMRI itself.** Every committed generation in
    the playground is stored as a one-step `llm_call` trace — prompt in,
    output out, with the model id, the duration and the token counts. Nothing
    to install and nothing to instrument; load a model, generate, and the run
    is there. Works on any model and either backend.
-   **Runs of your own agent code**, recorded with `modelmri-record` and sent
    to the same store. That is what the rest of this page is about, and it is
    where the tree — tool calls, subagents, nesting — comes from.

Runs of the first kind are labelled `this app` in the list, so a list holding
both stays readable.

## Install

```bash
pip install modelmri-record
```

Stdlib only. No torch, no SDK pins, 30.6 KiB. Instrumenting an agent should not
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
`error`, `retrieval`, `embedding`, `rerank`, `guardrail`. The viewer gives each
one a colour and a shape, groups by it, and lets you filter on it —
`kind:retrieval` in the search box, or a rubric rule that counts steps of one
kind.

The last four are what a retrieval pipeline is made of, and they are separate
kinds rather than four flavours of `tool_call` because the questions worth
asking about a RAG agent are all phrased in them: how much of the wall clock is
retrieval, how often a rerank changed the top document, whether a guardrail
fired. A `guardrail` step is deliberately not an `error` — a guardrail that
fires did its job.

The list is closed. A kind the viewer does not know is refused **on import**,
and refused for the whole document rather than the step: a run missing exactly
the step you were looking for is a worse answer than a run that did not import,
and it is one you would not notice.

A run carried inside a shared `.mri` bundle is the one exception, deliberately:
it is shown with the kind it was given, drawn grey and hatched instead of in a
colour, because refusing to open a file somebody sent you is worse than saying
"recorded, and this build does not know what it is". So a bundle written by a
newer ModelMRI opens here, and the steps this build has never heard of are
still on the timeline and still clickable.

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

## Sending a run to your own collector

If your team already runs Langfuse, Phoenix, Grafana or Honeycomb, a run does
not have to stay here:

```bash
modelmri export --otlp http://localhost:4318
```

That takes the most recent run; name one to pick it (`modelmri traces` lists
them). `--header "Authorization=Bearer ..."` for a hosted collector, and
`--dry-run` prints the body and sends nothing.

It speaks **OTLP/HTTP with a JSON body**, over the standard library. Port 4318
is the usual one; 4317 is gRPC and is not spoken. A collector configured for
protobuf only is refused with a sentence rather than approximated — supporting
it would mean either generated stubs or the OpenTelemetry SDK, and
`modelmri-record` is dependency-free on purpose because it gets imported into
other people's agents.

To export as runs finish rather than by hand:

```python
with rec.trace("nightly", deliver_otlp="http://localhost:4318"):
    ...
```

Off by default, and it exports the redacted document. It runs after the normal
delivery and cannot affect it: a collector being down must not cost you the
trace.

**Which vocabulary the spans speak is printed and stamped.** The `gen_ai.*`
conventions left the main semantic-conventions repo on 2026-06-12 for one with
no releases and no tags, so every span carries
`modelmri.semconv.generation` and the CLI prints it. When the vocabulary
moves, old exports stay readable because they say what they were written
against.

One thing to know when reading the result. A step recorded without a duration
has no end time, and OTLP has no way to express that — `endTimeUnixNano` is
required. Those spans go as zero-length, which on a waterfall looks
instantaneous. They are marked `modelmri.duration.recorded=false` and the CLI
tells you how many there were.


## What it does not do yet

- No streaming-response capture for the Anthropic wrapper — only the completed
  call is recorded.
- No OpenAI auto-instrumentation. Use `step()` directly.
- No sampling. Every trace in a `with` block is recorded in full.
