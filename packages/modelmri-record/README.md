# modelmri-record

Record what your agent actually did — LLM calls, tool calls, subagents,
retrieval, errors — and look at it on a timeline instead of scrolling logs.

```bash
pip install modelmri-record
```

**Stdlib only.** No torch, no numpy, no SDK pins. Instrumenting an agent
shouldn't cost you a 2.5 GB install.

## Use

```python
from modelmri_record import trace, step

with trace("fix-failing-tests"):
    step("llm_call", name="plan", input=prompt, output=answer,
         duration_ms=1200, tokens_in=900, tokens_out=200)

    with step("subagent", name="test-runner"):
        step("tool_call", name="pytest", input="-q", output="3 failed")
```

Nesting is automatic — a `step` used as a context manager becomes the parent of
everything recorded inside it, and its duration is measured for you.

### The kinds

The first argument is the kind, and it is a closed list — a viewer refuses a
whole document containing a kind it does not know, so this is the one argument
worth checking against the page. `modelmri_record.KINDS` is the same list at
runtime.

| kind | what it is |
|---|---|
| `llm_call` | a call to a model |
| `tool_call` | a tool, a shell command, a function |
| `subagent` | a nested agent; use it as a context manager and everything inside becomes its children |
| `mcp_call` | a tool reached over MCP, kept apart from `tool_call` because the transport is the thing that fails |
| `user_turn` | a person said something |
| `error` | a failure worth its own step; also synthesised for you when an exception escapes the `trace()` block |
| `retrieval` | fetching candidate documents — a vector store, a search index, a grep |
| `embedding` | text to vector |
| `rerank` | reordering candidates against the query |
| `guardrail` | a policy check on the way in or out — **not** `error`, since a guardrail that fires did its job |

A kind this recorder does not recognise is still recorded, and it prints one
line saying so. It does not raise, and it does not drop the step: your agent
must not fall over because a step was named wrong, and a run you cannot see is
still a run worth keeping.

Auto-instrument an SDK instead:

```python
from modelmri_record import instrument_anthropic
instrument_anthropic()      # every Messages.create is now an llm_call step
```

## Where traces go

POSTed to a running ModelMRI viewer on `http://127.0.0.1:5900`. If nothing is
listening, they're written to `./modelmri-traces/*.json` to import later — so
you can record on a box that has no viewer and look at it somewhere else.

To view them: `pip install modelmri && modelmri serve`.

## Credentials are redacted by default

Agent prompts routinely contain the key the agent was handed. A recorder that
writes those to disk verbatim is a liability dressed as an observability
feature, so redaction is on unless you switch it off:

```python
with trace("run"):                      # default scrubber
with trace("run", redact=my_function)   # your own str -> str
with trace("run", redact=False)         # verbatim, deliberately
```

Covered: `sk-…`, `hf_…`, `pypi-…`, `ghp_…`/`github_pat_…`, `xoxb-…`, Google
`AIza…`, AWS key ids, `Bearer …`, and whole PEM private-key blocks.

Patterns are deliberately narrow — known credential shapes, not "anything
high-entropy". A redactor that eats hashes and UUIDs makes traces useless, and
a useless trace gets the feature turned off, which protects nobody. Add your
own shapes:

```python
from modelmri_record.redact import make_redactor
red = make_redactor([r"ACME-[0-9]{6}"])
```

## It will not take down your app

Recording is best-effort by contract. If the viewer is unreachable, the disk is
read-only, or the payload won't serialise, it gives up quietly. A tracing
library that can raise is one nobody leaves switched on.

Quietly is not the same as secretly. A viewer that answers and *refuses* the
document — an unknown step kind, a malformed field, a version older than the
recorder that wrote the run — is a different thing from one that was never
running, and that one prints the refusal in the viewer's own words before the
run goes to disk.

Traces still open when the process exits are flushed by an `atexit` hook —
a crash or a `SIGTERM` is exactly the run you most wanted to look at.

## Licence

MIT. Part of [ModelMRI](https://github.com/muhammadmahadazher/ModelMRI).
