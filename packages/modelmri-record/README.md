# modelmri-record

Record what your agent actually did — LLM calls, tool calls, subagents, errors —
and look at it on a timeline instead of scrolling logs.

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

Traces still open when the process exits are flushed by an `atexit` hook —
a crash or a `SIGTERM` is exactly the run you most wanted to look at.

## Licence

MIT. Part of [ModelMRI](https://github.com/muhammadmahadazher/ModelMRI).
