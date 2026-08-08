## What changed

<!-- The user-facing problem, and the behaviour you chose. -->

## How you know it works

<!--
Not "tests pass" — what did you actually observe?

If this fixes a bug: what did the new test do when you ran it against the
UNFIXED code? If it passed, it isn't testing the bug.

If this changes something visual: what did you measure? Computed style, drawn
pixels, the control's state AFTER the operation and not only during it.
-->

- [ ] `uv run ruff check . && uv run ruff format --check .`
- [ ] `uv run pytest tests packages/modelmri-record/tests -q`
- [ ] `cd frontend && npm run build` (strict `tsc` included)
- [ ] Regression coverage added, and confirmed red against the unfixed code
- [ ] Docs updated if behaviour, commands, APIs, or panels changed
- [ ] Screenshot or recording attached for visual changes

## Correctness of the measurement

<!-- Skip only if this PR touches nothing that produces a number on screen. -->

- [ ] Activations are read at the hook point the artefact was trained for
- [ ] Nothing describes a generation that a different model produced
- [ ] A mismatch that can't be resolved raises, rather than being approximated
- [ ] No credentials, model weights, traces, or personal data are included
