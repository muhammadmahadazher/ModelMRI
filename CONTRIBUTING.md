# Contributing to ModelMRI

Thanks for helping make model internals legible. Contributions of code, tests,
documentation, model adapters, accessibility fixes, and carefully scoped design
improvements are all welcome.

## Before you start

1. Search [existing issues](https://github.com/muhammadmahadazher/ModelMRI/issues).
2. Open a proposal before large architectural work so we can agree on scope.
3. Read the one rule below. It is the whole culture of this repository.

## The one rule

**Do not ship a measurement you have not verified.**

ModelMRI's entire value is that the numbers on screen correspond to what the
model actually did. A visualization that looks plausible and is wrong is worse
than no visualization, because it is confidently misleading — and interpretability
work is exactly the domain where nobody has an independent way to notice.

Things this repository has actually shipped and had to fix:

- An SAE fed the residual stream *entering* a block when it was trained on the
  stream *leaving* it. No error, no crash — just confident features describing
  activations the autoencoder had never seen.
- A panel describing token ids from one model using the weights of another,
  because a load completed while a generation was streaming.
- A progress bar verified by reading its `width`, which was correct, while its
  `background-image` referenced a CSS variable that does not exist, so it
  painted nothing.

The pattern in all three: *the observable result matched what I expected, so I
stopped looking.*

So, concretely:

- **Write the regression test first and run it against the unfixed code.** If
  it does not go red, it is not testing anything. Say so in the PR.
- **Verify the thing a user sees, not a proxy for it.** Read the computed
  style, count the drawn pixels, check the button state *after* the operation
  and not only during it.
- **Prefer refusing to guessing.** A hook point that can't be placed, an SAE
  whose `d_in` doesn't match, a checkpoint missing tensors — raise with a
  message that says what was wrong. Every one of those was once a silent
  mismapping.

## Development setup

```bash
uv sync --dev
uv run modelmri            # serves the API and the built UI on :5900
```

The frontend is built and shipped inside the wheel, so you only need Node when
changing it:

```bash
cd frontend
npm ci
npm run dev                # Vite dev server, proxies the API
```

```bash
uv run python scripts/build_frontend.py   # bake the UI back into modelmri/static
```

## Quality gates

Run what your change touches; CI runs all of it.

```bash
uv run ruff check .
uv run ruff format --check .
uv run pytest tests packages/modelmri-record/tests -q
```

```bash
cd frontend && npm run build      # includes tsc --noEmit, strict
```

`modelmri-record` has **zero dependencies on purpose** — it goes into other
people's agents, where a dependency is a liability. A PR that adds one to
`packages/modelmri-record/pyproject.toml` needs a very good argument.

## Good first contributions

- **Another architecture.** [`modelmri/runtime.py`](modelmri/runtime.py) finds
  a model's blocks and residual stream; [`modelmri/saes.py`](modelmri/saes.py)
  places the SAE hook. Both currently assume a GPT-2-shaped decoder. Widening
  them means a test that asserts against real tensors, not a mock.
- **Layer statistics for custom models.** [`modelmri/custom.py`](modelmri/custom.py)
  reports shapes, activation ranges, dead units and non-finite counts. Gradient
  norms and per-layer weight histograms would answer the next obvious question.
- **A bug from your own use.** The most valuable issues here have been the ones
  where a number looked fine and wasn't.

## Pull requests

- One coherent change per pull request.
- Explain the user problem, the behavior you chose, and what you traded away.
- Include the regression test, and say what it did against the unfixed code.
- Update the docs when behavior, commands, APIs, or panels change.
- Include a screenshot or short recording for visual changes.
- Imperative mood in commit messages.

By contributing, you agree that your contribution is licensed under the
[MIT License](LICENSE).
