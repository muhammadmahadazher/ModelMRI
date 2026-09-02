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

If your checkout lives on Google Drive, OneDrive, Dropbox or iCloud, that
script notices and builds in a local temp directory instead of in `frontend/`,
copying the output back. It prints that it is doing so. Those filesystems
evict and corrupt `node_modules` — forty thousand small files is the case they
handle worst — and the resulting failure looks like a broken toolchain rather
than a broken filesystem: `package.json` reads as zero bytes, or `tsc` reports
"not recognized as an internal or external command" with the binary sitting
right there. Pass `--in-place` to build where the sources are anyway.

The work directory is reused between runs, so only the first build pays for
`npm ci`. Measured on a Drive checkout: 27s cold, 6s warm.

## Quality gates

Run what your change touches; CI runs all of it.

```bash
uv run ruff check .
uv run ruff format --check .
uv run pytest tests packages/modelmri-record/tests -q
```

The suite runs in parallel by default — `pyproject.toml` sets `-n auto`
capped at 8 workers, distributed per file. Measured on a 24-core machine:
276s serial, 80s at 8 workers. The cap is about memory rather than cores;
each worker is a fresh interpreter importing its own torch.

When a test is actually failing, turn it off:

```bash
uv run pytest tests/test_thing.py -n 0 -x   # serial, ordered output, -x stops where you expect
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

## Provenance

Every method here is implemented independently from the paper that
describes it, and the paper is named — in the module docstring and in the
changelog entry. Sources are for checking claims, not for lifting designs.
No code is copied from a repository that does not ship a usable license;
`modelmri/model_diff.py` declines to port two crosscoder implementations for
exactly that reason and says so. If your contribution brings anything you
did not write — code, data, text, a font, a fixture — say so in the pull
request, name where it came from and under what license, and make sure that
license allows it to sit under this repository's. The full list of what the
project ships from elsewhere is [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

## AI-assisted contributions

Using an AI assistant to write code, tests or documentation is fine here,
and much of this repository was built that way. What does not change is who
is responsible: the human who opens the pull request. Concretely, you —
not the tool — have:

- **read and understood** every line you are submitting, and can explain
  why it is there;
- **run it**, including the regression test against the unfixed code;
- **validated the measurement** — the one rule above applies to generated
  code exactly as it applies to typed code, and a generated implementation
  of a published method is checked against the primary source (the paper,
  the reference `cfg.json`, the format specification), not against what the
  assistant said the method does;
- **checked the license position** — generated code that reproduces a
  third-party implementation is third-party code, and the provenance rule
  applies to it; if you cannot tell where it came from, do not submit it;
- **disclosed sources** the assistant drew on when you know them, and the
  fact of assistance when it is material to review.

A pull request that its author cannot explain is not accepted, whoever or
whatever wrote it.

## Licensing and the CLA

By contributing, you agree that your contribution is licensed under the
license of the files you change — AGPL-3.0-only for the application,
Apache-2.0 for the packages listed in [LICENSING.md](LICENSING.md). New
files carry an `SPDX-License-Identifier` header saying which.

A Contributor License Agreement is drafted in [CLA.md](CLA.md): a license,
not an assignment — you keep your copyright, and the project promises that
your accepted work stays available under the license it was contributed
under. It is pending legal review and **is not in force**; nobody is asked
to sign it until it is, and the asking will be done on the pull request by
a CLA bot from the first external contribution onward. Until then the
paragraph above is the whole inbound rule.
