# Catching a change to the model's internals in CI

Nothing else in this category has a regression concept for model internals. The
state of the art for "did my quantisation change the model" is a Reddit thread
and somebody eyeballing two completions.

`modelmri diff` compares two saved analyses of the same prompt and exits
non-zero when something moved, so a repo can check in a baseline `.mri` and
have the pull request that broke it say so.

## The idea

1. Record a baseline once, from a run you trust, and commit the file.
2. In CI, produce a fresh `.mri` from the same prompt.
3. `modelmri diff baseline.mri fresh.mri` — exit 1 means something moved.

## Making the baseline

```bash
modelmri sweep prompts.txt --model gpt2 --layer 0 --out-dir baselines/
```

`--out-dir` writes one `.mri` per prompt. Commit the ones you want to hold the
line on. They are tens of kilobytes: the observation, not the model.

## The CI step

`diff` imports no torch and no transformers — both sides are already measured
and comparing them is arithmetic. A regression check that has to install torch
is a check nobody adds, so this one does not.

```yaml
name: internals
on: [pull_request]

jobs:
  did-the-model-move:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: astral-sh/setup-uv@v5
      - name: Rebuild the analysis from this branch
        run: |
          uv run modelmri sweep prompts.txt \
            --model "${{ vars.MODEL_ID }}" \
            --layer 0 --out-dir fresh/
      - name: Did which head carries the answer change?
        run: |
          for f in baselines/*.mri; do
            uv run modelmri diff "$f" "fresh/$(basename "$f")" --fail-over 0.05
          done
```

## Reading the exit code

| exit | meaning |
|---|---|
| `0` | nothing moved past the threshold, or nothing was comparable |
| `1` | something moved — the output names what |
| `2` | the two files are not about the same run, and were not compared |

`--fail-over X` is in **each metric's own units** — nats for the head ranking
and patching, attention weight for attention. Omit it to fail on anything past
the floor the files themselves record, which is the strictest honest threshold:
a difference below that floor is not one the files can represent.

**A changed generation always fails**, at any threshold. There is no magnitude
at which "the model now says something else" is within tolerance.

## What it will not do

- **Compare two different prompts.** Different tokens, prompt length, layer or
  head count is refused with exit 2 rather than diffed into numbers that look
  like a regression.
- **Threshold a sampled run.** Two `.mri` at temperature > 0 differ for reasons
  that are not the model. The `generate` receipt records whether the run was
  greedy, so this is checked rather than assumed.
- **Treat a missing section as unchanged.** A file with no ranking reports "not
  comparable", never "same". An absent measurement read as a passing one is the
  failure this whole tool is built to avoid.

A dtype or commit difference between the two files is reported as a note rather
than hidden: `patch.py` records bf16 moving a reference gap from 4.000 to 4.467
and changing the reference token itself, so a diff across float formats is
measuring the formats as much as the model.
