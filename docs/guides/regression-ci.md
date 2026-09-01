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
modelmri sweep prompts.txt --model Qwen/Qwen3-1.7B --layer 0 --out-dir baselines/
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

`--fail-over X` is in **each metric's own units** — nats for the head ranking,
patching and grounding, attention weight for attention, recovery fraction for
the patching graph. The run itself prints the list, read off the sections that
report actually contains rather than from a fixed list here, and it names only
the sections the threshold can gate at all: three of them are categorical by
construction and a number in "text units" or in tokens is not a threshold
anybody could set. Omit it to fail on anything past the floor the files
themselves record, which is the strictest honest threshold: a difference below
that floor is not one the files can represent.

**A changed generation always fails**, at any threshold. There is no magnitude
at which "the model now says something else" is within tolerance. So does a
patching-graph edge whose control verdict flipped, and for the same reason: a
boolean has no size at which it is within tolerance. Both directions of that
flip are the same finding and both fail — an edge that now clears the controls
it used to fail is reported in those words, not as one that stopped clearing
them.

**The forwarded attribution graph is compared without a floor**, because no
attribution graph records what the tool that computed it could resolve. Which
edges are present and which changed sign are reported as changes — and, having
no magnitude to be judged, they fail at *any* `--fail-over`. Which edges are
strongest is not among them: that is an ordering of the same unjudgeable
weights, so two a hair apart rank in whatever order the last digits fell, and
it is reported alongside the weight that moved as not comparable, with both
numbers printed, rather than judged against an epsilon nobody measured.

## The same gate over a whole set

`diff` holds the line on ONE prompt. The question people ask after an edit is
usually about a set of them — *did this help on the forty cases I care about* —
and averaging those into a single score is exactly what hides a regression: two
cases collapsing and three improving slightly average out to fine.

`modelmri experiments` compares two runs of one dataset case by case. Same
promise as `diff`: no torch, no accelerator, arithmetic over JSONL.

```bash
modelmri experiments before.jsonl after.jsonl \
  --metric faithfulness --higher-is-better --dataset cases.jsonl
```

```yaml
      - name: Did the edit hurt any case?
        run: |
          uv run modelmri experiments \
            baselines/run.jsonl fresh/run.jsonl \
            --metric faithfulness --higher-is-better \
            --dataset cases.jsonl
```

### The direction is not optional

`--higher-is-better` or `--lower-is-better` is **required**, and there is no
table of metric names that decides it for you. KL divergence is better lower
and faithfulness is better higher; a wrong guess inverts every conclusion in
the report while producing output that looks entirely reasonable.

### Reading its exit code

| exit | meaning |
|---|---|
| `0` | ran, and no more cases got worse than `--fail-on-worse` allows |
| `1` | ran, and too many got worse — the output names which |
| `2` | **could not run** |

`2` is the one worth wiring an alert to. An unknown metric, a missing file, or
two runs of different datasets all exit 2, and a gate that could not run is not
a gate that passed. Treating it as either of the others is how a broken
comparison gets read as a green build.

`--fail-on-worse` defaults to `0`: any regression fails. A number above zero is
deciding in advance how much breakage is acceptable, which is a decision worth
making out loud in the workflow file rather than inheriting from a default.

### Bringing an Inspect eval in as a run

A UK AISI Inspect `.eval` log already holds everything an experiment needs —
one row per sample, with what each scorer gave it. `modelmri eval-import`
reads one into the pair of files the gate above compares.

```bash
# Look first. With no --out this writes nothing and prints what the log scored.
modelmri eval-import logs/2026-08-15_arc-easy.eval

# Then convert.
modelmri eval-import logs/before.eval --out before.jsonl --dataset-out cases.jsonl
modelmri eval-import logs/after.eval  --out after.jsonl

modelmri experiments before.jsonl after.jsonl \
  --metric match_correct --higher-is-better --dataset cases.jsonl
```

Same promise as the rest of this page: no torch, no accelerator, no network. An
`.eval` is a zip of JSON and the reader is `zipfile` plus `json`.

**`match_correct`, not `match`.** Inspect's canonical score value is the string
`"C"` or `"I"` — its correct/incorrect markers — and a string cannot be ordered
against another, so a comparison over `match` reports every row as
`unmeasurable`. The import writes the marker through **unchanged** and adds a
separately named numeric column beside it, `C` → `1.0` and `I` → `0.0`, with
that mapping recorded in the experiment file's own `meta.score_markers` and
`meta.derived_scores`. Rewriting `match` from `"C"` to `1.0` in place would
make the file claim a number the log never wrote. Inspect's `P` (partial) and
`N` (no answer) get no number: deciding what partial credit is worth is a
judgement, and one invented here would be indistinguishable from one you made.

**A scorer in the log that already owns `<name>_correct` keeps it**, on every
row, even the rows it did not score. The column is then the log's, not this
converter's, and no derived number is written into any of it — the marker is
left exactly as the log wrote it and `meta.markers_not_derived` names which
marker got no number and why. Filling in only the rows that scorer missed
would put a value nobody measured in a column somebody did, with nothing on
the row saying which was which.

**A sample the eval never scored is not a sample that scored zero.** It lands
as a row with no metrics and the sentence saying so, and a comparison reports
it `unmeasurable` rather than folding a `0` into the distribution. A sample
the log marks **errored** says that instead, quoting what the log recorded;
one that is errored *and* scored keeps its scores, stays measured, and the
error is kept under `meta.sample_errors`.

**A log that states no time is not dated to the import.** `started_at` and the
dataset's `created_at` stay empty rather than being filled with the moment you
happened to read the file, and the printout says the log stated none.

**Every row carries a receipt** naming the log's own sha256, its filename, the
task, the model and the sample id and epoch it came from — so an experiment
file forwarded to somebody else says which log it came out of. The case id is
`<sample_id>:<epoch>`, because a multi-epoch eval runs every sample more than
once and one case id cannot carry two rows.

The `--metric` you want is in the printout, and `experiments` names the others
it found on its second line — a run's metrics are read off the files rather
than guessed at.

### Turning a failure you watched into a case

A run you saw go wrong can become a row in the set rather than leaving the
loop:

```bash
curl -s localhost:5900/api/traces/dataset \
  -H 'content-type: application/json' \
  -d '{"trace_ids": ["<id>"], "name": "regressions", "only_errors": true}'
```

Every case comes back with **no expected answer**. The row is evidence that a
run happened; deciding what the right answer was is a judgement, and one
invented for you would be indistinguishable from one you made.

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
