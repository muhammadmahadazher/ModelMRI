# ModelMRI 1.0 — the release that changes what the tool IS

The last four releases were the same shape: measure one more thing about a
language model, and measure it honestly. That shape is finished. Every theme
in `ROADMAP.md` has shipped, the sidecar is in, and the tool now does what it
set out to do.

It is also, in the maintainer's own words, **dull, hard to navigate, and not
attractive enough to convince anybody to use it.** That is the finding that
sets this release, and it is a fair one. A measurement tool nobody opens is a
measurement tool that measures nothing.

So 1.0 is not one more theme. It is three claims the tool cannot currently
make:

1. **It opens image models too.** Diffusion, text-to-image, object detection,
   vision transformers. Not a second product — the same MRI, on a model whose
   output is pixels instead of tokens.
2. **It has no equal on features.** Everything a competitor offers and this
   does not, built — or built better, which for a local-first tool that
   refuses to fabricate usually means a different design rather than a copy.
3. **It is worth looking at.** Navigation by what you are inspecting rather
   than by which panel happens to be next. More than two themes. Motion and
   graphics that carry information rather than decorate it.

The rules from every previous release still hold and are not restated per
item: nothing hardcoded, every number measured live on this machine, every
refusal a sentence, memory bounded, edge cases handled, and a receipt on
anything a reader might quote.

---

## Theme A — Image models

The single largest expansion of what the tool can open. Everything here obeys
the same discovery contract the text side already has: pull from HuggingFace,
read from Ollama, or find what is already on this disk — and refuse by name
when it finds something it cannot open, rather than guessing.

**A1. Detect what kind of image model this is, before opening it**
A checkpoint is a UNet, a DiT, a ViT, a CLIP tower, a detection head, or
something this does not know. The kind decides every panel that follows, and
guessing it wrong produces a diagram of a model that is not there.
*How:* Extend `modelmri/discover.py`. Read `model_index.json`, `config.json`
`architectures`, safetensors tensor-name prefixes, and GGUF metadata — the
same read-the-file-rather-than-assume rule `vla.py` was corrected to follow.
*Caveat:* An unknown architecture is a refusal naming what was found, never a
best-effort render. A diagram of the wrong model is worse than no diagram.

**A2. Cross-attention over the prompt, per denoising step**
Which words the image is actually attending to, and when. Early steps decide
layout, late steps decide texture; a single averaged map hides that entirely.
*Ahead of:* DAAM publishes maps for Stable Diffusion specifically and is
research code. Nothing local, general and installable does this across
families.
*Caveat:* Attention is the weak, correlational version — this project's own
standard, and the reason A3 exists.

**A3. Interventional token knockout for image models**
Remove one prompt token, regenerate at the same seed, measure what moved.
The interventional counterpart to A2, following the same logic that made
`patch.py` stronger than an attention heatmap.
*Caveat:* Same seed is doing the work. Without it the difference is sampling.

**A4. Where the denoiser committed**
Per-step latent divergence: at which step does the image stop changing
materially? Users pay for 50 steps and often buy nothing after 20, and nothing
local tells them which.

**A5. Object detection and classification: what the model looked at**
Grad-CAM-family attribution for CNN and ViT detectors, plus per-class
confidence with the calibration caveat this project already applies to
probability mass elsewhere.
*Caveat:* Softmax confidence is not probability of correctness. Say so on the
panel, not in the docs.

**A6. One `.mri` for an image run**
The recording format carries a diffusion or detection run the same way it
carries a generation — so an image result is shareable, diffable and
CI-gateable with the machinery that already exists.

---

## Theme B — Close every competitor gap

Not guessed. A five-lens sweep across mechanistic-interpretability tooling,
observability platforms, model-debugging tools, vision/diffusion
interpretability and robotics analysis raised **79 candidate gaps**; each was
then checked against this repo before it counted, and **37 survived**. Three
were already shipped and are recorded as such.

The standing rule: **parity is not the target.** For each gap, either build
the better version or write down why the competitor's approach is wrong for a
local-first tool that refuses to fabricate.

### B0. The one that is a hole in our own posture — do this first

**Scan the weights before loading them.** ModelMRI downloads arbitrary
checkpoints from the Hub onto somebody's laptop, checks the *size* with
`capacity.guard`, and then calls `from_pretrained` on them without ever asking
what is inside. It also accepts user-supplied `adapter.py` and TorchScript as
a documented feature. It already knows this risk class exists — `circuit.py`
reads `.pt` through a restricted unpickler — and applies the defence in
exactly one place.

*Who has it:* promptfoo's ModelAudit scans 30+ formats for malicious pickle
opcodes, decode-exec chains, unsafe Keras Lambda layers, embedded
PE/ELF/Mach-O, hidden credentials and zip bombs.

*How we beat it:* every reader this needs is already here — the safetensors
header parser in `fit.py`, the stdlib GGUF reader, the restricted unpickler in
`circuit.py`. promptfoo scans a file you point it at; ModelMRI can **refuse to
load one**, enforced on the download path beside the disk and VRAM refusals.
An unrecognised format reports as *unscanned, with the reason* — never clean —
the rule `gguf_read` already applies to unknown ggml types.

### B1. Datasets and experiments as first-class objects

The single load-bearing abstraction all thirteen observability platforms
share, and the one this does not have. `sweep` runs one metric over many
prompts; `diff` compares two `.mri` of the *same* prompt. Neither answers
"did my edit help on the 40 cases I care about".

*How we beat it:* every competitor's experiment row holds an output and a
score. Ours can hold the output, the score, **and the receipt plus the
internals that produced it** — so a regression row says the top-5 head ranking
changed and the patching site at (attn, L14, p7) flipped sign, not just
"faithfulness 0.71 → 0.63". JSONL on disk, no server; comparison as a
torch-free extension of `modelmri diff` so it runs in CI in milliseconds.

### B2. The cheap ones with real value

- **Trace → dataset.** A recorded failure currently leaves the loop. Curation
  needs no model at all, so we can do offline exactly what Braintrust needs a
  cloud LLM for. The row is *evidence*; naming the failure mode is the
  fabrication we already refuse.
- **Resumable long runs.** A sweep that dies at prompt 180 of 200 starts over.
  Losing four hours to a sleeping laptop is a worse failure than any missing
  feature here. `budget.py` already prices the remainder in exact passes.
- **Trajectory comparison.** Everybody scores this with an LLM judge. It is a
  sequence alignment — exact, offline, milliseconds — and it fits the
  no-verdicts rule: report *2 steps missing, 1 extra, 3 with changed
  arguments*, never "Plan Adherence 0.71", because a shorter path is not a
  worse path.
- **A named scorer library**, but only the metrics that need no model — and
  each one carrying its own **measured** error rate, the way `steer_vectors`
  publishes CAA 16.0% / RepE 13.0% against its own shuffled null. A catalogue
  whose entries carry a measured false-positive rate is a different product
  from one whose entries carry a docstring.
- **Numeric health and a weight/architecture table** for the loaded model —
  what Netron and TensorBoard's Debugger V2 give, on a model that is live.

### B3. Interpretability the field has and we do not

**STATUS 2026-08-25 — every item below is built.** `gradients.py` (integrated
gradients, refusing when the completeness gap is a large share of the move it
claims to explain), `anchors.py` (minimal sufficient token sets, precision as a
Wilson interval with the sample size beside it), `patch_screen.py` (attribution
patching as a screen, measuring its own agreement with the exact grid on the
sites it shortlists), `ov_circuits.py` (QK/OV as factored matrices, never
forming the 92 TB vocabulary square), `neurons.py` (raw neurons with NMF, for
the models that have no SAE — which is most of them), `saes.py`'s
`ce_recovered` (SAE fidelity in output space, with the ablation floor named
because it moves the answer 22 points), and `head_corpus.py` — the one this
file called "embarrassing to lack".

Each was built, then handed to an independent skeptic paid to refute it. All
six of the modules in that pass came back with defects — 66 of them, 9
blockers — and every blocker is now verified fixed by MUTATION rather than by
a passing suite: break the claim, run the tests, confirm they go red. Two were
still vacuous when re-checked and are recorded in the commits that fixed them.

Gradient attribution with the completeness check, anchors (minimal
*sufficient* token sets with measured precision), counterfactual generation,
attribution patching as a first-order screen before the exact grid, QK/OV
circuits as factored matrices, a raw-neuron browser with NMF for models with
no SAE, and SAE fidelity metrics (CE-recovered, L0) on your own model.

Plus the one that is embarrassing to lack: **corpus evidence for an attention
head**, the way we already have it for an SAE feature.

### B4. Robotics, now that the sidecar exists

**STATUS 2026-08-25 — built; the MCAP gap below is closed and the `.rrd` one
stands.** `vla_ood.py` scores every
frame against a reference set the payload names, in Mahalanobis distance over
the directions that reference actually varies in, with the episode's own rows
held OUT of the reference, its mean, its covariance and its null. A distance
and a percentile, never a boolean — "OOD" as a verdict would be a threshold
somebody chose. `vla_data.dataset_action_stats` summarises a whole dataset's
recorded actions in bounded memory (3.2 MB at 20,000 rows and at 80,000), and
carries the publisher's own statistics beside the measured ones so a
disagreement between them is visible. ACT policies open in the perception half,
and `analyse()` refuses to draw attention a ResNet does not have while naming
the occlusion sweep that genuinely works on it.

`robot_export.py` writes MCAP, and `POST /api/vla/export` now reaches it. THE
GAP WAS the retrieval: `write` needs a `Sweep` — rows plus the metric, unit,
dataset, policy, camera and strides that say what the rows ARE — and
`vla_sweep.stored` returns bare rows, so a route would have had to rebuild the
rest by guessing at a unit, which is what that module exists to prevent. It is
closed by storing the missing half rather than by inferring it: `vla_sweep`
grew a second table, `vla_sweep_run`, holding the unit, both strides, the
counts, the duration and the failure sample, written by `save()` in the same
transaction as the rows; and `retrieve()` reads a whole `Sweep` back out of it.

The refusals are the load-bearing part. Rows saved before that table existed
have no run record, and `retrieve()` names the migration and the fix rather
than reading the unit out of `METRICS` — a number arriving in Foxglove under a
unit ModelMRI supplied from memory is indistinguishable from one that was
measured in it. Two runs superimposed under one set of keys (rows are keyed by
episode and timestep, so a coarser second run leaves the finer run's extra
frames behind) are refused with both strides named, because a `Sweep` states
one stride for all of its rows. The route opens the sweep's own dataset for the
clock and refuses when it cannot, since `reader.fps` defaults to 10 for the
decoder's arithmetic and exporting that default would draw a seconds axis
nobody timed. `mcap` is still not a dependency: absent, the route answers 409
with the install command.

The `.rrd` half refuses on its own reasoning: Rerun's logging API moves between
releases, an `.rrd` is read by the SDK version that wrote it, and nothing here
has been run against an installed rerun-sdk — so emitting one would publish a
file whose correctness is a guess.

OOD scoring per frame, a synchronised multi-track episode timeline, export to
MCAP and `.rrd` so findings open in Foxglove and Rerun, dataset-level action
statistics, and ACT-family policies — the most common architecture in the
category, which we cannot open at all.

*(Predicted-vs-recorded action was raised as a gap and shipped as ROADMAP #50
while this analysis was running.)*

### What we are deliberately not building

Cloud-shaped features that would require abandoning local-first: hosted
leaderboards, seat-priced annotation, remote inference against a shared
cluster. And every **score** a competitor sells that we would have to
fabricate — risk scores, OWASP compliance letters, Plan Adherence numbers.
Where the underlying question is real, we answer it with counts and a receipt
instead.

---

## Theme C — The tool people actually want to open

**C1. Navigate by what you are inspecting**
Today the page is one long column of panels in build order. It should be
categories: text-to-text, text-to-image, robot policy, agents — and All —
with sub-categories beneath, so somebody who came to look at a diffusion model
is not scrolling past the SAE panel to find it.

**C2. More than light and dark**
Multiple palettes, chosen deliberately rather than inverted. The current two
are good; two is not a system.

**C3. Motion and graphics that carry information**
Not decoration. The existing measurement-rule dividers and section dots are
the right instinct; the release extends them into diagrams, transitions that
show where a value came from, and illustration where a paragraph currently
sits.

**C4. Learn the craft, then apply it**
Read design.google's guides and blogs end to end, and the design and motion
references the maintainer has collected. Implement from those principles
rather than from taste.

---

## Theme D — Say it properly

**D1. A README somebody finishes reading**
GIFs in BOTH themes — the current ones are dark only, which sells half the
product. Graphics and icons instead of paragraphs wherever a paragraph is
doing a diagram's job.

Optimised twice over, for two different readers:
- **SEO** — so a person searching for what this does finds it.
- **GEO** — so an assistant asked "how do I see inside a local model" cites it,
  with claims specific and checkable enough to be quotable, which is the same
  property that makes them honest.

**D2. Announce it**
X, Substack and Hacker News, once the release is real. HN in particular reads
overclaiming as a tell, which suits a tool whose entire pitch is that it
refuses to overclaim.

---

## What "done" means for 1.0

The bar is unchanged and it is the one that has held all year:

- nothing hardcoded — every value read, measured or refused;
- live, real outputs — no fixtures standing in for measurements;
- every edge case handled, and every refusal a sentence naming the fix;
- memory-efficient, and honest about what a thing will cost before it costs it;
- no known bugs, no open vulnerabilities;
- and every number carrying enough provenance that somebody could disagree
  with it.

A release that looks better and measures worse is not this release.
