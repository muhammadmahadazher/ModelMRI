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

Driven by a live analysis rather than a guess, so the list is not written
here: it is produced by comparing what ships against what every tool in the
category offers, verified against the repo before anything is called a gap.

The standing rule for this theme: **parity is not the target.** For each gap,
either build the better version or write down why the competitor's approach is
wrong for a local-first tool that refuses to fabricate. Copying a cloud
platform's feature into a laptop tool usually produces a worse version of
both.

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
