# Changelog

Notable changes to `modelmri` and `modelmri-record`. Format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/); versioning is
[semantic](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- **Token attribution** — `modelmri/attribute.py`, `GET
  /api/attention/attribute`, and a panel beside the head ranking. It masks one
  token out of every later position's attention, re-runs the model, and reports
  how far the next-token distribution at one position moved, in nats of KL.
  The input-side companion to head ablation, and built around what Phase 0
  measured rather than what the plan assumed:
    - **Index 0 is an attention sink and is reported outside the order.** On
      gpt2 (bf16/cuda, eager, "The capital of France is", last prompt token) it
      scores 4.86309, 2.79x the next candidate; prepending `<|endoftext|>` holds
      index 0 at 4.76083 while the word that moved off it falls 10.5x, to
      0.46107. The score follows the position, not the token.
    - **The attribution position itself is excluded on geometry**, not on size.
      Sparing the diagonal drops it to exactly 0.0 on gpt2 — but it is *not*
      small in general: on Qwen3-0.6B the same position scores 6.24429 and is
      the largest of all 13 candidates.
    - **The chat template dominates.** On Qwen3-0.6B the template's own '\n'
      and 'assistant' score 6.24429 and 2.02161 while every word the user typed
      sits between 3.1e-05 and 7.9e-05, so rows are grouped as `typed`,
      `template`, `generated` or `unknown` and never shown as one ranking.
    - **The scores are not shares and do not add up**, and the direction is not
      fixed: summing the rows shown over-states one joint mask by 1.82x on gpt2,
      while summing only the typed span under-states it by 0.35x on
      gemma-3-270m-it. `sum_of_singles` and `joint_kl` are both returned;
      neither is a correction factor.
  Refusals rather than guesses: an unbatched `[1, S]` only, a model whose
  answer moves when handed an explicit all-ones mask, a model that does not
  read the `position_ids` it is given, an attribution position whose next token
  is a control token, and a position with nothing before it but the sink.
- **A FAQ**, in the docs and in the README, answering what was previously only
  answerable by reading source: which models are actually verified, why
  attention needs eager attention, what a head-ranking KL does and does not
  mean, why an SAE may not exist for your model, where files are stored, and
  how ModelMRI differs from BertViz, TransformerLens, nnsight, Neuronpedia,
  SAELens, Langfuse, Phoenix and LangSmith. The comparison says plainly that
  each of those does its own job better.
- **A cross-platform CI matrix.** The README claimed Windows, macOS and Linux
  and `pyproject` claimed Python 3.10–3.13, while everything was tested on
  ubuntu-latest and one developer's laptop. Four jobs now span all three
  operating systems and all four Python versions.
- **Crawler-facing metadata**: `robots.txt`, `sitemap.xml` and an `llms.txt`
  summary, plus a description, canonical URL, Open Graph tags,
  schema.org `SoftwareApplication` data and a `<noscript>` fallback on the
  demo page — which, being a React app, previously showed a crawler nothing
  but the word "ModelMRI".

### Fixed

- **Attribution accepted a model that ignores `position_ids`.** The agreement
  check ran with an all-ones mask, under which `attention_mask.cumsum(-1) - 1`
  equals `arange(S)` by construction — so a model that derives its own
  positions from the mask agreed perfectly, and then every masked pass billed
  the suffix's phase shift to the one masked token. Written as a toy model it
  returned floor 0.0, `mask_verified` true and a full ranking. A pass with
  deliberately reversed `position_ids` now gates on the answer *moving*
  (measured: gpt2 2.166768, Qwen3-0.6B 0.011300, gemma-3-270m-it 4.616208
  nats). Reversal rather than a shift, because RoPE is invariant to shifting
  every position together: `arange + 1` moves Qwen3 by 5e-06.
- **A model swap mid-request produced a ranking of the wrong model.** Both
  `ablate_heads` and `attribute_tokens` took their epoch check *outside*
  `self._lock`, and `load` holds that lock across the epoch bump and the model
  swap. A load landing in the window returned scores computed from one model's
  token ids under another model's weights, while the identical call one moment
  later refused.
- **The panel filed the model's own output under "chat template scaffold"** —
  on gpt2, two lines below its own note saying gpt2 has no chat template.
  Attributing at a generated token there put 11 of 15 rows, including the
  highest-scoring token in the run, under that heading. `n_prompt` now reaches
  the client and rows past the prompt are their own group.
- **`typed_span=None` silently meant "all of it is yours".** `runtime._user_span`
  returns `None` for *"we could not locate your words"* — a slow tokenizer, or a
  prompt that occurs twice in the templated text — and rows came back labelled
  `typed`. They are now `unknown`, and the panel shows one list with no heading
  claiming authorship.
- **A truncated run told you your words "were not candidates".** They were
  candidates; the 64-token cap simply never reached them. Measured on gpt2 at
  position 100 of a 125-token generation: typed span [0,5], all five
  candidates, window starting at 36, zero tested. `tested_span` is now
  returned, the sentence says "not asked", and the untested in-cone chips carry
  their own mark on the strip instead of rendering blank beside scored ones.
- **Nothing scrolled the attributed token into view.** On a 96-token generation
  the ringed chip sat 4872px into a 965px window at `scrollLeft` 0, so the
  strip opened showing 19 chips of which the only one with a bar was the sink —
  under text pointing at it four times.
- **A test that asserted a race.** `test_sizes_come_from_the_registry` compared
  the order eight *concurrent* registry lookups happened in against the curated
  order. Measured over 60 runs, the order it asserted occurred 0 times; the
  test had been passing on scheduling luck. It now checks the two properties
  that hold: every name is resolved exactly once, and `pool.map` returns the
  rows in curated order — which is the order the user reads down.
- **Ten docs pages sharing one meta description.** Per-page `description` front
  matter was added, then five of the ten silently fell back to the site default
  because an unquoted YAML scalar containing `": "` is not a valid mapping.
  `mkdocs build --strict` reported nothing; the only way to see it was to read
  the generated HTML.
- `tomllib` is 3.11+, so the version-drift test could not run on the 3.10 CI
  cell at all. The dev group now backfills `tomli` there.

### Changed

- **Error text from libraries no longer reaches the browser.** ModelMRI now
  distinguishes a refusal it wrote from a failure underneath it, and the two
  get different answers. New `modelmri/errors.py`: `Refusal` (409, in its own
  words) and `BadRequest` (422, in its own words). Everything else is a 500
  carrying one sentence — "Something inside ModelMRI failed rather than
  refusing. The full error is in the terminal running `modelmri serve`." — and
  the traceback goes to the terminal instead of the response body.

  What that changes, concretely. Before, a torch failure such as `RuntimeError:
  CUDA out of memory. Tried to allocate 20.00 GiB (C:\Users\you\.cache\...)`
  came back as **409 Conflict with that text**, absolute paths included. A full
  GPU is not a conflict, and torch's message was not written for you. Measured
  on these routes, all of which answered 409 or 422 with the raw text and now
  answer 500 with the generic one: `/api/hub/models`, `/api/hub/signin`,
  `/api/ollama/pull`, `/api/sae/load`, `/api/lens`, `/api/vla/load`,
  `/api/vla/attention`, `/api/vla/analyse`, `/api/vla/frame`,
  `/api/vla/episodes`, `/api/traces/import`, `/api/attention`,
  `/api/session/open`, `/api/model/prompt`, `/api/custom/load`,
  `/api/custom/run`, and the `/ws/generate` socket.

  Deliberate refusals are unaffected and still arrive in full: "This is a
  recording, and a `.mri` does not carry one", "ollama unreachable at
  http://127.0.0.1:11434: Connection refused", "lerobot/pusht is not cached.
  Looked in: ...", and a broken adapter still reports its own
  `ModuleNotFoundError` at 422, because `custom.py` catches that at the call
  into your file and can say whose code raised.

- **Situations that now answer 500 rather than 409 or 422**, listed because the
  status is what a script sees:
    - Any exception from `hub.py`, `ollama.py`, `lens.py`, `saes.py`,
      `traces.py`, `vla.py` or `vla_data.py` that is *not* one of that module's
      own refusals. A transitional arm in `server.py` was answering all of them
      409/422-with-text, and by the time it was audited seven of the eight
      modules it named had been converted, so it was catching nothing but
      breakage.
    - A LeRobot dataset failing to open for a reason that is not "not cached" —
      pyarrow failing on a parquet file, av failing on a container. The
      handlers caught `FileNotFoundError` wholesale and published its path; the
      module's own "not cached" sentences are `Refusal`s now and still answer
      409. A missing `pyarrow` or `av` is still a 409, with an install line.
    - A frame that cannot be decoded (`/api/vla/analyse`). It was a 409 there
      and a 500 on `/api/vla/frame` — the same failure, two answers depending
      on which panel you clicked. Both are 500 now, which is the side
      `vla_data.py`'s own comment argues for.
    - A failure inside `custom.py`'s own path resolution, tensor allocation or
      hook installation on `/api/custom/load` and `/api/custom/run`. Those were
      echoed at 422 as though your file had caused them.
    - `attribute.py` detecting a fault in its own instrumentation. It was
      reported as a 409 "ModelMRI decided not to answer".

- **Situations that now answer differently in the other direction:**
    - `GET /api/attention/ablate?baseline=<unknown>` and
      `/api/attention/attribute?baseline=<unknown>` answer **422**, not 409.
      They are malformed parameters, and the layer-index check on the same
      endpoint has always answered 422.
    - `GET /api/session/export` with a non-finite attention map answers **409**
      with its explanation ("...the custom-model panel reports which layer
      first goes non-finite") instead of a generic 500. It was the one route
      serving `session.py` with no arm for `SessionError`, which is now a
      `BadRequest`.
    - Every route that had no error handling at all — 28 of 56, including
      `/api/ollama`, `/api/models/discovered`, `/api/vla/datasets`,
      `/api/custom/candidates`, `/api/paths` — now returns the same
      `{"error": ...}` JSON as the rest, rather than a bare `text/plain`
      "Internal Server Error".
    - A Hub that does not answer is a 409 ("Could not reach the HuggingFace
      Hub"), not a 500. It was caught only for a connection failure; a proxy
      that closes the connection, sends a malformed status line, or truncates
      the body raises something else entirely, and one bad response could empty
      the model picker's opening view.
    - Ollama dying mid-stream, or something between you and it rewriting the
      response, is a 409 naming the host. Only a failure to connect was
      recognised before.

- PyPI metadata: 4 classifiers to 15, 6 keywords to 14, and the package page
  now links to the docs, the changelog, the issue tracker and the demo instead
  of the repository twice.
- `ablate._distribution` and `ablate._kl` are now public as
  `ablate.distribution` and `ablate.kl_nats`, imported by `attribute.py` rather
  than copied. Two KLs in one package would drift into meaning two different
  things, and a head score and a token score are read on the same screen.
  `kl_nats` now documents what its 1e-12 floor on `q` costs: nothing at all on
  ordinary rows, and 0.001672 nats on gpt2's index 0, where masking collapses
  10483 of 50257 vocabulary entries below the floor.
- `tests/e2e_check.py` no longer claims to exercise "EVERY feature": measured,
  it called 22 of the 51 declared routes. The three attention interventions
  (`ablate`, `diff`, `attribute`) were added, taking it to 25, and the
  docstring now names what it still does not reach.
- `tests/ui_check.py` reports skipped sections as `NOT CHECKED` in its summary.
  CI starts a server with no model, so the head-ranking and `.mri` round-trip
  sections quietly did nothing — 14 checks, including all five over the
  head-ranking panel — and a green run read as if they had passed.

## [0.8.4] — 2026-08-10

**No functional change.** Identical code to 0.8.3; this release exists to
exercise the release path.

Every GitHub Action was upgraded to its current major in the preceding
commit, but `release.yml` only runs on a tag — so the one workflow that
builds the artifacts people install was the only one the upgrade had not been
tested against. Two of its steps changed behaviour underneath it
(`setup-node` v7 can now enable caching on its own; `setup-uv` v9 no longer
prunes its cache), and the honest way to find out whether the release still
builds a correct wheel is to cut one.

The published artifacts are the ones CI built and attached to the GitHub
release, not a local build — which is what makes this a test of the path
rather than a test of my laptop.

## [0.8.3] — 2026-08-10

### Changed

- **The robot panel takes any policy, not just SmolVLA.** Three values pinned
  it to one checkpoint: the tensor prefix
  (`model.vlm_with_expert.vlm.model.vision_model.`), the repo its vision
  config came from, and the module class. `load()` accepted a `repo`
  argument, so the plumbing looked general — but any other policy found zero
  tensors under that prefix and was told its layout was "not supported",
  which was true only because nothing had looked.

  All three now come from the checkpoint. The prefix is **discovered** by
  scanning tensor names for a vision-shaped path segment (`vision_model`,
  `vision_tower`, `vision_encoder`, `image_encoder`, `visual`), with the
  busiest candidate winning so one stray `visual_proj` cannot outvote a real
  tower. The config is read from the checkpoint's own `vision_config`, or
  from the VLM it names — SmolVLA's `config.json` carries
  `vlm_model_name`, which is where the constant came from. The module is
  built by `AutoModel.from_config`.

  Checked against the real weights before being trusted: discovery returns
  exactly the string that was hardcoded (197 tensors), and
  `AutoModel.from_config` produces the identical 197-parameter module.
  Loading SmolVLA is unchanged — perception mode, 12 × 12, 32 × 32 grid.

  A checkpoint with no recognisable tower is refused with the top-level
  names it *does* have — a report you can act on rather than a verdict. The
  panel has a **policy** box beside the dataset picker, so this is reachable
  rather than merely possible.

- **The agents panel says what it records.** It is a flight recorder for an
  external agent you instrumented with `modelmri-record` — usually calling a
  hosted API — and has nothing to do with the model loaded in the
  playground. Nothing on screen said so, which made "the calls that model
  made" the obvious and wrong reading.

  The bundled sample (`examples/record_demo.py`) has a deliberately failing
  step so a timeline has an error to render. It was marked with a small
  `demo` pill and nothing else, and **"Clear my runs" deliberately spared
  it** — so the one button that looked like it would remove it could not.
  There is now a sentence naming it and its file, a count of bundled samples,
  and a **Remove sample** button.

## [0.8.2] — 2026-08-10

### Fixed

- **Fifteen bugs an adversarial hunt found before this shipped.** Six
  independent finders over everything changed since 0.8.1 raised 31
  candidates; 16 went through refutation and 15 survived, three of them
  blockers. Most were in code written in the preceding two days and already
  called verified.

  * **The demo's refusal of an unrecorded prompt rendered as the word
    `undefined`.** The DEMO branch of `streamGenerate` destructured
    `{generation}` off a 422 body without checking the status, streamed
    `String(undefined)` into the output panel as the model's answer, and then
    refreshed every panel below as though a real generation had happened —
    exactly what the refusal existed to prevent.
  * **Features, SAE and steering were not scenario-aware**, so selecting
    Qwen3-0.6B showed a 768-dim GPT-2 SAE against a 1024-dim model, gpt2's
    tokens under Qwen3's output, and an A/B pairing one model's baseline
    against the other's steered text.
  * **`modelmri uninstall` reported partial deletion as total failure and
    exited 0.** `rmtree` stops at the first entry it cannot remove, having
    already deleted everything before it.

  Also: the `--models` flag did nothing when the size probe returned 0; the
  cache size shown before deleting counted symlinked blobs two or three
  times; `cache_dir()` nests inside `data_dir()` on Windows and the dedupe
  could not see it; Ollama's `fits` verdict was a constant; every Ollama
  model was reported instruction-tuned, including its base tags; the
  base-model caveat rendered inverted on the demo and under failed
  generations; "what changes?" refused the mean baseline and whole-model
  sweeps; and the robot panel served the nearest baked frame and layer under
  controls naming the ones you asked for.

### Added

- **`n_prompt` in the `.mri` format.** Additive and bounds-checked on parse —
  a value outside the file's own token list is discarded rather than
  believed, and 0 means unknown rather than "all prompt". Files written
  before it still open. Without it a shared analysis opened on the empty
  canvas the resting state was added to remove.
- **`instruct` is tri-state.** `None` means unknown, which is not `False` —
  `False` is the positive claim "this is a base model", which the UI states
  in those words. Read from Ollama's own `/api/show` template for Ollama
  models rather than assumed.

- **The hosted demo runs the whole tool, and CI keeps it that way.** It is the
  only ModelMRI most visitors will ever touch, and it was the least verified
  surface in the repo — the `.mri` viewer beside it is gated cell-for-cell,
  while the demo's entire gate was `test -f demo-dist/index.html`.

  * **The head selector works.** It read `layer` and never `head`, then fell
    back to the first baked slice, so **141 of 144 selections drew a different
    head's arcs than the dial said** — silently. All 144 slices are baked now,
    keyed on the pair, and a genuine miss returns 422 with the reason in the
    same words the viewer uses.
  * **Rank heads works** — the capability the README leads with had no handler
    and answered 409 under advice that could not work. Every layer under both
    baselines, both whole-model sweeps, and the 60 comparisons the ranked rows
    can ask for are baked from a real run.
  * **Twelve other dead endpoints** now answer: the accelerator badge, storage
    panel, logit lens, HF tab, session state and progress among them.
  * **An unrecorded prompt is refused** instead of answered. Typing "what is
    2+2" used to return a confident sentence about the Eiffel Tower, with
    attention over the Eiffel Tower's tokens underneath the words you typed.
  * **"Share this view" works**, serving a real `.mri` of the demo's own run —
    so the demo → viewer hop is something a visitor can do rather than read
    about.
  * The bundle records **what produced it**: model, revision, dtype, device
    and prompt.

- **`tests/demo_check.py`**, wired into the Pages workflow. It extracts every
  `/api/...` literal `api.ts` can call, diffs it against what `demo.ts`
  answers, and fails on any gap — so the next dead endpoint fails a build
  rather than a visitor.

## [0.8.1] — 2026-08-10

### Fixed

- **Eight more shipped numbers that nobody had measured.** A sweep of every
  numeric claim in the repo — README, docs, docstrings, comments, tests — put
  45 candidates through adversarial verification; 8 survived it.

  * `tests/test_ablate.py` still carried the **retracted `+21.96 / +18.06 /
    ~6x`** verbatim, and a bf16 noise floor "around 5e-3" that measures
    exactly 0.0. The test explaining why we rank by KL was doing so with the
    figure that was wrong.
  * `server.py` still shipped `0.12-0.68 s per layer against 1.4-19.6 s` —
    the stale timings corrected elsewhere, and this copy is served publicly
    in the OpenAPI schema at `/docs`.
  * **README's ranking block did not reproduce.** It showed `L0 H7 KL 0.866,
    p(" the") 0.112 → 0.073`; measured values are 0.784 / 0.085 → 0.062
    (fp32) and 0.898 / 0.098 → 0.057 (bf16). The block now names its model,
    prompt, baseline and dtype, because the same three heads score 0.784,
    0.898 and 0.825 under three different setups — a KL without them cannot
    be checked by anyone.
  * The recorder wheel's **`7 KiB`** in three files (README said 9 KiB)
    against a real 8.9 KiB. Now guarded by a test that checks all four sites
    against each other and against the built wheel.
  * "attention rows summing to **1.000**" for six models: the recorded
    figures are 1.000–1.002, and two of the six had no recorded run at all
    (Llama-3.2-1B-Instruct is gated; OLMo-2-1B's download stalls). The
    "Verified, not asserted" table now contains only what was.
  * "the reader is about **200 lines**" for `vla_data.py` — 286 non-blank,
    and 256 on the day the sentence shipped. Replaced with the property that
    stays true: it imports no `lerobot` code.
  * "public SAEs exist for about **a dozen** models", in six places, sourced
    from nothing. The registry knows four repositories, so it says four.

### Added

- **The head ranking says what it will cost before it runs.** The button
  carries the forward-pass count — the portable part, 146 for gpt2 and 450
  for Qwen3-0.6B — and once one layer has been ranked, an estimate measured
  on *your* machine. **all N layers** appears only at that point, because
  before it there is no measurement to quote and the difference between
  seconds and minutes is not something a user can guess.

  The estimate uses the **fastest** rate seen, not the latest, because the
  first ranking after a load pays for CUDA warm-up and runs several times
  slower — and the button appears exactly after that first ranking, so the
  latest-rate version quoted its worst possible number.
- **The mean baseline is selectable.** The panel already told users the
  ranking depends on what a removed head is replaced with; now they can see
  it. On gpt2 with a 261-token generation, zero-ablation ranks L0H7 first at
  KL 0.825 while mean-ablation drops it to fifth at 0.070 and removes L0H10
  from the top five.

### Fixed

- **A redundant second copy of the weights is no longer downloaded.** The
  prefetch skipped TensorFlow, Flax, ONNX and TFLite formats but not Rust
  (`*.ot`) or a `pytorch_model.bin` sitting beside safetensors. Measured on
  gpt2: **1.7 GB pulled where 523 MB was needed.** The `.bin` is dropped only
  when a root-level `.safetensors` exists to load instead, so pre-safetensors
  models still get their only weight file and an adapter's safetensors in a
  subfolder does not condemn the real weights. With no repo listing, nothing
  is skipped — a guess there fails the load with the weights deliberately
  absent.
- **"No download needed" can no longer be said while downloading.** That
  verdict was decided once from the cache tree's size at t=0, and the tree was
  large because it held the redundant `.bin` above. Real case: gpt2 measured
  1045 MB against an expected 551 MB, so the load announced it had everything
  and then downloaded for 275 seconds under that message, with the byte
  counter climbing to 149%. New bytes arriving now retracts the verdict.
- **A whole-model ranking opens the head it just named.** It kept the current
  layer's best instead of the global winner, so the list said L0 H7 while the
  arcs showed something else. A ranking also survives the layer change it
  causes, rather than clearing the result the moment it arrives.

### Changed

- **Corrected every measured figure in the head-ranking documentation.** The
  logit-difference, baseline-ordering, additivity and noise-floor numbers in
  `ablate.py`, the CHANGELOG and the README came from a design review rather
  than a measurement, and none of them reproduced. The timings existed in
  three mutually inconsistent versions (README 1.8 s, `runtime.py` 1.4 s,
  actual 10.28 s). All are now measured, and `ablate.py` names the prompt it
  measured with — a KL without one cannot be checked by anybody, which is how
  the wrong ones survived. See LOG.md for the full before/after table.

## [0.8.0] — 2026-08-09

### Added

- **Compare two runs.** Rank the heads in a layer, then ask **what changes?**
  — and the panel shows the attention map with that head removed, subtracted
  from the one without. Arcs run both ways: one colour where the model
  attends *more* without the head, another where it attends *less*.

  The comparison is two forward passes over **one token sequence**, never two
  generations. That is the whole design. A cell-by-cell difference means
  something exactly when index *i* is the same token on both sides, and two
  generations do not guarantee that: sampling diverges above temperature
  zero, and chat templates insert a different number of leading tokens per
  model (measured: 0 for gpt2, 8 for Qwen3-0.6B, 29 for Qwen2.5-0.5B-Instruct).
  Subtracting misaligned sequences produces a smooth, plausible, entirely
  fictitious picture.

- **It tells you when zero is the only possible answer.** Ablating a head
  removes its *output*, so the layer it lives in is computed from an
  unchanged input and its attention is bit-identical — every time, by
  construction. The first build of this shipped a button that compared a
  layer against an ablation in that same layer and could therefore only ever
  show nothing; it now opens at layer L+1, the first layer that can differ,
  and an all-zero result explains itself rather than rendering a blank
  canvas.

  Measured on gpt2 removing L0 H7: 0.000 at layer 0, then 0.086 at layer 1,
  0.493 at layer 3, 0.113 at layer 11 — the change propagates rather than
  landing in one place.

### Changed

- Steering now installs through one shared function used by both generation
  and attention capture. Two implementations of "what steering does" would
  drift, and the comparison would then be between a real run and an
  approximation of one.
- The attention cache is keyed by intervention, so two runs can be held at
  once. All variants are dropped together — a stale baseline beside a fresh
  intervention would render a difference between two different generations.

## [0.7.0] — 2026-08-09

### Added

- **Rank attention heads by ablation.** The panel offered 144 heat maps and
  no reason to open any of them. **Rank heads** zeroes each head in a layer
  in turn, runs the model again, and measures how far the next-token
  distribution moves — so the dropdown becomes ordered, the top head is
  selected for you, and browsing becomes asking.

  The cost is **14 forward passes for one layer of gpt2 and 146 for all
  144 heads**; Qwen3-0.6B's 28 × 16 is 450. One layer is a click; the whole
  model is a wait, and the panel says which it is about to be.

  *(Corrected twice. This entry originally read 0.2 s and 1.8 s, with
  `runtime.py` carrying a third figure. It then read 1.0 s / 10.3 s / 137 s
  — measured, but not reproducible: the same model ranged 12–71 ms/pass
  across sessions on the same GPU, so seconds from one machine were never
  going to transfer. It now states the pass count, which does. See
  Unreleased.)*

  Four things make the number mean what it says, and each was a way to ship
  a confident wrong answer:

  * **The cut goes before the output projection**, where heads are still
    separable — after `o_proj` they are summed and cannot be pulled apart.
    Verified by construction: zeroing all heads one slice at a time gives
    bit-identical logits to zeroing the whole tensor.
  * **`head_dim` is read from the projection, not computed as
    `hidden_size // n_heads`.** That quotient is right for gpt2 and wrong by
    2× on Qwen3-0.6B (128, not 64) and wrong on gemma-3-270m (256, not 160),
    where it would ablate half of one head plus half of the next and rank
    them confidently. Mismatches are refused rather than guessed.
  * **KL divergence, not a logit difference.** Softmax is shift-invariant and
    ablation moves whole logit vectors: on gpt2 L0H0 with "The capital of
    France is", the top token's logit moves −0.258 while the vocabulary mean
    moves −0.145, so the honest residual is −0.113 and a raw logit difference
    would call that head 2.3× more important than it is.
  * **A measured noise floor**, from running the same forward pass twice with
    nothing ablated. It measures exactly 0.0 on this path — CPU and CUDA,
    fp32, bf16 and fp16 — and one pass per ranking is what establishes that
    rather than assuming it. Batching, TF32 or another accelerator can lift
    it above the smallest real signals, and nothing else would notice.

  Two things it deliberately does not claim. These are **not** each head's
  share of the prediction — measured on gpt2 layer 0, the twelve per-head
  scores sum to 1.995 while ablating the whole layer gives 0.208, and on
  gemma-3-270m-it layer 0 four per-head scores sum to 0.0007 against 6.57 for
  the whole layer — and the order depends on the baseline: zero-ablation
  ranks heads 7, 10, 9 there, while replacing each head with its own mean
  ranks 3, 1, 10. Both baselines are offered, the one used is named in the
  response and on screen, and the panel says plainly that the scores do not
  add up.

  *(Corrected 2026-08-09 — the logit, noise-floor and mean-baseline figures
  in this entry were wrong as first published. See Unreleased.)*

## [0.6.3] — 2026-08-09

Everything here came from a hostile audit of the code 0.6.0–0.6.2 shipped
earlier the same day. Nineteen findings were confirmed against the running
code and four were refuted; these are the ones that mattered.

### Security

- **The viewer's `?f=` guard was bypassable, and the bypass was real.** The
  filter tried to spot absolute URLs by pattern — reject `scheme:` or a
  leading `//`. A backslash walked through both: `?f=\/evil.example/x`
  resolves protocol-relative, and I reproduced the viewer issuing a live
  cross-origin GET to `http://evil.example/x`. A link was enough, and the
  page is publicly hosted. It now resolves the URL and compares the origin,
  which is the only rule that cannot be spelled around, and rejects
  backslashes and control characters outright. Ten hostile values are fired
  at a real browser on every CI run.

- **`_clean_partials` could delete outside the cache.** It built a directory
  name by replacing `/` in an id that arrives in an HTTP body, leaving
  backslashes and `..` intact — and that function deletes files. The id is
  now reduced to characters that can only name a directory inside the cache.

- **The local viewer server answered to any `Host`**, so a page on any site
  could point a name it controls at `127.0.0.1` and read the recording
  (DNS rebinding). It now answers only to loopback names, and `--host`
  prints a warning when it is not loopback.

### Fixed

- **A `.mri` had no bounds at all**, and it is a format designed to be
  forwarded. A 3 MB gzip bomb allocated 3 GB; a 31 KB file claiming 20,000
  tokens asked for 400 million floats per map — in the recipient's browser
  too. Decompression is now incremental and bounded, and the cell count,
  file size, and layer/head counts are all checked before anything is built.
- **A malformed `.mri` wedged the server.** `open_session` installed the
  parsed object before validating it, so one bad file made every attention
  request 500 for the rest of the process. Validation now happens in
  `parse`, which cannot leave broken state behind.
- **NaN attention exported as a plausible blank heat map.** NaN loses every
  comparison, so the peak became NaN, the scale became NaN, and every cell
  quantised to zero — a smooth, believable picture of nothing, which is the
  precise failure this project exists to prevent. It is refused now, on both
  the tensor and the pure-Python path.
- **"Download it anyway" applied to the wrong model.** The refusal carried
  only its message, so the override used whatever was selected at the moment
  of the click — and picking a different model left a refusal on screen
  asserting something no longer true. It carries its model now, names it on
  the button, and a new pick retires it.
- The viewer server was single-threaded despite setting `daemon_threads`
  (which does nothing on `TCPServer`), so one held keep-alive socket stalled
  every later request; `HEAD` returned 404 where `GET` returned 200; and a
  client disconnecting mid-response printed a traceback at someone who only
  wanted to look at a file.

## [0.6.2] — 2026-08-09

### Changed

- **`modelmri open` is 34× faster: 8.9 s → 0.26 s warm, and about 26 s → 0.3 s
  cold.** It was starting the whole application — torch, transformers,
  FastAPI, uvicorn — to display a 54 KB recording that needs none of them.
  0.6.1 made that wait honest; this removes it.

  The viewer bundle now ships inside the package, and `modelmri open` serves
  it with `http.server` from the standard library, on the loopback interface
  only, exposing nothing but the viewer's own directory and the one file you
  named. The page receives it through a `?f=` link, so the analysis is on
  screen when the tab opens rather than waiting to be dropped.

  The split is now clean: `modelmri serve` is the tool, `modelmri open` is a
  file reader. A test asserts the reader imports none of torch,
  transformers, fastapi, uvicorn or numpy — verified to fail the moment one
  convenient top-level import is added back.

  It is the same bundle published to `/viewer/`, copied from one build, so
  what you read locally cannot drift from what you sent someone a link to.

## [0.6.1] — 2026-08-09

### Added

- **A zero-install browser viewer for `.mri` files.** `modelmri open` works,
  but it imports torch and transformers first — **measured at 26 seconds** —
  to display a 54 KB recording that needs neither, and the first person to
  try it pressed ctrl-c through the wait. The viewer is the same React app
  built with the API answered from a file you drop, so a recipient reads a
  shared analysis with nothing installed and nothing uploaded. Published at
  `/viewer/` on GitHub Pages.

  The browser and Python implementations of the format are checked
  cell-for-cell on the same file by [tests/viewer_check.py](tests/viewer_check.py)
  — on the current fixture, 6,912 cells with identical checksums. A viewer
  that renders a *slightly* different matrix would be worse than none,
  because nothing on screen would say so.

- **Pull any Ollama model by name.** Ollama publishes no search API, so the
  tab offers a name box rather than a result list — which reaches strictly
  more: any tag, any namespace, anything published since. It resolves
  against the registry first, so the size is on screen before the button,
  and a model that cannot fit says "Won't fit" instead of offering a click
  the server would refuse.

### Fixed

- **`modelmri open` looked like it had hung.** It printed "serving on …" and
  *then* spent 26 seconds importing torch. It now says what it is doing
  before the wait, and ctrl-c prints `stopped.` instead of a thirty-line
  traceback through uvicorn and transformers.
- **The HuggingFace tab took 3.4 seconds to show anything.** The curated list
  fetched eight repos one at a time; it fetches them concurrently, so the
  view the picker opens on takes 1.6 seconds.
- **`npm run build:demo` did not exist**, and the `VITE_DEMO=1 npm run build`
  form it replaced is not valid syntax in PowerShell or cmd — so the demo
  target only ever built on a POSIX shell. All three targets are now vite
  modes (`--mode demo|viewer`), selected the same way on every platform.

## [0.6.0] — 2026-08-09

### Added

- **Shared sessions (`.mri`).** An analysis you can send to someone who has
  no GPU. "Share this view" in the attention panel writes one file holding
  the tokens, the attention, the generation and a note; opening it drives
  every panel through the same calls a live model does, so a laptop with
  nothing installed reads it identically. A 29-token gpt2 run with all 144
  attention maps is **54 KB** — attention is quantised to uint8 against each
  matrix's own maximum (worst measured error 0.002, and the strongest
  attention in every row is preserved) and gzipped. The file states its own
  precision, because a number that has quietly lost some is the thing this
  project exists to catch. Drag one anywhere onto the page to open it.

  Loading a model or generating your own run closes an open session: reading
  your output above someone else's heat map is a discrepancy nothing on
  screen could explain.

- **`modelmri open somebody.mri`.** One command: it validates the file,
  starts the server with the analysis already loaded, and opens a tab. The
  recipient does not have to know what a server is. A file that is not a
  session is refused with one sentence and exit code 2, before anything
  starts.

- **A download you can stop, and one that is refused before it starts.**
  Clicking `zai-org/GLM-5.2` in the picker began fetching **1506.7 GB** onto
  a laptop with an 8.6 GB GPU and 88 GB free. Nothing warned, and the only
  way to stop it was to kill the server. Now:

  * The picker shows the download size on every HuggingFace row — it was
    querying the Hub with `full=true`, which does not return `safetensors`,
    so every row came back with no size at all. Sizes are computed from the
    repo's own per-dtype parameter counts, so a 753B-parameter model reads
    1.5 TB in BF16 and 756 GB in FP8 rather than one number for both.
  * A shared guard (`modelmri/capacity.py`) refuses what cannot fit —
    checked against the actual free space on the volume that download would
    land on. Disk refusals cannot be overridden; "too big for your GPU" can,
    with a second deliberate click. Enforced on the server, because a check
    the browser performs is a check the browser can skip.
  * **Ollama pulls go through the same guard**, sized from the registry
    manifest and checked against Ollama's own models directory rather than
    the HuggingFace cache — `deepseek-r1:671b` is 404 GB and had no check at
    all. One rule, so the two cannot drift.
  * A **Stop** button that works. `from_pretrained` downloads inside the
    calling thread and Python cannot interrupt a thread blocked in a socket
    read, so the fetch now happens in a child process that can be
    terminated; partial blobs are removed and the freed bytes reported.

### Fixed

- **A load could hang forever with the weights already downloaded.** The
  prefetch child was spawned with `stderr=PIPE` and nothing draining it;
  `huggingface_hub` writes progress to stderr, so once the ~64 KB pipe
  buffer filled the child blocked and never exited. The UI sat at
  "551 MB / 551 MB · 234s · reading from local cache" indefinitely. Both
  streams go to `DEVNULL` now, and two tests hold it there.

- **`MODELMRI_HOME` did not relocate everything it promised.** A surviving
  `~/.modelmri/traces.sqlite` from 0.5.1 or earlier still won, so the same
  command produced different storage on two machines depending on their
  upgrade history. An explicit instruction now beats the compatibility
  fallback. Found by a new test that runs the whole app inside a synthetic
  home and fails if any absolute path in any API response points outside it.

- **The local model list showed models that were not there.** Asking the Hub
  what a repo *is* downloads its `config.json`, and a refused or abandoned
  load leaves that behind — so a repo appeared under "On this machine" at
  0.00 GB, and clicking it would have restarted the whole download. A cache
  directory now has to contain weights to be listed as cached.

Seventeen portability and path bugs, from an audit of code that was
already tested and shipped. They share one shape — a location computed
correctly in one module and approximately in another, so the tool downloads
to a directory it does not search.

- **`import modelmri` crashed where no home directory resolves.** `LEGACY =
  Path.home() / ".modelmri"` ran at import, and `Path.home()` raises rather
  than degrading. That killed `modelmri serve` and even `modelmri where` on
  a container running as a UID with no passwd entry — and it fired *before*
  `MODELMRI_HOME`, the documented fix for that exact situation, could be read.
- **The HuggingFace token was created world-readable, then narrowed.**
  `write_text` makes the file at 0644 under a typical umask; the `chmod` came
  after. On a shared host that window is enough. It is now opened at 0600 and
  moved into place atomically, so an interrupted write cannot leave a
  half-written credential either. SECURITY.md now also states plainly that
  the mode is POSIX-only, and both it and the docs stopped naming
  `~/.modelmri/hub.json`, which has not been the location since 0.6.
- **`HF_HUB_CACHE` was ignored in four places.** The robot panel reported a
  checkpoint missing while it sat in the real cache, and the fix it suggested
  re-downloaded into the directory it was not reading. The dataset picker
  listed datasets from three roots that the opener looked for in one, so it
  advertised datasets it then refused to open. The download meter watched a
  directory nothing was written to. `HUGGINGFACE_HUB_CACHE` — still honoured
  by `huggingface_hub` itself — was missing everywhere, and a blank-but-set
  variable resolved to the working directory.
- **`MODELMRI_MODELS_DIR=~/models` matched nothing.** Neither of the two
  places that parsed it expanded `~`, so it became the literal directory
  `<cwd>/~/models`: the scanner silently dropped it, and the adapter loader
  refused every file under it as outside the allowed roots. One resolver now.
- **`HF_HUB_CACHE=D:\hf` made the model scan walk the whole drive**, because
  the scan root was the cache's *parent*.
- **Undelivered traces were written to the working directory** — which, for a
  library imported by your agent, is normally your git repo. A trace holds
  full prompts and tool output.
- **`modelmri where` named directories nothing was using.** It now reports the
  actual trace database, token file and undelivered-trace directory, resolved
  by the same code the callers use, and survives having no home.
- **`OLLAMA_HOST` was never read**, so Ollama on another port or another
  machine reported as not running. Bare `host:port` is accepted.
- **`modelmri where` could die with a UnicodeEncodeError** on a Windows
  console whose code page cannot encode the user's own path — the command
  that answers "where is my stuff?" failing precisely for the people whose
  stuff is hardest to find.

## [0.5.1] — 2026-08-08

### Fixed

Two wrong measurements, found by an adversarial audit hours after 0.5.0
shipped. Both are the exact failure this project exists to prevent: a
confident, plausible number that is wrong.

- **The logit lens read the model through the wrong transform.** HuggingFace
  decoders apply the final norm and *then* record the hidden state, so
  `lm_head(hidden_states[-1])` reproduces `logits` exactly. The lens applied
  the norm again — `head(norm(norm(h)))` — and a norm with learned gamma/beta
  is not idempotent. On gpt2 completing "…located in the city of", the top row
  read `' the'` while the model actually said `' Paris'`. That row supplies
  `final`, which anchors `settled_at` and the whole agreement column, so one
  wrong row mislabelled the table. Now detected at runtime rather than
  assumed, so it holds across transformers versions and model families.

- **Saturation was inverted.** The threshold measured distance from the
  tensor's own maximum instead of the activation's real bounds, and tested
  magnitude only. 9,000 sigmoid units pinned at 0 — gradient ~0, textbook
  saturation — reported **10%**, because the 1,000 healthy units at 0.5 were
  the ones counted. A maximum-entropy uniform softmax reported **100%**.
  Bounds are now written down per activation, both rails count, and Softmax
  and LogSigmoid get no figure at all: per-element saturation over a
  distribution is not a meaningful quantity.

## [0.5.0] — 2026-08-08

### Added

- **Custom models.** Point ModelMRI at a network you trained yourself — a
  `state_dict`, a TorchScript file, or a Python adapter exposing an
  `nn.Module` — and get a live map of every layer: shapes, activation
  statistics, dead and saturated units, and the flow of a real forward pass.
  See the [custom models guide](docs/guides/custom-models.md).
- **Adapters as a user extension point** — a `.py` file with
  `def load(): return your_model`, discovered by reading text rather than by
  importing, so a candidate that would crash on import is listed safely. See
  `examples/adapter_template.py`.
- Open-source project files: contribution guide, code of conduct, security
  policy, support guide, issue and pull-request templates, citation metadata,
  CODEOWNERS, and Dependabot.
- CodeQL scanning for Python and TypeScript.
- `tests/ui_check.py` and a CI job for it — assertions only a browser can make:
  that no expensive endpoint is called before you ask, that every panel starts
  inert, that no button renders as bare text, and that nothing overflows
  sideways at 1440, 768 or 375 px. The unstyled-button check injects the bug
  into the live page on every run and fails if it isn't detected, so it can't
  quietly become a no-op.

### Changed

- **Nothing loads until you ask for it.** The robot-policy panel used to open
  its dataset and decode a video frame the moment the page rendered — 396 MB of
  resident memory and about 4.4 seconds for a panel many people never open. It
  is now inert until clicked, and the server holds 9 MB after a page load
  instead of 405 MB.
- `GET /api/vla` now reports the configured `dataset_repo` and `policy_repo`,
  so the resting panel can name what a click will read instead of assuming the
  default.

### Fixed

- Secondary buttons rendered as bare text. Tailwind's preflight resets
  `<button>` to transparent and borderless, so one without a class is still
  clickable — which is why it passed every functional test and failed the only
  thing that mattered.
- The activation column could not shrink, pushing about 45 px of the layer
  table off the right edge of a phone.
- A long footer URL widened the whole page rather than wrapping.
- The model picker offered ModelMRI's own `saes.py` and `vla.py` as models you
  had trained: a substring search for `def load(` matches every `load` method
  in existence.
- **Every liquid-glass surface was fully transparent.** `--glass-fill:
  var(--glass-fill)` is a cycle, so it computed to nothing and took the
  `background` declaration with it. The model picker was blur with no frost,
  and the hero headline read straight through the model list. The scrim behind
  it — which carries the whole effect, because a parent with `backdrop-filter`
  becomes a backdrop root for its children — was `blur(3px)`; it is now
  `blur(28px) saturate(135%)`.
- **Keyboard focus was invisible on half the app.** `:focus-visible` and
  `.model-row` share specificity (0,1,0), so the eight `all: unset` rules below
  it won on source order, and `.theme-seg button` at (0,1,1) won outright. 19
  of 20 controls in the picker moved focus with no ring.
- **A page reload discarded your analysis.** The attention and feature panels
  were gated on a client-side counter, so refreshing unmounted them while the
  server still held the activations.
- **Errors reached the screen as JSON.** All 19 error paths showed
  `Error: 422: {"error":"…"}` around sentences written for humans.
- The model picker resized 266px under the cursor when its list arrived.
- The footer read `MRI-0.3` for the whole 0.4 line, and the hosted demo's
  "On this machine" tab said "Nothing found".
- Touch targets below 44×44 on a phone: 13 → 3.

## [0.4.0] — 2026-08-07

### Added

- **Robot policies (VLA).** Read a LeRobot v3.0 dataset frame by frame and
  paint the attention of `lerobot/smolvla_base`'s vision tower over the camera
  image, alongside the real robot state and action vectors.
- **HuggingFace sign-in and model browser.** Search the Hub, see which
  repositories your account can actually use, and open the acceptance page for
  the ones it can't — access is checked live rather than assumed from sign-in.
- **Ollama support** for models you already serve locally (text only; the
  internals never leave Ollama's process).
- **Local model discovery.** ModelMRI scans your working directory for
  HuggingFace caches, model folders and GGUF files instead of asking you to
  type a path, and says so honestly when the tree was too large to finish.
- **GPU autodetection** across NVIDIA, AMD, Intel and Apple silicon, with an
  accelerator badge that explains its own choice on hover.
- **Load progress** with real byte counts, and a stall warning after 45 seconds
  without new data — a dead Hub transfer doesn't raise, it just stops moving.
- A zero-install [hosted demo](https://muhammadmahadazher.github.io/ModelMRI/)
  running against baked fixtures.
- Design system v5: a real OKLCH colour system, dark mode, and motion that
  fires on new data rather than on every render.

### Fixed

- The access check answered "no" for every repository, including open ones,
  because the Hub's `auth-check` endpoint replies `200` with an empty body and
  the code tried to parse it as JSON.
- The picker offered gated models the loader then refused.
- The attention strip could grow but never shrink.
- The progress bar painted nothing: it referenced a CSS variable that does not
  exist, and an undefined `var()` invalidates the whole declaration. It had
  been "verified" by reading its width, which was correct.
- The page scrolled sideways to 7859 px, because grid items default to
  `min-width: auto`.
- The repository had not been installable from a clean clone since v0.1.0.

## [0.1.0] — 2026-07-08

### Added

- Model runtime with token streaming over WebSocket.
- Per-layer, per-head attention captured from a real forward pass with
  `attn_implementation="eager"` — SDPA and FlashAttention never materialise the
  weights.
- Interactive arc inspector and playground.
- Sparse-autoencoder feature extraction and activation steering.
- Agent Mode: `modelmri.record`, a trace store, and a timeline UI.

---

# modelmri-record

## [0.1.3] — 2026-08-08

### Added

- `trace(..., meta={...})` — anything you want stored alongside a run: a git
  sha, an environment, a ticket id. Recorder identity is merged last, so a
  caller cannot shadow it.

  `{"demo": True}` is the case that prompted this. ModelMRI ships a sample
  trace that deliberately fails a `git push` so its timeline has an error to
  render, and in the viewer that was indistinguishable from the user's own
  agent failing. Scripted sample data should never pass for a real recording.

### Note for `modelmri` users

`modelmri.record` is now a re-export of this package rather than a second copy
of it. The copy had fallen behind and was missing credential redaction — on
the import path `modelmri`'s README documents. If you import from
`modelmri.record`, upgrade `modelmri`.

## [0.1.2] — 2026-08-08

### Fixed

- Concurrent asyncio tasks shared one ancestry list, so parallel agents
  produced an interleaved and wrong tree. Parentage is now per-task.
- Repeated quick runs of the same agent could collide on one filename and
  overwrite each other.
- `step()` raised `TypeError` when used as a context manager outside a trace,
  and inside `ThreadPoolExecutor` workers — contextvars don't cross threads.
  It now returns a falsy no-op that still supports `with`.
- Unserialisable payloads (cyclic graphs, tuple keys) raised into the host
  application instead of degrading quietly.

## [0.1.1] — 2026-08-08

### Fixed

- **A private key could reach disk in the clear.** Redaction runs at delivery,
  but the recorder truncates long values at capture, which severed the
  `-----END` line that the PEM pattern required. Redaction now also matches a
  headless key block and a dangling `END`.

## [0.1.0] — 2026-08-08

### Added

- Standalone, dependency-free recorder split out of `modelmri`: `trace()`,
  `step()`, `instrument_anthropic()`, and a credential scrubber.

[Unreleased]: https://github.com/muhammadmahadazher/ModelMRI/compare/v0.5.0...HEAD
[0.5.0]: https://github.com/muhammadmahadazher/ModelMRI/compare/v0.4.0...v0.5.0
[0.4.0]: https://github.com/muhammadmahadazher/ModelMRI/compare/v0.1.0...v0.4.0
[0.1.0]: https://github.com/muhammadmahadazher/ModelMRI/releases/tag/v0.1.0
