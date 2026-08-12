# ModelMRI next release: where we stand, and what to build

## Where ModelMRI actually stands

**What it genuinely leads on.** Three things, and they are narrow.

First, the activation patching grid with per-site controls. Nothing else in the category runs eight same-norm random draws per patch site and a shifted-position control by default. `transformer_lens.patching` and nano-causal-interventions do the patching; neither runs the null. That is a real lead and it is currently the strongest thing in the repo.

Second, the SAE activation-convention calibration. `saes.py` tries four conventions, reports FVU, and refuses when none reconstructs. sae_vis and Neuronpedia will both happily render a dashboard for an SAE being fed the wrong convention, because they assume the convention from metadata. This lead is invisible to most users and worth explaining more loudly than it currently is.

Third, and most durably: ModelMRI holds the recorder and the weights in one process. LangSmith, Langfuse, Phoenix, Braintrust, Weave, Opik and Laminar all stop at the API boundary; Neuronpedia and circuit-tracer have no agent traces; Docent has transcripts and no model. Nobody else can join a failing agent step to the heads that moved the token. **That join is not built yet.** The two panels sit beside each other and do nothing for one another. This is the single largest gap between what the architecture permits and what ships.

**What it is behind on.** A long list, most of it not important. TransformerLens has direct logit attribution, head-type detectors, edge-level patching and a decade of research idioms. Neuronpedia has 36,000 precomputed dashboards. circuit-tracer has attribution graphs. Every runner in the local-LLM category — llama.cpp, Ollama, LM Studio, Jan, vLLM, TextGen — exposes an OpenAI-compatible `/v1` and ModelMRI cannot be dropped into any existing client. Every agent-tracing platform has full-text search over transcripts and a latency waterfall. None of this is close.

**What it is behind on that matters.** Four things.

1. **No probing, no steering without an SAE, no baseline except zero and mean.** `ablate.BASELINES` is `("zero", "mean")` and the module's own docstring records that the two disagree on gpt2 layer 0 and does nothing about it. The field's literature says both are off-distribution. Shipping a ranking whose baseline choice silently changes the answer, while the docstring admits it, is the largest correctness debt in the repo. Steering dies entirely for any model without a published SAE — which is almost every model a laptop user runs.
2. **No cost preflight.** circuit-tracer's tracker has an open request for exactly this (#109), an OOM on an 80 GB H100 (#92) and an OOM from defaulting to GPU0 (#40). Memory is the loudest complaint in the category and no tool prices the run before you pay for it. ModelMRI already has the pattern in `capacity.py` and uses it only for downloads.
3. **GGUF is a dead end.** `discover.LOOSE_WEIGHTS[".gguf"]` is a flat refusal note. The scanner finds the file format the majority of local users actually run and then says it cannot open it. Zero of eleven runners expose attention inside a GGUF.
4. **Every number is measured once and presented once.** "A number measured once is a sample, not a property" is currently a README line, not a behaviour. Nothing aggregates over prompts, nothing compares against a null model, nothing re-runs.

The robotics half is a special case: it paints attention, which by the project's own standard is the weak, correlational version — and the field has already published that interventional masking beats attention weights on explanation fidelity. The action expert is unimplemented (`vla.py:301` refuses with "needs the optional lerobot extra"), which blocks about half the robot features anyone would want. Whether that gate is worth an XL release line is an open question, not a settled one.

---

## The list

### Theme 1 — Correctness of what already ships

These fix or qualify measurements the tool already makes. Highest ratio of correctness to effort in the document.

**1. Analysis cost preflight**
Before a patching grid, head sweep, feature sweep or corpus pass, run one probe forward pass over the real sequence, measure its peak accelerator memory and wall time on this machine right now, multiply by the pass count the analysis already knows it needs, and show "132 passes, about 3.0 s, peak 1.4 GB of your 8.6 GB free — run it?" Above a stated fraction of free memory, refuse with both numbers named.
*Ahead of:* circuit-tracer #109 asks for this and it is unbuilt; nothing in the category prices a run first.
*How:* New `modelmri/budget.py`, called by `POST /api/patch`, `GET /api/attention/ablate`, `/api/attention/attribute`, `/api/features/ablate`. `torch.cuda.reset_peak_memory_stats()` / `max_memory_allocated()` plus the ROCm/XPU/MPS equivalents `modelmri/devices.py` already discriminates. Pass counts already exist — `patch.trace` returns `passes`, `ablate.rank_heads` returns `passes` and `elapsed_s`. Refusal reuses `capacity.TooBig(overridable=...)`.
*Effort:* S
*Caveat:* An extrapolation from one probe pass is a sample and must be labelled as one — fragmentation, a growing KV cache and a second process on the same GPU all break it. On CPU and MPS the memory reading is weaker than CUDA; those paths say what they could not measure rather than reporting a confident zero.

**2. Resample ablation, and the disagreement between the three baselines**
Add a third baseline beside zero and mean: replace a head's contribution with the same head's activation from a different real sequence, drawn 8 times, reporting median with min/max. Beside it, one line the panel does not have — a Spearman rank agreement across the three rankings, so the panel can say "zero and resample disagree on 6 of the top 10 heads here" instead of showing you whichever you happened to select.
*Ahead of:* Every tool in the set ships zero and mean while the literature says both are off-distribution. `ablate.py`'s own docstring documents the disagreement and does not act on it.
*How:* Add `"resample"` to `ablate.BASELINES` and a third arm to `ablate._cut`. Replacement activations come from a local corpus file or the prompts already in the SQLite store `traces.TraceStore` opens, captured at matched positions with the existing `out_projection` pre-hook. Spearman computed inside `ablate.rank_heads` and returned as a field for the existing baseline selector in `AttentionPanel.tsx`.
*Effort:* M
*Caveat:* When no corpus sequence is long enough to reach the ablated position, refuse with the reason — silently degrading to mean is exactly the `.get(name, 0.0)` bug class. The corpus is part of the measurement: its name and token count go in the response next to the score. Cost is draws × the current sweep, so gate behind the preflight.

**3. The random-weight control**
A toggle beside the head ranking, token attribution and logit lens re-runs the identical measurement on the same architecture initialised with random weights, and shows the two side by side. If your head ranking looks about the same on an untrained model, the panel says so in a sentence, with the rank correlation, the seed, and which null it used. No download — the control is built from the config alone.
*Ahead of:* A 2025 result showed automated interpretability metrics failing to distinguish trained transformers from random ones. No company selling interpretability has an incentive to ship a button that concludes "this measurement is uninformative on your model."
*How:* `AutoModelForCausalLM.from_config(AutoConfig.from_pretrained(...))` — config only, no weights fetched, works air-gapped — seeded with `torch.manual_seed`. Run through the exact same `ModelRuntime.ablate_heads` / `attribute_tokens` / lens code paths, not a parallel implementation, which is the only thing that makes the comparison mean anything. On 8 GB the two cannot coexist: load the control after `ModelRuntime.unload()`, cache scores keyed by (config hash, seed, prompt) in the SQLite file.
*Effort:* M
*Caveat:* A random-config model is not *the* null — a weight-shuffled control answers a different question — so name which null ran. Architectures whose config cannot be instantiated without downloading remote code are refused by name, never approximated with a nearby config.

**4. Held-out KL for the plain logit lens**
Every lens row gains one number: how far the plain logit lens's distribution at that layer is from the model's real final distribution, measured on a held-out split. Where the plain lens is unusable on this model family, the panel says so with the KL that proves it, instead of rendering a confident ranked list of tokens that describes nothing.
*Ahead of:* tuned-lens (608 stars) has been stale 12 months and is a library with no UI. TransformerLens, nnterp and ModelMRI itself all ship the plain lens with its documented silent-failure mode and no reliability number.
*How:* Extend `modelmri/lens.py`. Reuses `lens._final_norm(model)`, `model.get_output_embeddings()` and the measured `last_is_normed` check so the final row is not double-normed. One `output_hidden_states=True` pass per batch. No training, no corpus curation.
*Effort:* S
*Caveat:* The KL is measured on whatever text you gave it — corpus name and token count on screen, and no reliability verdict below a stated minimum held-out size.

**5. Tuned lens, trained locally, shown beside the plain one** *(reviewers split: 7 KEEP / 6 REVISE — the revision was "ship #4 first", which this sequences after)*
After #4, train a per-layer affine translator on the session's own generations, a local `.txt`/`.jsonl`, or the prompts in your trace store, and render both columns with the per-layer held-out KL gap. The tuned lens never silently replaces the plain one; both rows stay on screen.
*How:* New `modelmri/tuned_lens.py`, Belrose objective — minimise KL(final logits ‖ head(norm(A_L h_L + b_L))). At d_model 768 that is 590K params per layer: `torch.optim.Adam`, fp32, minutes on a 4060. Lenses save as safetensors under `paths.py`'s cache root keyed by (model_id, dtype, corpus hash, token count). `GET /api/lens` gains `kind=plain|tuned|both`.
*Effort:* L
*Caveat:* A lens trained on 200 sequences of your own text is a lens for that text. Fetching pretrained lenses from HF breaks offline-first, so it is an explicit opt-in action, never a default. A tuned lens can be trained to look confident everywhere.

**6. Direct logit attribution, sited as a secondary column** *(reviewers split: 8 KEEP / 6 REVISE — the revision, adopted here, is to site it inside the ablation panel and make the zero-contribution case a first-class cell state)*
For the token the model actually predicted, a signed bar chart of how many logits each head and MLP contributed. Underneath, the reconstruction residual — every component plus bias against the real logit — which is exactly what the frozen-LayerNorm assumption cost on this run. A head whose direct contribution is near zero renders as "direct-path attribution cannot see indirect effects," not as a zero.
*Ahead of:* TransformerLens makes DLA valid by folding LayerNorm into the weights, which changes the model you are studying and cannot be checked from the output. Here the model stays as loaded and the cost of the approximation is a number on screen.
*How:* New `modelmri/dla.py`. Head slices via `ablate.head_geometry(block, n_heads)` — reused deliberately, since it reads head_dim off the projection's input width and refuses when `n_heads * head_dim` does not match (the quotient is wrong by 2× on Qwen3-0.6B and wrong on gemma-3-270m-it). MLP output via `patch._capture_out`. Project through `lens._final_norm` with the norm scale frozen at the value a hook recorded from the real pass. Report contribution minus vocabulary mean, for the shift-invariance reason `ablate.py`'s docstring gives for using KL.
*Effort:* M
*Caveat:* Exact only under a frozen LayerNorm scale; on GPT-2 the residual is not zero, so displaying it is mandatory or the chart is a fabricated 100%.

**7. Head type labels, gated on a matched null**
The head list stops being 144 anonymous numbers. Random repeating-token sequences label heads that behave like induction, previous-token, duplicate-token or sink heads — and for every label, matched *non*-repeating sequences build that head's own null, so a head is labelled only when it clears the null by a stated margin. Everything else reads "no type detected."
*Ahead of:* TransformerLens ships these detectors returning a bare score with no null, so a 0.3 reads identically whether 0.3 is remarkable for this model or ordinary.
*How:* New `modelmri/head_types.py`. Repeating sequences run through the same `output_attentions=True` path `ModelRuntime._capture` (runtime.py:1093) already uses. Score attention mass at offset −L+1 (induction), −1 (previous-token), −L (duplicate-token) and index 0 (sink — `attribute.py` already documents a real 4.86-nat effect on gpt2). Labels render in `AttentionPanel.tsx` and travel into the `.mri` via `session.build`.
*Effort:* M
*Caveat:* These are behavioural labels measured on random repeated tokens, not claims about real text, and the label must never be carried into the ablation ranking as if it explained the KL. A byte-level tokenizer that cannot be sampled cleanly is a refusal, not a best effort.

---

### Theme 2 — Measurements that do not exist yet

**8. Contrastive steering vectors, gated on a shuffled-label null**
Paste contrastive prompt pairs (or point at a JSONL of `{positive, negative}`), derive a steering direction per layer, sweep layers to show where it has most effect, and apply it during generation with the deterministic A/B the SAE steering already has. Refit on shuffled labels K times and show that null beside the real effect — a direction that does not beat its own shuffled version is reported as not measured, not as a discovery. Today, every model with no published SAE is unsteerable.
*Ahead of:* repeng (753 stars) and steering-vectors (160) do CAA/RepE as libraries with no UI and no null: you get a vector, a coefficient, and no way to know whether the direction is real.
*How:* New `modelmri/steer_vectors.py`. Last-token residual per layer via the `register_forward_pre_hook` pattern `patch._capture` uses; CAA is mean(positive) − mean(negative), RepE is the first component from `torch.pca_lowrank` over paired differences. Layer sweep applies candidates through `ModelRuntime._steer_handle` (runtime.py:1274), extended to accept a raw direction instead of only an SAE decoder row, so `_attn_variants` and `attention_diff` keep working. Shuffled refits reuse `patch.py`'s `CONTROL_SEED`. Fit on half the pairs, score on the other half.
*Effort:* M
*Caveat:* With few pairs the direction is mostly noise, hence the null and a stated minimum pair count. Coefficients are not comparable across models or layers, so report applied scale relative to the measured residual norm at that layer, never a bare number. A steered generation is one sample: seed and decode settings print with every A/B.

**9. Vector store with provenance, built inside #8** *(reviewers split 6/6, both REVISE — both said the same thing: this is the persistence layer for steering, not a standalone panel)*
Any direction you derive — SAE feature, fitted probe, difference-of-means — saves as a named vector carrying how it was derived, from which model and revision, which layer, which dtype. Reload it next session, steer with it, A/B it. Loading onto a model with a different hidden size is refused by name, not silently reshaped.
*How:* JSON + safetensors under `paths.data_dir()`, same platform-dir discipline as `paths.trace_db_path()`. Dimension check copies `saes.py`'s rule (d_in must equal hidden_size, refuse loudly at load). No separate panel, no separate API surface — it lives under steering and probing.
*Effort:* S (as part of #8)
*Caveat:* Equal hidden size does not mean equal basis. A vector lifted from a base model onto a fine-tune produces plausible, possibly meaningless steering — record model id and revision and warn loudly rather than blocking, since cross-checkpoint transfer is a legitimate experiment when the user knows they are running it.

**10. Layer-sweep linear probes with a permutation null and a majority-class line**
Give labelled examples, get a probe at every layer and the standard "where does this information appear" curve. Behind it: the majority-class rate, and a permutation null band from K refits on shuffled labels. A probe reading 62% on an imbalanced set renders inside the null band rather than as a finding. The fitted direction exports into the steering sweep and the ablation harness.
*Ahead of:* elk and concept-erasure are research libraries with no UI; SAELens has a probe trainer buried in training code. The version worth building is not "we have probes" but "we show you when your curve is inside the null."
*How:* New `modelmri/probe.py`. Residual capture at a chosen position per layer with one `output_hidden_states=True` pass per example — the call `lens.logit_lens` already makes. Logistic fit in pure torch (runtime deps stay torch/transformers/fastapi). Permutation null reuses `CONTROL_SEED`. Direction exports in the same safetensors shape as #9.
*Effort:* M
*Caveat:* A probe finding information is not the model using it — the ablation follow-up is the only thing that upgrades the claim. Minimum examples per class and minimum test-set size are enforced in code, not documented. Adds a second column: the same held-out set scored by whichever SAE feature you believe encodes the concept, since that is the field's own two large negative SAE results as a button.

**11. Edge-level path patching, residual receivers only** *(reviewers split: 7 KEEP / 5 REVISE — the revision, adopted, drops q/k/v receivers from v1)*
Click a bright patching cell and ask what put it there. Patch one sender component's contribution into one later block's residual input, everything else corrupt, and get a ranked sender list with the same signed recovery score, the same eight same-norm control draws and the same shifted-position control the node grid uses. "Position 7 layer 12 matters" becomes "head 9.6 wrote it."
*Ahead of:* `transformer_lens.patching` and nano-causal-interventions have edge patching, both notebook-only, neither with per-edge controls.
*How:* Extend `modelmri/patch.py` with `path_trace()` and `POST /api/patch/path`. `_capture_out` caches the clean sender (head-sliced with `ablate.head_geometry`); the corrupt run adds `(clean_sender − corrupt_sender)` into the receiver's residual input via a forward pre-hook. Score is the same `(gap_of(out) − ld_corrupt) / gap` fraction as `run_patched`, so edge and node numbers are comparable. Seed edges only from sites the node grid flagged, and state the seeding rule in the response.
*Effort:* L (M with residual-only scope)
*Caveat:* v1 does not split q/k/v — freezing q/k/v across GQA, fused QKV and rotary embeddings in arbitrary HF architectures is the fiddliest thing in this document, and getting it subtly wrong produces confident, ordered, plausible-and-wrong numbers. The response names the scope it ran. Edge count is quadratic, so the seeding rule is reported, not hidden.

**12. Patchscopes, as its own labelled experiment** *(reviewers split: 7 KEEP / 5 REVISE — the revision, adopted, is to not render it as a lens column)*
Pick a hidden state at some layer and position, splice it into a second prompt designed to make the model describe what it is holding, and see what the model says. Beside every decode, two controls: the target prompt with its own original activation, and with a same-norm random vector — so a decode that reads the same regardless of what was patched is visibly the target prompt talking.
*Ahead of:* nnterp ships patchscopes as a library call with no controls and no UI; nothing else has it. The controls are the addition — the method's known failure is a target prompt that describes anything fluently.
*How:* Extends `modelmri/patch.py`, which already owns `_splice` and already clones rather than writing in place, so the source cache cannot be corrupted. `_capture` at (layer, position), splice into the target, `ModelRuntime.generate_stream` for the decode. Controls reuse `CONTROL_SEED` and the same-norm draw construction (`r / r.norm() * norm`). Default target prompt from Ghandeharioun et al., shown in an editable field.
*Effort:* M
*Caveat:* The target prompt is part of the result and must be visible and editable, never a hidden constant. Source layer L into target layer L′ is only meaningful where the streams are comparable. A decode is a generation and therefore a sample — never presented as what the state "means." Sited as its own surface, not as a lens column that reads as another measurement of the same thing.

**13. Feature evidence page from your own corpus**
An SAE feature stops being "feature 14203 fired." Point at a local text file, sweep it through the already-calibrated SAE, and a per-feature drawer shows top-activating spans in that corpus, an activation histogram, the firing rate, and — needing no corpus and exact — the tokens the feature promotes and suppresses in vocabulary space. With the causal ranking `feature_ablate.py` already produces, one feature has three independent readouts: what it fires on, what it pushes toward, what removing it does.
*Ahead of:* sae_vis and Neuronpedia have richer dashboards for models and SAEs someone else chose; Neuronpedia self-hosting is documented failing on 8 GB (#198, #219). This inherits `saes.py`'s calibration refusal, so unlike every dashboard in the set it cannot show features from an SAE fed the wrong convention.
*How:* New `modelmri/feature_corpus.py`. Stream a local `.txt`/`.jsonl` (never a download) through `SAEHandle` using the convention `saes.calibrate` selected. Top-k spans, fixed-bin histogram and firing rate in a new `feature_activation` table in the SQLite file `traces.py` opens. Logit-weight table is pure weight math and exact: `W_dec[f]` through `lens._final_norm` and `model.get_output_embeddings()`.
*Effort:* L (the logit-weight half is S and can ship first)
*Caveat:* These are top activations in *this* corpus — corpus name, token count and the fraction of features that never fired sit next to the dashboard, and a feature with zero activations in 200K tokens is "not seen here," never "dead." No natural-language labels.

**14. Grounding: did the answer come from the document or the weights?**
Attach a text file, Markdown or pasted passages, ask a question, and after the answer get each attached chunk scored by how much masking it out moves the answer's next-token distribution, beside how much attention the answer's tokens actually paid to it. Chunks the model looked at but did not depend on are called out — that combination is the signature of a plausible answer coming from the weights. When nothing clears the measured noise floor, the panel says so rather than picking a winner.
*Ahead of:* llama-cpp-python #1141 asked for exactly this to tell hallucination from genuine retrieval and the requester gave up on the skill barrier. Open WebUI, GPT4All and Jan all ship RAG and all show *which chunks were retrieved*, never whether the answer used them.
*How:* New `modelmri/ground.py`. Paragraph/heading chunking and chunk-to-token-span mapping via the tokeniser's offset mapping — no embeddings, no index, this is not a retrieval engine. Scoring reuses `ModelRuntime.attribute_tokens`, which already masks a token out of every later position's attention, feeds explicit `position_ids` on every pass, and runs the deliberately-wrong-ordering check. The noise floor is that method's existing floor pass, so "not grounded" is a measured statement.
*Effort:* L
*Caveat:* Masking a whole chunk is a bigger intervention than masking one token and effects are not additive — present as "removing this chunk moved the answer by X nats," never as percentage shares. One forward pass per chunk, so an explicit chunk cap with the projected pass count shown. Refuse an attachment exceeding the fit calculator's number rather than truncating it silently.

**15. Causal ablation for your own nn.Module**
The custom-model panel maps one forward pass today. Add the causal question: sweep every layer replacing its output with the batch mean over your sample inputs and report how far the output moves; occlude each input feature or image patch and report the same — with eight same-norm random draws per site, so a hot cell has to beat noise before it is called hot.
*Ahead of:* Every platform in this category is a fixed transformer catalogue. None will ever look at the CNN somebody trained last week. This extends the one structurally uncopyable surface and holds it to the same standard as the LLM panels.
*How:* Extend `modelmri/custom.py` (1147 lines, currently descriptive only). Control discipline copied verbatim from `patch.py` — `CONTROL_DRAWS = 8`, `CONTROL_SEED = 0`, same-norm Gaussian draws — because that file already measured what one draw costs (single-site spread of 2.654 against a real recovery of 0.427). Sample inputs from an optional `sample_inputs()` beside the adapter's `load()`, refused with a generated stub when absent.
*Effort:* M
*Caveat:* The output metric is task-dependent — KL is right for a classifier and wrong for a regressor — so it comes from the adapter's declared task and refuses when the task is unstated, rather than defaulting to something that still yields a plausible ordering. Batch-mean over one sample is meaningless; enforce a minimum count.

---

### Theme 3 — A number measured once is a sample

**16. `modelmri sweep` / `modelmri run`: one iteration engine** *(merged from two proposals; both reviewers on both entries said ship one loop, not two)*
Hand it a JSONL of prompts or a small job file and it re-runs the head ablation ranking, token attribution or feature ranking once per prompt, then shows every head as a distribution — median, IQR, n, and how often it landed in the top-k — instead of one number. A head that tops one prompt and sits at rank 40 on the other nineteen displays as exactly that. Runs headless over SSH with no browser; emits one `.mri` per prompt plus a JSONL of scalars with the full setup on every row.
*Ahead of:* This is HeadVis's Head Finder (corpus-averaged metrics across 37 models) computed on your model and your prompts. It also converts the project's house rule from README copy into shipped behaviour. Nothing else in the analysis surface is scriptable at all today.
*How:* New `modelmri/corpus.py` holding a `Sweep` that loops `ModelRuntime.ablate_heads` / `attribute_tokens` / `patch_trace` / `rank_features` and aggregates order statistics (never a mean without spread), plus a `Job` dataclass instantiating `ModelRuntime` directly — no FastAPI, no uvicorn. `sweep` / `run` subcommands in `modelmri/cli.py`. Progress and cancellation reuse `modelmri/progress.py`. Results persist in a `sweep` table at `paths.trace_db_path()`. JSONL rather than CSV so a refusal row carries its sentence in a `could_not_measure` field.
*Effort:* L
*Caveat:* Cost is N prompts × per-prompt cost, and the 8-draw control discipline multiplies through every row — the projected pass count prints before starting, and a sweep it cannot finish is refused rather than hung. Aggregating across prompts of different lengths is valid for head and feature metrics and invalid for position ones; that boundary is enforced in code. Refusals are recorded as rows, never skipped, or the output file quietly describes only the prompts that happened to work.

**17. Receipts on every number**
Each measurement gains a machine-readable receipt: model id and resolved HF revision sha, dtype, device, attention implementation, seed, tokenizer hash, prompt hash, ModelMRI version, and the exact request that produced it. `.mri` carries the receipts. Findings become auditable without anything re-running.
*Ahead of:* pyvene serializes interventions to the Hub; Neuronpedia and circuit-tracer share graphs. Nothing ships a result whose setup travels with each individual number.
*How:* A `Receipt` dataclass in new `modelmri/receipts.py`, stamped where numbers are produced — `ModelRuntime.ablate_heads`, `attribute_tokens`, `rank_features`, `patch_trace`, the lens path — each of which already reports its setup in prose, so this structures existing facts rather than inventing them. Revision sha from the snapshot directory `modelmri/hub.py` resolves. `session.build` gains a receipts list, validated in `parse` like every other untrusted section.
*Effort:* M
*Caveat:* `tests/test_no_machine_leaks.py` must be extended to cover every new field — receipts must carry no paths or usernames.

**18. `modelmri verify`** *(one reviewer scored the combined receipts+verify feature 8 KEEP; the other scored 7 REVISE asking to split it. Split adopted: #17 ships first, this follows)*
`modelmri verify run.mri` re-runs every receipt it can on the current machine and prints, per number: "reproduced to 1e-6", "differs: 0.41 vs 0.38", or "could not verify here — this GPU runs bfloat16, the file says float32". It never reports a pass it did not run.
*Ahead of:* No hosted platform can offer this — it can hand you its own assertion, never the re-run. The field just learned that automated interpretability metrics fail to distinguish trained transformers from random ones; being the tool whose numbers the recipient can re-run is the answer to that.
*How:* New `verify` subparser in `modelmri/cli.py` replaying each receipt through the same `ModelRuntime` methods the server calls. Tolerance derived from a measurement, not a constant: `ablate.rank_heads` already takes a noise-floor pass (the same forward twice) for exactly this purpose. The dtype sensitivity `patch.py` documents — bf16 changes the reference token itself, gap 4.000 vs 4.467 — becomes a first-class output class rather than a docstring warning.
*Effort:* L
*Caveat:* Bit-exact reproduction across machines is not achievable — kernels, cuDNN versions and TF32 all move numbers — so the report classifies rather than passes/fails, and every tolerance is measured before it is claimed. Cross-machine verification is the honest weak point and the CLI output says so rather than implying a clean pass means the finding is right. Ship only after the noise-floor pass has actually measured the per-metric envelope.

**19. `modelmri diff a.mri b.mri`, with a CI exit code**
Diff two saved analyses of the same prompt: which heads moved in the ablation ranking and by how much, where the logit-lens trajectory diverged, which patching sites changed sign, whether the generated text changed. `--fail-over 0.05` exits non-zero, so a repo can check in an `.mri` baseline and have CI say "your quantisation changed which heads carry this answer."
*Ahead of:* Nothing in the category has a regression concept for model internals. The state of the art for "did my quant change the model" is a Reddit thread.
*How:* New `modelmri/mri_diff.py`, `session.py::parse` on both sides, refusing when `tokens`, `n_prompt`, `n_layers` or `n_heads` disagree. The noise floor is not invented: `session.py::_quantise` stores attention as uint8 against each matrix's own max, so the per-cell floor is exactly that block's `scale`, and every delta is compared against it. `--json` matching `inspect`'s flag; a GitHub Actions snippet in `docs/guides/`.
*Effort:* M
*Caveat:* Two `.mri` at temperature > 0 differ for reasons that are not the model — detect the recorded sampling config and refuse to threshold on non-greedy runs. Version-gate per block and say which comparison is unavailable rather than treating a missing block as zero; that is precisely the 0.10 bug class. Refuse across dtype and device, or label each affected metric incomparable.

**20. Finetune-vs-base diff on a prompt set**
Pick two model ids — base and finetune, or two checkpoints of your own run — and a small prompt set. ModelMRI loads them one at a time (8 GB will not hold both), caches derived results, and shows the layer where the answers first diverge, which heads moved most in the ablation ranking, which prompt tokens the finetune newly depends on, and per-layer residual cosine at every position — with per-prompt spread, not one prompt's diff presented as a property of the finetune.
*Ahead of:* crosscode and OpenMOSS do model diffing by training a shared crosscoder — GPU-months, and both ship with no license file, which legally blocks reuse. The person who just finetuned a 0.5B on a laptop is a large audience served by nothing.
*How:* New `modelmri/model_diff.py`. Sequential load through `ModelRuntime.load` / `unload`, caching derived quantities rather than weights: `lens.logit_lens` trajectory, `ablate.rank_heads` ranking, `attribute` token ranking, per-layer residual cosine/norm via `patch._capture`'s pre-hook. Refuse when tokenizers produce different ids, when hidden sizes or layer counts differ, when vocab sizes differ (naming both), and never normalise different depths to a 0–1 depth fraction. Greedy decoding on both sides. New `diff` section in `session.build`.
*Effort:* L
*Caveat:* Extend `ModelRuntime`'s epoch discipline across the pair: a dtype, device or `attn_implementation` change between halves invalidates the diff rather than being diffed. This is a diff of behaviour on a prompt set — it cannot say a shared feature moved and must never be called model diffing in the crosscoder sense.

---

### Theme 4 — The agent↔model join

**21. Adopt a recorded LLM call into the mechanistic panels**
When a step in an agent trace was produced by a model on this machine, the step inspector grows one button: open this generation in the attention, logit-lens, ablation, patching and SAE panels. ModelMRI loads the model if needed, re-tokenises the recorded prompt, checks the ids against what the recorder captured, and refuses with a named mismatch if they differ. For a hosted-API step, the button is absent and a line says the weights are not on this machine.
*Ahead of:* This is the join no platform in the category can build, and the reason is structural: LangSmith, Langfuse, Phoenix, Braintrust, Weave, Opik and Laminar all stop at the API boundary and none ever holds the weights. It converts every existing mechanistic feature into an agent-debugging feature for the cost of one adopt path.
*How:* New recorder entry points `instrument_transformers()` and `instrument_ollama()` in `packages/modelmri-record/modelmri_record/__init__.py`, patching `transformers.GenerationMixin.generate` and the Ollama chat call to record `{model_id, input_ids, dtype, device, generation_config}` into a new nullable `meta` column on the `step` table. New `ModelRuntime.adopt_step(trace_id, step_id)` setting `last_ids`, `last_prompt`, `last_n_prompt_tokens`, bumping `epoch` and clearing `_attn_variants`/`_feats` — the same state every panel already reads (runtime.py:412-433) — with a tokenisation-equality refusal modelled on the one `patch.py` already raises.
*Effort:* L
*Caveat:* Only works for local models, a minority of agent traffic today — the absent-button case needs a sentence that explains rather than apologises. Re-tokenising can legitimately differ across a transformers or tokenizer upgrade; that is a refusal naming both versions, because adopting near-identical ids would point every downstream panel at a sequence the model never saw. **No substitute-model path**: replaying a Claude trace's prompt through a local 0.5B and labelling it a substitute is a machine for confident wrong conclusions even with loud labelling.

**22. Ship an agent failure as one `.mri`**
Export a bundle carrying the agent run and the mechanistic snapshot of the step that failed — timeline, tokens, attention, logit lens, any patching trace — in one gzipped file that opens in the browser viewer with nothing installed. The recipient sees the failing tool call, clicks it, and lands in the attention view of the generation that produced the bad argument, on a machine with no GPU.
*Ahead of:* Every competitor's share artefact is a link into their hosted trace UI, which dies when the account lapses. Helicone went into maintenance mode in March 2026, Langfuse changed owners in January, migration checklists are circulating. No hosted platform can match the mechanistic half.
*How:* Extend `session.build()` with `trace=` and `step_ref=`, validated through the same bounds discipline `_patch()` and `_boundary()` use. Follow the additive-section pattern — `FORMAT_VERSION` does not need to move; old readers ignore an unknown key, which is how the `patch` block already works. Mirror the parsing in `frontend/src/viewer.ts` and extend `tests/test_viewer_parity.py`.
*Effort:* L
*Caveat:* Two blocking items, not caveats. The recorder's redaction (`modelmri_record/redact.py`) runs at delivery, so a bundle assembled server-side from stored steps bypasses it entirely — export needs its own redaction pass over trace *and* `.mri` prompt/generation fields, plus a preview of what is about to leave the machine. And the trace half needs its own size budget against the current ~54 KB baseline: a long run with 20,000-character payloads per step is orders of magnitude bigger.

**23. Full-text search across every recorded trace**
A search box over every trace on the machine: full-text across step inputs and outputs, combined with structured filters (`kind:tool_call`, `error:true`, `duration>2000`, `name:pytest`). Results are steps, not runs, and clicking one opens that run scrolled to that step. The response states which engine answered — SQLite FTS5 or a substring scan — and, for FTS5, that matching is by whole word so a multi-word query is a contiguous phrase.
*Ahead of:* Langfuse needs ClickHouse for this and Braintrust built a bespoke columnar store; Langfuse users report ClickHouse eating 2 GB+ on a personal deployment plus documented delete-path landmines. FTS5 is compiled into essentially every CPython SQLite build, so this is the same capability inside a pip install with no container, no daemon, no memory floor.
*How:* FTS5 virtual table over `step(input, output, name)` in `traces._SCHEMA`, created inside a try/except so a Python built without FTS5 degrades to a `LIKE` scan reporting `engine: "substring-scan"` — the same degrade-and-say-so shape as the existing WAL guard. Structured filters parsed by a small allow-listed tokeniser in new `modelmri/trace_query.py`, never string interpolation into SQL.
*Effort:* M
*Caveat:* Backfill the index once on upgrade, not on every start, and not blocking the server banner. Every read goes under `TraceStore`'s existing `self._lock` — the two previously unguarded readers already caused intermittent 500s from `GET /api/traces`, and a new query path is a fresh chance to repeat that exact bug.

**24. Structural pattern findings, with no model asked**
Run a structural analysis over one run or across every run of the same agent and get exact, countable findings: this (kind, name, input) triple executed 14 times, this tool failed and was retried 6 times in 4 seconds, this call sequence repeated as a cycle of length 3. Each finding links to the steps it is made of, and cross-run it reports how many of the N recorded runs contain it — "12 of 19 runs".
*Ahead of:* Laminar's Signals asks an LLM to extract a described behaviour, LangSmith Engine clusters issues with a model, Braintrust's Loop proposes metrics from patterns — three model judgements dressed as findings, none of which runs without a cloud LLM. A loop is a structural fact about a graph, computable exactly, offline, in milliseconds.
*How:* New `modelmri/patterns.py` over the step list from `TraceStore.get_trace`: repeat detection by `hashlib.sha256` over `(kind, name, input)`, retry storms by grouping consecutive same-name error steps inside a time window, cycles by repeated subsequences in the kind+name sequence. Cross-run aggregation reuses the name grouping `AgentsPanel.tsx` already computes.
*Effort:* M
*Caveat:* A legitimate loop — paginating an API 14 times — is structurally identical to a pathological one, so findings are counts and never verdicts; the moment one is worded as a verdict it becomes the model judgement this exists to avoid. Input hashing misses near-repeats (a timestamp in a prompt), and that belongs in the UI copy, not a docstring.

**25. Truncation markers in the step inspector**
`traces._clip` already truncates payloads at 20,000 characters and appends a marker that the UI currently renders as if the agent produced it. Parse that suffix server-side into a `truncated_by` field so a clipped tool output shows "+18,412 characters not stored" as a marker rather than ending mid-sentence.
*Ahead of:* Every tool in this list caps stored payloads and none of them says so on screen. A truncated tool output that reads as a complete one is how you debug the wrong thing for an hour.
*How:* Parse `_clip`'s `… [+N]` suffix in `traces.py` into a per-step field, render as a marker in `AgentsPanel.tsx`. Ship alongside the `duration_ms` nullable migration (below).
*Effort:* S
*Caveat:* Surfacing the cap invites "why not store it all", which needs an answer about SQLite row size rather than a silent raise.

**26. `duration_ms` becomes nullable**
`traces.py` declares `duration_ms INTEGER NOT NULL DEFAULT 0`, so a step recorded bare is currently indistinguishable from a zero-duration step — the same class as the 0.10 default bug. Migrate to nullable so "duration not recorded" survives the round trip and renders as a marker rather than a zero-width bar.
*How:* `ALTER TABLE` migration in `traces._SCHEMA`, plus the render path in `AgentsPanel.tsx`.
*Effort:* S
*Caveat:* None. This is a defect fix, listed separately because it is a wire-format change other features (the waterfall, the transcript) depend on.

**27. Instrumentation that refuses to patch an SDK it does not recognise** *(reviewers split: 8 KEEP / 6 REVISE — the revision, adopted, narrows the fingerprint to Anthropic only)*
Before monkey-patching a provider SDK, check the call's signature and the response object's shape against a recorded fingerprint. When they no longer match, do not patch: print one line naming the package, the installed version and the attribute that moved, and mark every step it would have produced "not captured" instead of a span with empty fields. `python -m modelmri_record doctor` prints whether the SDK on this machine is instrumentable and what changed if not.
*Ahead of:* This is the systemic failure across the category: AgentOps breaking on `openai.resources.beta.chat` and on `google.adk.telemetry`, Langfuse dropping PydanticAI's instructions field, Opik's LangChain tracer leaking per-trace state. Every one is auto-instrumentation monkey-patching a moving SDK and then emitting an incomplete span rather than failing loudly.
*How:* New `packages/modelmri-record/modelmri_record/verify.py` with a fingerprint covering only the attributes actually read — for Anthropic: `usage.input_tokens`, `usage.output_tokens`, `content[].text`. `instrument_anthropic()` today patches after only a `getattr(Messages.create, "_modelmri_wrapped")` check and then reads `getattr(result, "usage", None)`, so a moved attribute silently yields a span with empty token fields. Gains a verify gate and returns a reason string instead of a bare bool. Per-step `captured: "full" | "partial"` in the new `meta` column.
*Effort:* M
*Caveat:* A fingerprint that is too strict refuses a working SDK after a harmless minor release, which is worse than a partial span — cover only attributes actually read, and the refusal must say how to force it. Fingerprint Anthropic only, the one instrumentation that actually exists. Three providers verified beats seventy claimed; one verified beats three guessed.

**28. Token ledger (no bundled prices)** *(reviewers split: 8 KEEP / 4 REVISE — the revision, adopted, drops the shipped price map)*
Every LLM step shows provider-reported input/output/cache-read/cache-write/reasoning tokens, rolled up per subtree and per run. Token counts the provider did not return read "not reported by provider", never 0. Cost appears only if the user points `MODELMRI_PRICES` at their own file, with exact-string model matching and no regex; an unpriced call reads "no price on file for `<model>`" and the run total reads "partial — 3 of 11 calls unpriced".
*Ahead of:* Every competitor derives cost from a hand-maintained price map — Langfuse matches model names by regex with priority-ordered tiers, and a regex matching the wrong model produces a plausible dollar figure with no signal it is wrong. OTel deliberately defines no cost attribute for exactly this reason.
*How:* Extend the `step` table with `tokens_cache_read`, `tokens_cache_write`, `tokens_reasoning` as nullable columns, so "not reported" is distinguishable from zero, migrated with `ALTER TABLE`. `instrument_anthropic()` already reads `result.usage`; extend to `cache_read_input_tokens` / `cache_creation_input_tokens`. Rollups server-side in `TraceStore.get_trace`. Optional `modelmri/pricing.py` with exact-key lookup only and a test asserting no prefix or regex matching.
*Effort:* M
*Caveat:* No bundled `prices.json`. A price map goes stale between releases and a user on a six-month-old release would see six-month-old prices; for an audience running local models the honest unit is tokens, and tokens are free. Cache-token pricing is asymmetric per provider and getting the write-vs-read multiplier wrong is a silent error.

**29. Retrieval, embedding, rerank and guardrail step kinds** *(reviewers split: 7 KEEP / 5 REVISE — the revision, adopted, softens whole-trace rejection to a per-step partial)*
Add first-class kinds for the parts of an agent that are currently invisible. A retrieval step carries the documents it returned as a list with ids and scores; the inspector renders them ranked, and the run-vs-run diff shows which document fell out of the top five. Today a RAG failure looks like an anonymous tool call.
*Ahead of:* `VALID_KINDS` is `{llm_call, tool_call, subagent, mcp_call, user_turn, error}` — coarse against MLflow's fifteen span types and OpenInference's nine. Retrieval specifically is contractually shaped elsewhere (MLflow's RETRIEVER spans expect a document list; OpenInference reserves document.id/content/score) but only as documented expectation.
*How:* Extend `VALID_KINDS` and add a per-kind shape check in `import_trace`. Documents in the nullable `meta` column as `{"documents": [{"id", "score", "text"}]}`, bounded by the same `_clip` discipline. `KIND_COLOR` and the legend extended in `AgentsPanel.tsx`.
*Effort:* S
*Caveat:* A retrieval step with no documents is a per-step `captured: partial` with a named reason, not a whole-trace `BadRequest` — an older ModelMRI receiving a newer recorder's `retrieval` step already rejects the whole trace on unknown kind, and a second hard shape check compounds skew into data loss. Publish the version-skew policy alongside.

**30. Structural CI assertions** *(reviewers split: 7 KEEP / 6 REVISE — the revision, adopted, drops cost gates to opt-in)*
`modelmri check <trace.json or trace-id>` exits non-zero against assertions you choose: no error steps, at most N steps, no retry storms. Plus a pytest plugin so a trace produced inside a test can gate a prompt or tool-definition change in CI. Timing and cost assertions exist but are opt-in and documented as flaky.
*Ahead of:* Opik ships a PyTest integration, Laminar runs evals in CI, Braintrust has a `bt` CLI — all three need their platform reachable from the runner, which means an API key in the build and a vendor in the critical path of every merge. A stdlib-only recorder and a SQLite file run inside a GitHub Actions container with no network and no account.
*How:* New `modelmri/check.py` wired as a `check` subparser in `modelmri/cli.py`, reading a JSON file or the local `TraceStore` via `paths.trace_db_path()`. Reuses `modelmri/patterns.py` for `--no-loops`. Pytest plugin in `packages/modelmri-record/modelmri_record/pytest_plugin.py` registered via `[project.entry-points.pytest11]`, exposing an `mri_trace` fixture wrapping `trace()` with delivery redirected to an in-memory document.
*Effort:* M
*Caveat:* Wall-clock gates fail on shared CI runners for reasons unrelated to the change — default set is structural, timing opt-in. Be honest in the docs that this requires the user's agent to be callable from a test; the plugin cannot paper over an agent that only runs as a CLI.

**31. Local judge that reads probability mass, not sampled text**
Point a rubric at a set of steps ("did this answer use the retrieved document") and have the loaded model score them locally. The score is not a sampled label: it is the model's probability mass on the verdict tokens read off the logits, run over k paraphrases of the rubric, reported as min/median/max with the model, dtype and seed beside it. When the verdict tokens carry almost no mass, it refuses — "this model did not answer the rubric."
*Ahead of:* Langfuse, LangSmith, Opik and Weave all run LLM-as-judge and all return a single number from a single sample of a hosted model — a sample presented as a property. Reading the probability instead of sampling the text is only possible if you hold the weights.
*How:* New `modelmri/judge.py` reusing what `lens.py` and `ModelRuntime._capture` already do: build the rubric prompt, one forward pass, `softmax` over the verdict token ids at the final position. Refuse when p(yes) + p(no) falls below a stated floor, raising `Refusal` from `modelmri/errors.py` so it answers 409 like every other deliberate no. One forward pass per paraphrase, not a generation, so it fits 8 GB.
*Effort:* L
*Caveat:* A small local judge is a weak judge, and a well-calibrated report of a weak judge's opinion is still a weak judge's opinion — name the judge model next to every score and never aggregate into a project-level metric, which is where the number would start being treated as a property. Verdict-token selection is tokenizer-dependent (`" yes"` vs `"yes"`); make that an explicit refusal when the tokenizer produces no single unambiguous verdict token, not a heuristic. Ship after the deterministic rubric predicates.

**32. Deterministic rubric predicates** *(reviewers split 5/4, both REVISE — the LLM-judged half is rescoped into #31, this is what remains)*
Score every recorded trace against exact predicates: regex over tool input, step-kind counts, error steps, duration outliers. Filter and chart the results. No model involved, so no calibration gate is needed and nothing is a judgement.
*How:* New `modelmri/rubric.py` plus a `rubric` table in the existing schema in `modelmri/traces.py`. Predicates run as SQL against the `step` table, which already indexes `(trace_id, seq)`. `AgentsPanel.tsx`'s trace list becomes filterable and chartable, clicking through to the single-trace timeline that ships today.
*Effort:* M
*Caveat:* Duration outliers over three traces are not statistics — print n and refuse to flag below a stated minimum.

**33. Read Inspect `.eval` logs (reader only)** *(reviewers split: 7 KEEP / 5 REVISE — the revision, adopted, drops the writer)*
Drop a UK AISI Inspect `.eval` log onto the agents panel and its samples, messages, tool calls and scores render as ModelMRI traces — same timeline, failing sample highlighted. The import result lists, per field, what was mapped and what was dropped.
*Ahead of:* Inspect is where eval interop is consolidating — Docent integrates natively, Apollo publicly adopted it. Speaking it turns the timeline into a second viewer for logs people already have, and stops `.mri` being a private dialect.
*How:* New `modelmri/inspect_io.py`. An `.eval` is a zip of JSON, so stdlib `zipfile`/`json` — no new dependency, works air-gapped — mapping messages and tool events onto `traces.VALID_KINDS`. Version-gate on the log's own schema field and refuse an unrecognised version with the version named, the same discipline `session.parse` applies to `format_version`.
*Effort:* M (reader only)
*Caveat:* Inspect's schema is not frozen; map only fields actually present and report the dropped ones in the response. No writer — committing to track an unfrozen schema in both directions forever is not solo-maintainer work, and someone with Inspect logs already has Inspect's viewer.

---

### Theme 5 — GGUF and quantisation

**34. Open a GGUF and read what is inside it**
Click any `.gguf` the scanner already found and get a reader view instead of a refusal note: every metadata key/value (architecture, context length, rope params, tokeniser model, quantisation version), a sortable tensor table (name, ggml type, dims, element count, bytes, file offset), and per-layer roll-ups showing effective bits-per-weight and which tensors were left at higher precision (`output.weight` and `token_embd.weight` frequently are). Nothing is loaded, no GPU touched — memory-mapped, so a 4 GB GGUF costs a few MB of RAM. The same view works on an Ollama blob resolved from its manifest.
*Ahead of:* llama.cpp ships this as `gguf_dump.py` console output plus an unmaintained Qt editor; HuggingFace ships `npx @huggingface/gguf` as a text dump. LM Studio, Jan, Ollama and Open WebUI show a quant *label* and a file size and nothing else.
*How:* New `modelmri/gguf_read.py` wrapping `gguf.GGUFReader` (pip package `gguf`, pure-Python + numpy, `numpy.memmap` backed). Effective bpw is exact arithmetic (`n_bytes*8/n_elements` per tensor, summed per layer prefix), not estimation. Ollama blob resolution reuses `modelmri/ollama.py::resolve` plus manifest layer digests. Flip `discover.LOOSE_WEIGHTS[".gguf"]` from a flat refusal to "inspectable, not loadable by transformers unless dequantised". `gguf>=0.10` as an optional `[gguf]` extra.
*Effort:* M
*Caveat:* `gguf-py` is versioned with llama.cpp and new ggml quant types appear regularly — an unknown `tensor_type` enum renders as its raw integer with "unknown ggml type", never silently bucketed. Bits-per-weight must not be averaged into a headline number without naming excluded tensors. Ollama's on-disk layout is undocumented and has changed before: resolve defensively and say "could not locate the blob" rather than guessing.

**35. Run a GGUF through the real introspection stack**
Load a GGUF into ModelMRI proper — not text-only through Ollama — and every existing panel lights up: attention arcs, head ranking by ablation, activation patching, logit lens, the "What changes?" diff. The GGUF dequantises into a genuine PyTorch `nn.Module` on load, so the forward hooks the whole codebase is built on apply unchanged. Before loading, the panel states the dequantised memory cost (computed exactly from the tensor table), the detected architecture family, and the caveat that the numbers come from these weights run through HuggingFace kernels, not llama.cpp's.
*Ahead of:* Zero of eleven runners expose attention; vLLM, the only one exposing hidden states, calls its own GGUF support "highly experimental." Three dead feature requests (llama.cpp discussion #3660, llama-cpp-python #237 and #1141) all died on the C++ wall. Every competitor's introspection story is blocked on their own engine; none has a PyTorch hook stack to fall back on.
*How:* New `modelmri/gguf_load.py` calling `AutoModelForCausalLM.from_pretrained(dir, gguf_file=name)` and the matching tokenizer — transformers' shipped GGUF path. Wire as a third `ModelRuntime.backend` value alongside `"hf"`/`"ollama"`, keeping `attn_implementation="eager"` so `_capture` still materialises attention. Gate with `capacity.guard` using an exact dequantised size from `gguf_read.py`. Refuse unsupported architectures by name against transformers' `GGUF_SUPPORTED_ARCHITECTURES` and unsupported ggml types by enum. `ModelStatus` gains `source="gguf"`, `quant`, `dequantised_to`.
*Effort:* L
*Caveat:* Dequantisation blows memory up 4–8×, so on 8 GB this is a 1–3B-class feature and the UI must say so before the click, not after OOM. transformers' coverage lags llama.cpp and IQ-series quants are largely unsupported; every unsupported case refuses with the specific reason. Loading is slow (per-tensor dequantise) and needs `modelmri/progress.py`'s killable child. Biggest honesty risk: users will read these as "what llama.cpp does", so the kernel-path caveat goes in the panel header, the `.mri` `meta` and `modelmri inspect` — not the docs.

**36. Measure what your quantisation actually cost**
Point at a quantised model and its full-precision original (or two quants of the same model) and get a damage report: per-tensor weight error (RMS, max absolute, cosine similarity, fraction of weights that changed sign), then on one prompt, per-position KL between the two next-token distributions, per-layer attention divergence over the identical token sequence, and a list of positions where the argmax token flipped with both candidates shown. The weight half runs on CPU in numpy, one tensor at a time, on any machine.
*Ahead of:* `llama-perplexity --kl-divergence-base` measures exactly one of these numbers, is CLI-only, and demands an 11–37 GiB FP16 logit dump first. `llama-imatrix --show-statistics` frames activation stats as calibration, not damage. Nobody joins weight-level error to behaviour-level divergence, and it needs both a GGUF reader and a hookable forward pass in one process.
*How:* New `modelmri/quantdiff.py`. Stream tensors from `gguf_read.py`, dequantise with `gguf.quants.dequantize`, map GGUF names back to HF via transformers' tensor-name mapping, compare against the FP16 side via the safetensors header + `safetensors.numpy.load_file` — one tensor resident at a time. Behaviour side reuses `ablate.kl_nats` and `distribution` so the two agree, with the second model loaded after the first is unloaded.
*Effort:* L
*Caveat:* Tensor-name mapping is transformers' table and incomplete for some architectures — unmapped tensors are listed as "not compared", never dropped from totals. It measures the quantiser's damage through HF kernels, not llama.cpp's end-to-end damage. One prompt is one sample: route it through the sweep (#16).

---

### Theme 6 — Fit, telemetry and integration

**37. Fit calculator that grades its own prediction**
Before loading anything, show the arithmetic: weight bytes read exactly from the safetensors header or the GGUF tensor table, KV-cache bytes as `2 × n_layers × n_kv_heads × head_dim × seq_len × dtype_bytes` with every term visible and a sequence-length slider, and the eager-attention buffer ModelMRI itself needs. It answers "what is the longest context this model can hold on my card" with a number you can check by hand. After the load, the same panel shows the measured allocation next to the prediction and names the gap as unpredicted runtime overhead.
*Ahead of:* KoboldCpp #1480 is literally a request for this calculator and it went unbuilt; the standing community answer to "why won't my 8 GB card run this" is still "use the GGUF version." LM Studio and Jan show an *estimate* with no formula. Nobody closes the loop by grading the prediction.
*How:* New `modelmri/fit.py`. Safetensors header is 8 bytes of JSON length then a JSON dict of dtype/shape/offsets — no torch import needed. KV geometry from `config.json` (`num_hidden_layers`, `num_key_value_heads` falling back to `num_attention_heads`, `hidden_size`, `head_dim`), refusing and naming the missing key rather than substituting a default. Budget from `devices.detect().vram_gb`. Feed the numbers into `capacity.guard` so the download refusal and the calculator quote the same arithmetic.
*Effort:* M
*Caveat:* The formula is exact for standard MHA/GQA and wrong for MLA, sliding-window and hybrid-SSM — refuse those by name, do not approximate. Activation and workspace memory is not predicted and is labelled excluded. `device_map="auto"` offload makes "fits" a multi-device question; scope v1 to a single accelerator and say so. Shares probe plumbing with #1 — build them together.

**38. Telemetry bar with the introspection cost broken out**
A persistent bar showing, for the run you just did: tokens/sec measured over the streamed tokens, prompt-processing time separated from decode, peak accelerator memory from the allocator, context fullness against the model's own `max_position_embeddings`, and a separate line for what introspection cost — because ModelMRI forces eager attention and `output_attentions=True`. It names the eager-attention memory as its own number, `n_layers × n_heads × S² × 2 bytes`, which at S=4096 on a 12-layer, 12-head model is 4.6 GB, and warns before a run that would exceed the card.
*Ahead of:* Live telemetry is table stakes (TextGen's tokens/sec, LM Studio's Developer page, llama-server `/metrics`), so this closes a visible gap. The introspection-cost line is the differentiated half and pre-empts the fair complaint that ModelMRI is slower than Ollama.
*How:* New `modelmri/telemetry.py` timing around `generate_stream`'s existing `TextIteratorStreamer` loop and `_capture`'s forward pass; memory from `torch.cuda.max_memory_allocated`/`memory_reserved` with the XPU/MPS equivalents `devices.py` already knows about. Context fullness from `tokenizer.model_max_length` and `config.max_position_embeddings`, refusing to display a percentage when the model reports a sentinel like `1e30`.
*Effort:* S
*Caveat:* Allocator numbers are PyTorch's view, not the driver's — label them "allocated by PyTorch", not "VRAM used". CPU and MPS report far less than CUDA; those cells read "could not measure", never zero. One generation is one sample: print prompt and sequence length beside tokens/sec.

**39. OpenAI-compatible `/v1` that returns the internals**
Point any existing client, harness or eval loop at `http://127.0.0.1:5900/v1` and it works: `/v1/models`, `/v1/chat/completions` (streaming and not), `/v1/completions`, with genuine `logprobs`/`top_logprobs` read from the real logits. Then pass `"modelmri": {"lens": true, "heads": 10, "attribute": true}` and the response carries a `modelmri` block with the logit-lens trajectory, the top ablation-ranked heads and per-token attribution for that exact completion, plus an `.mri` id. Any OpenAI parameter ModelMRI cannot honour (`n>1`, `logit_bias`, `seed` on a backend that ignores it) returns a 400 naming the parameter.
*Ahead of:* Every runner exposes `/v1` and ModelMRI cannot currently be dropped into any client, so this closes a hard blocker — and inverts it, since everyone else's `/v1` returns text. Refusing unsupported parameters is the opposite of the category norm, where llama.cpp and Ollama both silently ignore them.
*How:* New `modelmri/openai_api.py` mounted onto the existing app in `server.create_app`. `/v1/models` from `discover.discover`. Chat completions wrap `generate_stream` and reuse `ablate.distribution` on captured logits for `top_logprobs` (real values — the stack already keeps them). The extension block calls `ablate_heads`, `lens.logit_lens` and `attribute_tokens` after the completion commits. SSE `data:` frames matching OpenAI's shape.
*Effort:* L (one reviewer called it M; the SSE shape, real top_logprobs and the compatibility tail are more than medium)
*Caveat:* OpenAI compatibility is a long tail — claim only the implemented surface and enumerate the rest rather than half-supporting it. The internals block roughly doubles or triples latency and must report the measured extra time. Binding beyond `127.0.0.1` turns this into an unauthenticated remote path onto the user's GPU: loopback default, explicit `--host`, printed warning, no exceptions.

**40. MCP server, read-only first cut** *(reviewers split: 7 KEEP / 6 REVISE — the revision, adopted, defers `load_model` and `export_mri`)*
Run `modelmri mcp` and Claude Code, Claude Desktop, Cline or any MCP client gains tools it can call: `list_models`, `status`, `rank_attention_heads`, `attribute_tokens`, `logit_lens`, `inspect_mri`. An agent debugging a model can ask "which heads carry this answer" and get structured JSON. Every tool refuses with the same sentences the HTTP API uses, so an agent never receives a fabricated number it will confidently repeat.
*Ahead of:* MCP has become a client-side convention — LM Studio is an MCP host with OAuth, Jan and Open WebUI are hosts, llama.cpp's WebUI merged a browser-side client in March 2026. Not one tool in the category exposes model inspection *as* MCP tools.
*How:* New `modelmri/mcp_server.py` speaking JSON-RPC 2.0 over stdio, stdlib only (matching `packages/modelmri-record`'s posture). Each tool is a thin adapter over an existing `ModelRuntime` method, so there is exactly one implementation per measurement and MCP and HTTP answers cannot drift. `--attach http://127.0.0.1:5900` to drive an already-running server instead of loading a second copy.
*Effort:* M
*Caveat:* Demand is speculative — the honest current answer to "which agent wants to rank attention heads" is "the maintainer, dogfooding." Defer `load_model` (must serialise on `ModelRuntime._lock` and answer "busy loading X") and `export_mri` (writes a file on an agent's say-so; confine to `paths.data_dir()`). Pin the protocol version reported in `initialize` and refuse an unknown one rather than best-efforting.

**41. Constrained-decoding mask viewer, on `lm-format-enforcer`** *(both reviewers REVISE — both said take the proposal's own alternative and skip the compiler)*
Turn on constrained decoding with a JSON schema and get a per-step view: the model's unconstrained top-k beside the grammar's allowed set, the probability mass the mask deleted, and a flag on every step where the token the model most wanted was forbidden. Steps where the mask removed most of the distribution are highlighted — that is where your structured output stopped being the model's answer and started being the schema's.
*Ahead of:* Ollama, llama.cpp (GBNF), vLLM and LM Studio all ship constrained decoding as a black box: you get valid JSON and no idea what it cost. Nobody shows the mask's effect on the distribution, and it is a direct diagnostic for the widespread complaint that structured-output mode makes models dumber.
*How:* New `modelmri/grammar.py` depending on `lm-format-enforcer` (pure Python, offline) as an optional extra, passed through `generate_stream`'s existing `gen_kwargs` as a `LogitsProcessor` that records `(step, pre_mask_logits_topk, allowed_count, deleted_mass, wanted_token, wanted_was_allowed)`. Same buffer shape `lens.py` produces, so `LensPanel.tsx`'s rendering is reusable.
*Effort:* M (with the dependency; L+ and research-grade without it)
*Caveat:* Do not hand-write an incremental token-level grammar compiler — a wrong mask is an invisible failure in a tool whose premise is not shipping invisible wrong answers. Deleted probability mass is measured on the pre-softmax distribution at that step under that sampler, and carries its setup.

**42. OTLP ingest, JSON only** *(reviewers split: 6 KEEP / 4 REVISE — the revision, adopted, drops the emit side)*
ModelMRI's server accepts an OTLP/HTTP JSON POST, so anything instrumented with OpenLLMetry, OpenInference or the Vercel AI SDK renders in the agents timeline without ModelMRI writing a provider integration. Every trace stores which semconv generation it was written against, and the UI prints it.
*Ahead of:* Matches the ingest side of Langfuse/LangSmith/Braintrust/Weave interop. Beats them on honesty: they present `gen_ai.*` as a standard, while it was deprecated out of the main semconv repo on 2026-06-12 into `semantic-conventions-genai` with no releases, no tags and nothing marked stable.
*How:* New `modelmri/otel.py` with `from_otlp(payload) -> trace_doc`, mapping `gen_ai.operation.name` onto `VALID_KINDS` and reading `gen_ai.input.messages` / `gen_ai.output.messages` / `gen_ai.usage.*`. New route `@app.post("/api/otel/v1/traces")` next to the existing `/api/traces/import`, reusing `TraceStore.import_trace` and its `BadRequest` contract. Convention generation stored as `meta.semconv` and surfaced in `AgentsPanel.tsx`.
*Effort:* M
*Caveat:* OTLP's common wire format is protobuf; the stdlib-only constraint means JSON only, so a large share of real collectors will not connect — refuse `application/x-protobuf` with a sentence naming the limit rather than mis-parsing. The `gen_ai.*` names have already churned (v1.41.0 split `invoke_agent` into CLIENT vs INTERNAL) and the mapping table will need re-pinning; recording the pinned generation is what keeps that honest. No emit side — the recorder already delivers to ModelMRI.

---

### Theme 7 — Robotics

The robot half is currently correlational — it paints attention, which by this project's own standard is the weak version. Two of these run today with nothing new installed. The rest are gated on a sidecar that is a release line, not a feature.

**43. Audit a robot dataset for the corruption that does not crash**
`modelmri vla audit lerobot/pusht` proves — or disproves — that the episode-to-frame mapping tiles the frame table exactly with no gaps and no overlaps, that every episode's video routing lands inside a file that exists and covers its span, that recorded normalisation statistics match the data they describe, which state and action dimensions are constant, and whether actions lag state. It reports contradictory demonstrations as named episode pairs with the distance thresholds printed, and deliberately does not issue a grade.
*Ahead of:* Traceplane found a metadata boundary bug in v2.1-to-v3.0 conversion that silently corrupts episode-to-frame mapping, and a float drift that breaks frame decoding after roughly 45 episodes. That is exactly the class of bug that produced ModelMRI's own 0.10 failure — a `.get(name, 0.0)` that made 206 episodes show one video. ORBIT ships an A-to-F readiness grade calibrated on 82 other people's datasets; refusing to grade and showing the checks is a stronger position.
*How:* New `modelmri/vla_audit.py` on `LeRobotV3Reader`. Tiling proof from `EpisodeInfo.data_from` + `length` against `len(_frame_table()['episode_index'])`. Routing proof per camera by walking `video_chunk`/`video_file`/`from_ts`/`to_ts` against the files `_video_file` resolves and PyAV's container duration, plus a sampled first/last-frame decode with a hash comparison — the direct generalisation of the 206-episodes bug. Normalisation from `meta/stats.json` against the parquet columns via pyarrow. Nearest-neighbour contradiction search with epsilon and delta stated in the output.
*Effort:* L
*Caveat:* The contradiction check can mislead — two similar states with different actions is often legitimate multimodality, so present as pairs to inspect with thresholds visible, never as a defect count. Nearest-neighbour is expensive; a stated frame cap, and a capped scan is a sample. Cross-correlation lag assumes a fixed control frequency, which `info.json` states; refuse when it does not. **Needs no sidecar, no policy, no GPU, no lerobot.**

**44. Causal patch occlusion on the camera frame**
A toggle next to the layer slider: "attention" or "caused it". The second occludes each block of the camera frame in turn, re-runs the policy, and paints how far the output actually moved — with a control band from N seeded occlusions of the same area at random locations, so a block that beats nothing is drawn as beating nothing. Beside the two maps, the rank correlation between them for *this* frame, so you can see the attention map and the causal map disagree on your own checkpoint.
*Ahead of:* Embodied Interpretability measured that interventional masking beats attention weights on explanation fidelity, and VLA-Trace uses attention knockout rather than attention viewing. The field has concluded attention is the weak version and ModelMRI ships exactly the weak version. Dr.VLA and Embodied Interpretability are research artefacts; physical-AI-interpretability paints attention only.
*How:* New `modelmri/vla_occlude.py`. Tile at the tower's own patch grid, already exposed as `VLAStatus.grid` (32×32 on SmolVLA), stride configurable, two fill baselines named on screen the way `ablate.BASELINES` are (episode-mean pixel, and the tower's normalisation midpoint). Perception-only, score the shift in the tower's pooled output in units of that embedding's per-dimension std measured over the episode. Controls follow `patch.py` verbatim: fixed `CONTROL_SEED`, N same-area draws, per-block `clears_control`.
*Effort:* L
*Caveat:* Occlusion is out-of-distribution for the encoder — a grey box is itself a stimulus, which is why two fill baselines ship rather than one. The perception-only score is a shift in an embedding, not an effect on behaviour, and must never be labelled "caused the action" when the sidecar is absent. A full 32×32 sweep is 1,024 tower passes: coarse grid by default, fine opt-in, frame count and estimated seconds on screen before the run.

**45. Compare two checkpoints over the same frames — tower half now, behaviour half later**
Point at a base checkpoint and your fine-tune. ModelMRI runs each over identical frames of identical episodes and shows where in the network they diverge (per-layer representation distance between the two towers on the same input). If the checkpoints disagree about anything that would make the comparison meaningless — action dimensionality, normalisation statistics, image size, camera keys — it refuses and names the field that differs. The behaviour half (predicted-action distance per frame) is gated on the sidecar and labelled as absent.
*Ahead of:* rollout-doctor warns when harness, config, task set or sample size differ between runs and TRI STEP does sequential A/B, but both treat the policy as opaque, so neither can say *where* the fine-tune changed the model. VLA-Trace did CKA-style representation drift and shipped no code.
*How:* New `modelmri/checkpoints.py` (not `compare.py` — `tests/test_compare.py` owns that name). Sequential: load A, run the strided frame set, write per-frame outputs and per-layer pooled activations to a run file under `paths.py`'s data root, unload, load B, repeat, compare. Layer capture reuses `ModelRuntime._capture`'s hook pattern. Distance is plain centred-kernel alignment plus cosine, both named. Compatibility fields read the way `vla._vision_config` reads them; mismatch raises `Refusal` naming the field.
*Effort:* L
*Caveat:* Representation distance is descriptive, not causal — it says where they differ, not which is better, and the panel must not imply a winner. Two full loads is minutes; progress and cancellation via `modelmri/progress.py` or people assume it hung. Say plainly which half ran.

**46. `.mri` section for robot findings**
The share button in the VLA panel writes a `.mri` holding the camera frame, per-layer attention, the causal occlusion map with its control band, the knockout bars, and exactly which policy revision, dataset, episode, timestep and camera produced them. It opens in the existing browser viewer with nothing installed.
*Ahead of:* There is no portable, no-account artifact for robot-policy internals anywhere. Foxglove archived its open-source Studio for a paid platform; Rerun's `.rrd` carries what the robot recorded, never what the network computed; HF Spaces need an upload and an account.
*How:* Add a `vla` section to `session.py` the way `patch` was added — optional key, never written empty, its own `_vla()` validator mirroring `_patch()`'s bounds, so older readers ignore it and `FORMAT_VERSION` does not move. Attention grids are 32×32 through the existing `_quantise`; the frame is the PNG data URL `vla_data.encode_png` already produces. Extend `frontend/src/viewer.ts` and `cli.inspect_session`.
*Effort:* M
*Caveat:* Two blocking items. A stated frame resolution cap with the downsampling named — a silently shrunk frame under a causal map is a wrong picture. And the same untrusted-input treatment `_patch()` got, including finite-value and rectangularity checks, since a `.mri` is designed to arrive from a stranger. Sequence after #44; there is nothing to distribute before it.

**47. Read ALOHA/robomimic HDF5 datasets** *(reviewers split: 6 KEEP / 5 REVISE — the revision, adopted, drops the RLDS half)*
Point the VLA panel at an ALOHA-style HDF5 file and it opens the same way a LeRobot dataset does — episodes, cameras, states, actions, frames — with the same panel, the same occlusion maps and the same audit.
*Ahead of:* `vla_data.py` reads LeRobot v3.0 only. "LeRobot format" does not mean one thing — GR00T still ships a v3-to-v2 downconverter and Rerun could not load v3.0 at all until a patch this year. ORBIT reads four formats and is winning the diagnose-before-training slot partly on that.
*How:* New `modelmri/hdf5_data.py` presenting the interface `LeRobotV3Reader` already presents (`episodes()`, `frame()`, `raw_frame()`, `cameras`, `use_camera()`, `summary()`), so `VLAPanel.tsx` and every `/api/vla/*` route is untouched. `h5py` in the `vla-lite` extra, following the ALOHA/robomimic convention (`/observations/images/<cam>`, `/action`). An unrecognised schema raises `Refusal` listing the top-level keys found — the shape `vla.discover_vision_prefix` gives today.
*Effort:* M (HDF5 only)
*Caveat:* HDF5 layouts vary between labs — refuse an unfamiliar one rather than guessing which dataset is the action. **RLDS is deliberately excluded**: hand-rolling a TFRecord framing reader plus a `tf.Example` protobuf parser is well beyond L, the plan skipped masked CRC32C validation, and accepting a silently corrupt record in the same release as a dataset-integrity auditor is an incoherent posture.

**48. Cross-episode sweep over metrics that run today** *(reviewers split: 7 KEEP / 5 REVISE — the revision, adopted, scopes v1 to non-sidecar metrics)*
Instead of one episode at a time, run a chosen measurement over a strided sample of every episode: attention entropy, or perception-only occlusion. You get an episodes-by-time heat strip and a sortable table where clicking a row drops you on that exact frame with the causal tools already pointed at it.
*Ahead of:* RoboLab's dashboard auto-logs typed failure events and jumps to the failing frame; Event-SAE clusters kinematic keyframes. Both are cross-episode over simulator-emitted event labels. Ranking over measured internals is the version nobody else can do.
*How:* New `modelmri/vla_sweep.py` writing to a `vla_sweep` table in the existing SQLite file. Metric callables come from `vla_occlude.py` and the existing attention path, so the sweep adds iteration, not measurement. Frame count and estimated seconds before starting, progress via `modelmri/progress.py`, cancellation via the killable-child pattern.
*Effort:* L
*Caveat:* No unsupervised failure-mode clustering with names — a cluster labelled "dropped the object" that ModelMRI never verified is exactly the fabrication it refuses. Rank by a stated measured quantity and say the ranking is by that quantity and nothing else. Print the stride, because a strided ranking may miss the worst frame.

**49. Action expert sidecar (`modelmri-policy`)** *(reviewers split: 8 KEEP / 6 REVISE — the revision, adopted, is that this is a release line, not a next-release feature)*
`modelmri policy install` builds a separate venv, installs lerobot (or openpi), and starts a local process holding the full policy — vision tower plus action expert. The VLA panel gains an "action" status beside "perception", and you can ask a robot policy what it would *do* on the frame you are looking at. Absent the sidecar, every action-dependent control says so and names the one command.
*Ahead of:* `vla.py:301` refuses with "the action expert needs the optional lerobot extra" and nothing implements it. Event-SAE measured that in pi0.5 the backbone shows almost no single-feature leverage while the action expert is where the sensitivity lives — a backbone-only tool is looking in the wrong place. physical-AI-interpretability, the only shipped competitor, is ACT-only and still needs a modified LeRobot fork.
*How:* New `packages/modelmri-policy/` mirroring `packages/modelmri-record/` — own pyproject, own wheel, pinned lerobot. Install via `venv` + a pip subprocess driven by `modelmri/progress.py`'s killable-child machinery. Sidecar serves stdlib `http.server` JSON on 127.0.0.1 with a contract version: `POST /act {frames, state, instruction, seed}` → `{action_chunk, dtype, device, policy_repo, revision, normalisation}`, plus `POST /hidden` returning captured activations as safetensors bytes. New `modelmri/policy.py` is the client, with a `Refusal` naming the install command when absent.
*Effort:* XL
*Caveat:* **Do not start this until the perception-only causal work (#43, #44) has shown the VLA panel has users.** A process boundary is the right permanent answer to lerobot's torch/numpy pin conflict — it is precisely why physical-AI-interpretability is stale on a fork — but this is a second wheel, a venv builder, a driven pip subprocess and an HTTP contract against a dependency whose API churns. The version handshake must be mandatory, not advisory: contract drift silently serving actions from a stale policy is the worst failure available here. Two processes each holding weights on 8 GB needs an explicit `capacity.guard` refusal, not a hope.

**50. Sidecar-gated robot features (blocked on #49)**
Listed together because none of them exists without the action expert, and each is M once it does.

- **Predicted vs recorded action, clickable into the internals.** Per-dimension track of demonstrator vs policy across the episode; click a spike, jump the scrubber, hand the frame to the occlusion map. Ahead of NVIDIA GR00T, whose open-loop predicted-vs-ground-truth curves are terminal — where, never why. New `modelmri/vla_actions.py`; recorded actions already come out of `LeRobotV3Reader._frame_table()['action']`, joint names from `meta/info.json` `action.names`. *Caveat, on the panel not in docs:* a recorded action is one human demonstration, not ground truth — a policy can be right and differ; and open-loop teacher forcing on recorded observations is not closed-loop behaviour where error compounds. Refuse when the policy's normalisation disagrees with the dataset's; overlaying two curves in different units is the plausible-wrong output.
- **Instruction-swap test.** Run the policy with its own task string and every other distinct task string the dataset contains; show the spread across instructions against the spread across noise seeds on the identical frame. If swapping "pick up the red block" for "close the drawer" moves the action less than re-rolling the sampler does, say so in those words. The reference is the policy's *own* sampling variance, so no threshold is invented and no calibration is borrowed. Refuse a single-task dataset (naming that fact — a fabricated distractor instruction would be inventing the experiment) and refuse a deterministic policy, where the reference collapses. Expect a narrow audience: most hobbyist recordings are single-task.
- **Input-stream knockout.** One bar per input the policy consumes — each camera, the instruction, proprioceptive state — showing how far the action moves when that stream alone is replaced by its episode mean. `vla_data.py` already enumerates every camera and was explicitly fixed to stop showing one view as the dataset. VLA-Trace did modality knockout and never released code. *Caveat:* mean-substitution is a specific baseline, not "removal", and single-stream knockouts do not add up — carry the non-additivity caveat in the response body following `ablate.py` and `attribute.py`'s existing `means` convention. Label the empty-instruction arm "no instruction", never "the instruction did not matter".

---

### Theme 8 — Interop and comparison (maintainer-requested)

The five gaps in this theme were raised directly by the maintainer. Three were
already covered by the list above and are cross-referenced rather than repeated.
Two had been cut in triage; both are reinstated here in the scoped form that
survives the objection that got them cut, and the objection stays on the record,
because being overruled does not make it wrong.

**Already covered above.** *Comparing two models or two checkpoints* is #19
(`modelmri diff a.mri b.mri`, with a CI exit code), #20 (finetune-vs-base over a
prompt set) and #45 (two robot checkpoints over the same frames) — and the 8 GB
constraint shapes all three: models load one at a time and derived results are
cached, never both sets of weights at once. *An eval loop* is #16 (the one
iteration engine), #30 (structural CI assertions), #31 (a local judge that reads
probability mass rather than sampled text), #32 (deterministic rubric predicates)
and #33 (read Inspect `.eval` logs). *A GGUF story* is #34, #35 and #36, and #34
is already ranked ninth in the ship list.

**51. Emit OTLP, version-stamped**
`modelmri traces export --otlp <endpoint>` sends a recorded run to whatever the team
already runs — Langfuse, Phoenix, Grafana, Honeycomb — and `modelmri-record` gains an
opt-in `deliver_otlp=` that is off by default. Every exported span carries the semconv
generation it was written against as an attribute, and the CLI prints which one it
targeted.
*Ahead of:* this is not an "ahead" feature and should not be sold as one. It is the
price of being adoptable inside a team that has already chosen a stack. Every
competitor in the tracing category ingests; a local tool that cannot hand its traces
onward is a dead end for anyone with an existing collector.
*How:* extend the `modelmri/otel.py` introduced by #42 with `to_otlp(trace_doc)`, the
exact inverse of `from_otlp`, sharing one mapping table so ingest and emit cannot drift
apart. New CLI verb in `cli.py`; OTLP/HTTP JSON over stdlib `urllib.request`, no new
dependency, which keeps `modelmri-record` stdlib-only.
*Effort:* M
*Caveat:* **reinstated against triage.** The reviewers cut this because `gen_ai.*` was
deprecated out of the main semconv repo on 2026-06-12 into `semantic-conventions-genai`,
which has no releases, no tags and nothing marked stable — so emit is a maintenance
obligation against a moving target, and a second delivery path to keep working. That
objection is correct and does not go away. What makes it survivable is that the pinned
generation ships as a span attribute and in the CLI output, so a consumer can always
tell which vocabulary a span speaks. JSON only, same as #42: protobuf collectors are
refused with a sentence naming the limit, not approximated.

**52. An attribution graph built from patching, not transcoders**
Run #11's edge patching over the sites the node grid already flagged and draw the result
as a graph — nodes are (component, layer, position), edges are the signed recovery one
sender contributes to one receiver, pruned to the edges that beat their own controls.
Click an edge for the eight control draws behind it. It answers "what wrote this?" across
a whole prompt instead of one cell at a time.
*Ahead of:* circuit-tracer's attribution graphs are the crown jewel of the category
(0 → 2,895 stars in 14 months) and they need transcoders, which exist for a handful of
models and whose gemma-2-2b set does not fit 8 GB. This builds a comparable object out of
patching, which needs nothing but the model already loaded. Different method, stated as
such: it is a patching graph, not a transcoder attribution graph, and the panel says so
rather than borrowing the more famous name.
*How:* `patch.py::path_trace` from #11 supplies the edges; a new `modelmri/circuit.py`
does the pruning and the transitive walk backwards from the answer. The front end reuses
`ArcCanvas.tsx` geometry rather than adding a graph library. New `.mri` section beside the
existing patching trace.
*Effort:* L, and strictly blocked on #11.
*Caveat:* **reinstated against triage in a different form.** What was cut was
reimplementing circuit-tracer, and that cut stands — see #53 for the part of it worth
having. Edge count is quadratic in sites, so the seeding rule and the prune threshold are
printed with the graph rather than hidden; a graph whose edges were chosen by an
undisclosed rule is a picture, not a measurement. Every edge carries the same eight
same-norm draws the node grid uses, and an edge that does not beat its controls is drawn
differently rather than dropped silently.

**53. Open somebody else's circuit-tracer graph**
`modelmri open graph.pt` renders a circuit-tracer attribution graph in the same viewer as
everything else, behind a banner naming the file, the tool that produced it and the model
it was computed on — and no claim that ModelMRI measured any of it.
*Ahead of:* nothing else opens one outside circuit-tracer's own Neuronpedia flow, and the
read-only path costs a fraction of the build the reviewers rejected.
*How:* reader in `modelmri/circuit.py` using `torch.load(..., weights_only=True)`; a new
`.mri` section so the graph travels the way every other finding does. Same validation
posture as `session.parse` — a ragged graph or an implausible node count stops at the
reader, not in the recipient's browser.
*Effort:* M
*Caveat:* the `.pt` layout has no versioning guarantee, so the reader pins a layout and
refuses an unrecognised one by name rather than guessing. The provenance banner is not
optional chrome: a rendered graph ModelMRI did not compute must never be mistakable for
one it did.

---

## Ship in the next release

Ten items. Everything else is later.

1. **Analysis cost preflight (#1)** — S, and it gates three other features on this list. Build first.
2. **Fit calculator (#37)** — shares the probe plumbing with #1; building them together means the download refusal and the calculator cannot disagree.
3. **Resample ablation + baseline disagreement (#2)** — the largest correctness debt in the repo, admitted in `ablate.py`'s own docstring. Needs #1 to be affordable.
4. **Random-weight control (#3)** — the strongest single expression of the ethos, and the only feature here that can conclude "this measurement is uninformative on your model."
5. **Held-out lens KL (#4)** — S, no training, no corpus curation, and it delivers the entire competitive claim against tuned-lens and TransformerLens on its own.
6. **Contrastive steering vectors + vector store (#8, #9)** — closes the largest coverage cliff: every model without a published SAE is unsteerable today. Highest user-pull-per-effort in the interpretability half.
7. **Adopt a recorded LLM call into the panels (#21)** — the product thesis, structurally unavailable to all seven agent-tracing competitors, and currently unbuilt.
8. **Trace search + truncation markers + nullable `duration_ms` (#23, #25, #26)** — one M plus two S. Two of the three are defect fixes in the same class as the 0.10 bug.
9. **GGUF reader (#34)** — M, no GPU, first-in-category GUI, and it is the on-ramp for #35 and #36.
10. **Telemetry bar (#38)** — S, and it pre-empts the entirely fair complaint that ModelMRI is slower than Ollama by naming the buffer that makes it so.

**Sequencing.** #1 and #37 share the probe pass and go first; #2 is unaffordable without #1's projected pass count, and the corpus sweep (#16) is unaffordable without both. #4 must land before #5 — measure the plain lens's error before training a replacement for it. #9 is built inside #8, not beside it. #21 is the prerequisite for #22 (nothing to bundle without it) and makes every existing mechanistic feature reachable from the agents panel, which is why it outranks features that are cheaper. #34 unlocks #35, which unlocks the behaviour half of #36 and feeds the exact dequantised size into #37's arithmetic. #26 is a wire-format change that the transcript and waterfall work would otherwise duplicate, so it lands with #25 now rather than later.

**Maintainer-requested additions.** #51 (emit OTLP) and #53 (open a circuit-tracer
graph) are both M and both independent of everything above, so they fit this release
alongside the ten. #52 does not: it is blocked on #11, which is an L in its own right, so
the honest sequencing is #11 in this release only if something else comes out, and #52 in
the one after. Saying all three fit would be exactly the kind of estimate this document
exists to avoid. The other three gaps raised are already here — #19 and #20 for comparing
models, #16 with #30–#33 for the eval loop, and #34 for GGUF, ranked ninth above.

The robot half deliberately does not appear in this list. #43 (dataset audit) and #44 (causal occlusion) are the two robot features that run today with no sidecar, and they are the right next-release-plus-one pair — but #49 should not start until they have shown the VLA panel has users.

---

## Deliberately not doing

**Reimplementing attribution graphs.** *(Partly reversed at the maintainer's request — see #52 and #53. The cut below stands for the transcoder pipeline itself; #52 builds the graph from patching instead, and #53 is the reader half this very paragraph already called the version that survives.)* circuit-tracer went 0 → 2,895 stars in 14 months and it is the crown jewel of the category. Transcoders exist for a handful of models, gemma-2-2b's set does not fit 8 GB, and the maintainer cannot run the flagship path on the dev machine. An XL build whose honest behaviour on most machines is a capacity refusal is not a feature. Opening and pruning *someone else's* graph with a provenance banner is the version that survives, and even that is parked — the `.pt` layout has no versioning guarantee and circuit-tracer's user base runs big GPUs.

**Agent replay from the failing step.** XL, and its own scope statement concedes the ceiling: it needs the user's agent to swap in a replay client at the provider-client site, and any agent with hidden external state — a database write, a clock read, a browser DOM — cannot be replayed faithfully no matter what is recorded. The divergence contract is the right ethos wrapped around a feature that does not justify the cost. The one piece worth keeping is a `seed` parameter on `ModelRuntime.generate_stream`, which is a line of code and belongs to #8's A/B discipline anyway.

**Replaying a hosted-API prompt through a local model.** Three proposals wanted this, all as a "substitute" path with loud labelling. The prompt your agent sent to Claude and the prompt you run through a local 0.5B are not the same experiment, and loud labelling does not stop the screenshot from travelling. When the weights are not on the machine, the answer is a sentence explaining that, not an approximation.

**Bundling a price map.** Every competitor derives cost from a hand-maintained map with regex model matching, and a regex matching the wrong model produces a plausible dollar figure with no signal it is wrong. OTel deliberately defines no cost attribute for the same reason. A user on a six-month-old release would see six-month-old prices with no way to know. Tokens ship; prices are `MODELMRI_PRICES` or nothing.

**Autointerp with a local labeller.** The Delphi detection-score framing is right and the gate-on-score rule is the only honest way to do it, but the economics do not work: it needs an Ollama install, holds a judge model beside the SAE'd model on 8 GB, costs many judge passes per feature, and a 3B labeller writes weak explanations — so the shipped experience is mostly "explanation not supported by its score (0.54, n=40)". That is an L build to render refusals. The corpus evidence page (#13) is the useful half and needs no labeller.

**A hand-written incremental grammar compiler.** A wrong token mask is an invisible failure in a tool whose entire premise is not shipping invisible wrong answers. The visualisation is the valuable half; `lm-format-enforcer` is a pure-Python offline dependency that costs one optional extra.

**Rollout success-rate statistics.** This duplicates Verdikt — a separately published project for robot-policy evaluation statistics with a decision layer. Splitting the same Wilson/Clopper-Pearson/bootstrap machinery across two portfolio projects weakens both and doubles the maintenance. It is also the most crowded slot in robotics (RoboLab, rollout-doctor, TRI STEP, VLA Foundry all inside a year) and it is the one place ModelMRI's actual advantage — the internals — plays no part. Hand it to Verdikt and link the two.

**RLDS / TFRecord reading.** A hand-rolled framing reader plus a `tf.Example` protobuf parser is well beyond L, and the plan skipped masked CRC32C validation. Shipping a reader that silently accepts a corrupt record in the same release as a dataset-integrity auditor is an incoherent posture. HDF5 (#47) is the small, local, actual hobbyist half.

**Action-chunk sampler disagreement.** Sidecar-blocked *and* it refuses on deterministic policies — which includes the ACT/SmolVLA-style checkpoints that are the primary target. A plausible outcome is an M-effort feature that declines to draw anything for the main audience. Its clearest purpose was supplying the reference distribution for the instruction-swap test, which can compute that itself where it applies. The self-check pattern — run twice with two seeds, refuse if bit-identical rather than drawing a flat zero that reads as confidence — is worth reusing elsewhere.

**Head taxonomy without a null.** Proposed separately from #7 as the same detectors with a sort control and no matched non-repeating baseline. That is TransformerLens's bare score, where a 0.3 reads identically whether it is remarkable or ordinary. The null is the entire reason the feature is worth building.

**A second `verify` path.** Two proposals arrived for recipe-in-`.mri` plus `modelmri verify`. One verify path, stamped where the numbers are produced (#17), not reconstructed from an ordered call log — a second source of truth would drift, and a missing recipe field defaulting rather than refusing would recreate the 0.10 flagship bug at the level of shared findings.

**Emitting OTLP.** *(Reversed at the maintainer's request — see #51. The objection below is unchanged and is carried there as the feature's standing caveat.)* The recorder already delivers to ModelMRI. Emitting adds a second delivery path and a maintenance obligation against a semconv that was deprecated out of the main repo in June 2026 with no stable tags, in exchange for nothing the user asked for.

**Writing Inspect logs.** Reading is a zip of JSON and cheap. Writing commits a solo maintainer to tracking an unfrozen schema in both directions forever, and someone with Inspect logs already has Inspect's viewer.