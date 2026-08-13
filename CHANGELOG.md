# Changelog

Notable changes to `modelmri` and `modelmri-record`. Format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/); versioning is
[semantic](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- **#18, completed: `modelmri verify FILE.mri`.** Re-runs the measurements in a
  recording on the machine you run it on and reports, per number, whether it
  came back the same. The other half of #17: a receipt says what produced a
  number, and this is the thing that acts on one.

  **Three verdicts and no pass/fail.** Bit-exact reproduction across two
  machines is not achievable — kernel selection, cuDNN version, TF32 and
  reduction order all move the last digits, and `ablate.py` already records
  measuring 4.863085746765137 against 4.863086102936881 for the identical
  computation. So a number is `reproduced`, `differs`, or `not verifiable`,
  and the last one always says which.

  **Every tolerance is measured, never asserted.** There is not a hardcoded
  epsilon in the module. Each check establishes its own floor by running the
  same computation twice on this machine and taking the spread — the technique
  `ablate.rank_heads` already uses for its noise floor. Attention gets a second
  floor the file supplies itself: `session._quantise` stores each map as uint8
  against that map's own maximum, so its `scale` is the finest difference the
  file can represent, and the per-block tolerance is the larger of the two.

  **It refuses to claim a check it did not run.** A sampled generation is not
  compared, because a different continuation would be the sampler and not the
  model. A file whose dtype or resolved commit differs from this machine's
  blocks every numeric check with a sentence naming both. Attention depends on
  re-establishing the run, so when the generation cannot be reproduced the
  attention check says exactly that rather than reporting a difference it
  cannot attribute; patching runs its own forwards and is checked either way.
  The head ranking is reported as unverifiable rather than skipped — the `.mri`
  records that a ranking ran and does not carry it, and silence would read as
  "it reproduced". Every stored head map is checked, not a sample of one.

  Exit 1 only for a real disagreement. A file this machine cannot check is not
  a broken file, and exiting non-zero for it would make `verify` useless in CI
  the moment somebody ran it on a different accelerator.

- **The generation now carries a receipt of its own**, including `temperature`
  and an explicit `greedy` flag. It is the receipt every other one depends on:
  each names a prompt, and this says how that prompt was answered. Without it
  `verify` cannot tell a model that changed from a sampler that rolled
  differently, and no sampling configuration was recorded anywhere before.

- **#17, completed: receipts on every number.** Each measurement now carries a
  machine-readable record of what produced it — model, resolved HF revision
  sha, dtype, device, attention implementation, seed, tokenizer hash, prompt
  hash, ModelMRI version and the exact request — returned on every measurement
  route and written into the `.mri`.

  Every panel already printed its setup in prose for whoever was looking at the
  screen at the time. None of that survived an export, and none of it could be
  checked by anything. This is what makes `modelmri verify` (#18) possible at
  all: you cannot re-run a measurement whose setup you have to infer.

  **It does not guess.** Three fields can genuinely fail to resolve, and each
  answers `null` with a sentence saying why rather than a plausible default. The
  revision is read from the local cache and never the network, so it works
  air-gapped; `refs/main` is consulted first, and when several revisions are
  cached with no ref to disambiguate them the answer is "naming one would be a
  guess" rather than the newest directory. The tokenizer hash covers the full
  fast-tokenizer definition where there is one and SAYS SO when it could only
  reach the vocabulary — two tokenizers with the same vocabulary and different
  normalisers produce different token ids, so the two hashes must not be
  compared. A `seed` of `null` means the measurement was not seeded, which is
  not seed `0`.

  Receipts carry no filesystem paths and no usernames. `hf_id` is an absolute
  path for a model loaded from a folder, and the `.mri`'s own `model_id` field
  had already shipped that leak once; the reduction happens in `stamp` rather
  than only at export, so it holds for every route and not just the one writing
  a file. Any absolute path inside `request` is reduced the same way — found by
  the leak test rather than by review, after `rank_features` put a local SAE
  directory in its receipt.

  A collapsed one-line "measured by" strip appears under the head ranking,
  feature ranking, patching grid and logit lens, expanding to the full record.
  A measurement whose revision could not be established is marked, because a
  finding that cannot be re-run against the same weights is the single fact
  there most worth noticing.

### Fixed

- **A patch trace could be exported into a `.mri` describing a different
  prompt.** `_patch_for_export` guards on the epoch, and its docstring says the
  guard exists so that "a trace measured on an earlier prompt" is not written
  beside a different run's tokens — but the epoch moves on load and unload and
  deliberately NOT on generation, so the guard never fired for the case it
  describes. Measured: patching "The Eiffel Tower is in the city of", then
  generating "Bananas are yellow because", produced a file whose tokens and
  attention were the bananas and whose patch section was the Eiffel Tower, with
  nothing downstream able to tell. `adopt_step` clears it on the same rebase;
  the generate path was the one rebase that did not.

- **A section navigator, because the page is now nine panels deep.** Every
  panel was reachable only by scrolling past the ones above it. A rail in the
  left gutter lists the sections, marks the one being read, and jumps; ⌘K /
  CtrlK opens a filterable palette from anywhere, and below the width where
  the gutter exists the rail gives way to a single button so it never overlaps
  what it navigates.

  **It reads the page rather than carrying a list of sections.** The set of
  panels here is not a constant — it is decided at runtime by the viewer and
  demo builds, by whether anything has been run yet, by whether the model is
  introspectable, by replay, and by whether an open session carries a graph. A
  hand-written list would offer entries that jump to nothing and would silently
  omit whatever panel is added next; discovering them from the DOM means a new
  panel appears in the rail without anybody editing the navigator. Section
  colour is taken from the `.dot.d-*` class each section already carries, so
  the one place a colour is defined stays the one place.


- **#42, completed: point any OpenTelemetry exporter at ModelMRI.**
  `POST /api/otel/v1/traces` reads an OTLP/HTTP JSON export, so a team that is
  already instrumented gets the agents panel without this project writing an
  integration for their stack:

      OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:5900/api/otel
      OTEL_EXPORTER_OTLP_PROTOCOL=http/json

  **Which vocabulary a span spoke is recorded, never assumed.** There is no
  single spelling for the prompt: OpenLLMetry writes `gen_ai.prompt`,
  OpenInference writes `input.value`, the semantic conventions write
  `gen_ai.input.messages`, the Vercel AI SDK writes `ai.prompt`. Each field is
  tried in order and **the key that matched is stored in `meta.otel_keys`**, so
  a reader can tell which vocabulary a step was read through rather than
  trusting that it was read at all. A span that does not say which semconv
  generation it was written against is recorded as `unstated` and the trace
  carries a note saying so — `gen_ai.*` left the main semconv repo on
  2026-06-12 and the names have churned since, so "I do not know" is a real
  and common answer here, and the honest one.

  An operation with no ModelMRI step kind is filed as `tool_call` with the
  original kept in `meta.otel_operation`, rather than being invented into a
  closer-sounding fit. A span whose end equals its start reads back as unknown
  duration, not as a measured zero. Protobuf bodies are refused with a 415 that
  names the limit and the one-line fix, instead of being mis-parsed into a
  trace that looks real.

### Fixed

- **`modelmri serve` failed on first run for every new user.** `paths.data_dir()`
  answers where the trace database belongs and creates nothing — `paths.ensure`
  is the creator, by design — and none of the three `TraceStore` call sites
  called it. On a machine with no legacy `~/.modelmri`, opening the database
  raised `unable to open database file` inside `create_app`, before the server
  printed its URL. It survived this long because every machine it had been run
  on had already been an older version's machine. `TraceStore` now creates its
  own parent directory, which is the moment of writing the design intends.

- **An OTLP span id is only unique within its trace**, but `step.id` is the
  primary key of the whole table, so two different exports reusing a span id
  collided. Step ids imported from OTLP are namespaced by their trace.

### Added

- **#53, completed: a graph travels in a `.mri` and renders in the viewer.**
  `modelmri open graph.pt` now writes a forwardable `.mri` and serves it in
  the same viewer as every other finding, and `modelmri inspect` reports a
  graph rather than printing an apparently-empty session.

  The section is additive like `patch`, so the format version does not move
  and an older reader ignores the key. One rule is new and specific to it:
  **`provenance.measured_by` is required.** A `.mri` that renders somebody
  else's attributions under ModelMRI's chrome without saying so is exactly
  the confusion the feature exists to prevent, so a graph without it is
  refused — by `session.build` when writing, by `session.parse` when reading,
  by the server route, and by the viewer's own shim. Four copies, because the
  last one runs in the recipient's browser on a file a stranger forwarded.

  The writer is as strict as the reader on purpose. Dropping the section
  instead would hand back a file the caller believes carries a graph and which
  does not, silently — and the reason it was dropped is the one guarantee the
  section makes.

  Bounds like the rest of `parse`: an edge pointing outside the declared node
  count, a non-numeric or non-finite weight, an edge list that is not a list,
  a node count that is not an integer, and more than 50,000 edges are each
  refused with the reason. Indices reach the viewer as array subscripts.

  The panel draws the provenance first and above every number, in the file's
  own words — the disclaimer is rendered from the payload, never composed in
  the UI, because one assembled in a component is one a refactor can drop.
  Edge bars are signed: an edge that suppresses is not a weak edge that
  promotes, and a bar drawn from the absolute value loses that.

  Also: the viewer's served filename now derives from the file instead of
  being fixed at `session.mri`, so the URL says what is open.

- **#53 Open somebody else's circuit-tracer attribution graph.**
  `modelmri open graph.pt` reads one and prints it behind a banner naming the
  file, the tool that produced it and the model it was computed on. Nothing
  outside circuit-tracer's own Neuronpedia flow opens these.

  **The banner is the feature, not chrome.** A graph ModelMRI did not compute
  must never be mistakable for one it did, so provenance is a required field
  of the result — `measured_by` is a sentence inside the payload rather than a
  flag the UI has to remember to interpret, and it prints before any number.

  **Reading a pickle from a stranger.** A `.pt` is a pickle and unpickling
  runs code. The roadmap said `torch.load(weights_only=True)`; measured, that
  refuses a real circuit-tracer graph outright, because `cfg` is a
  `UnifiedConfig` and `logit_targets` are `LogitTarget` objects:

      UnpicklingError: Unsupported global: GLOBAL ...UnifiedConfig was not an
      allowed global by default

  Refusing means never reading the model name the banner needs;
  `weights_only=False` means executing whatever the file says. So the
  unpickler is restricted instead: `find_class` allows torch's tensor-rebuild
  machinery and answers every other class with an inert stub **this module**
  defines, so the named module is never imported and none of its code runs.
  The attributes still arrive, which is how `cfg.model_name` reaches the
  banner without trusting the file.

  Proved with a control. A `__reduce__` payload that writes a file on unpickle
  is neutralised by the reader — and the same payload through plain
  `pickle.loads` fires, so the test cannot pass against a payload that never
  worked.

  **Nothing materialises the matrix.** A graph is nodes x nodes; at 10,000
  nodes that is 400 MB, and `.tolist()` on it is several gigabytes of Python
  floats. `summary()` reduces on the tensor and `edges()` uses `topk` on a
  flattened view, so a 1,000-node graph — a million possible edges — returns
  under 20 kB of JSON. The cap is reported, so a pruned graph is never
  mistaken for a whole one, and a zero edge is dropped rather than reported as
  an edge of no weight.

  Refusals with the same posture as `session.parse`: a ragged (non-square)
  adjacency matrix, a non-tensor one, a wrong number of dimensions, a node
  count above the bound (checked against the shape, before anything squares
  it), a torch file that is not a dict, and a dict missing the required keys —
  which names what the file *does* hold. Non-finite weights are reported and
  not cleaned; an unknown key is noted rather than refused, so a newer
  circuit-tracer still opens and an unread field is never mistaken for an
  absent one.

- **#51 Emit OTLP, version-stamped.** `modelmri export --otlp
  http://localhost:4318` hands a recorded run to whatever the team already
  runs -- Langfuse, Phoenix, Grafana, Honeycomb -- and `modelmri-record` gains
  an opt-in `deliver_otlp=` that is **off by default**.

  This is not an "ahead" feature and is not sold as one. Every competitor in
  the tracing category ingests; a local tool that cannot hand its traces
  onward is a dead end for anyone with an existing collector.

  Three things make it honest rather than merely working.

  **One table, both directions.** `otel.FIELDS` is the single mapping between
  a recorded step and its OTLP attributes, and `to_otlp` and `from_otlp` both
  read it. Two hand-written mappings drift -- one gains a key, the other does
  not, and a column disappears through a round trip while both functions still
  look correct alone. The test round-trips a document and compares over
  `FIELDS` itself, so the table is what is under test.

  **The vocabulary is stamped on every span.** `gen_ai.*` was deprecated out
  of the main semantic-conventions repo on 2026-06-12 into a repo with no
  releases, no tags and nothing marked stable. That objection is real and does
  not go away. What makes it survivable is that every span carries
  `modelmri.semconv.generation` and the CLI prints it, so a consumer can
  always tell which vocabulary a span speaks. The stamp is a date, not a
  version, because there is no released version to cite.

  **JSON only, stdlib only.** OTLP/HTTP with a JSON body over `urllib`. No
  OpenTelemetry SDK, which is what keeps `modelmri-record` importable into
  somebody else's agent -- it still declares `dependencies = []`. A collector
  that accepts protobuf only is refused with a sentence naming the limit
  rather than approximated.

  Absences survive the wire. A missing token count is **omitted**, never sent
  as 0 -- `gen_ai.usage.input_tokens: 0` is a claim that the call used none. A
  step with no recorded duration is the harder case: OTLP requires an end time
  and cannot say "unknown", so it goes as a zero-length span, which on a
  waterfall reads as an instantaneous operation. It is marked
  `modelmri.duration.recorded=false` and the CLI prints how many spans carry
  it, which is the most the format allows.

  Verified against a real HTTP collector, not a mock: valid 32/16-hex ids,
  times as strings (proto3's JSON mapping), parent links preserved, auth
  headers delivered, and every refusal path exercised -- 415 (protobuf),
  unreachable host, a bare `localhost:4317` (that is gRPC), and a malformed
  `--header`.

  `deliver_otlp=` needs `modelmri` importable, and says so if it is not rather
  than silently doing nothing. It runs after the normal delivery and cannot
  affect it: a collector being down must not cost you the trace. It exports
  the **redacted** document -- shipping raw payloads to a third party while
  the local copy is scrubbed would be the redactor working backwards.

- **#36, the other half: what quantisation cost your model's BEHAVIOUR.** The
  weight half shipped last release and answers how far the numbers moved.
  This answers the question people actually have -- whether the model still
  says the same thing -- and the join between them is the point. A tensor can
  move a long way in RMS and change no answer; a tensor can barely move and
  flip the argmax at the one position that mattered.

  Three measurements over one identical token sequence: per-position KL
  between the two next-token distributions (`ablate.kl_nats`, so it is the
  same quantity a head ranking reports), every position where the argmax
  flipped with **both** candidates and both probabilities, and per-layer mean
  attention divergence so damage can be located in depth rather than totalled.

  Measured on `SmolLM2-135M-Instruct-Q4_K_M.gguf` against
  `HuggingFaceTB/SmolLM2-135M-Instruct`, RTX 4060 / bfloat16, "The capital of
  France is":

  | | |
  |---|---|
  | positions | 5 |
  | median KL / max KL | 0.0357 / 0.0641 nats |
  | answers actually changed | **0** |
  | ties broken | 1 |
  | most divergent attention | layer 22 of 30 |

  That split is the feature. A naive report says "1 of 5 tokens changed",
  which sounds like real damage. The one flip sits at a **0.038 margin** --
  the original ranked `,` at 0.322 against ` is` at 0.319 -- so quantisation
  broke a coin-flip, it did not change the model's mind. A flip is called
  CONTESTED when the reference model's own top-1 beat its top-2 by under 0.05,
  and both counts are reported rather than netted into one that has quietly
  decided for the reader. The final answer, ` Paris`, is unchanged.

  **Two models never sit in memory at once.** Load, capture to CPU, tear down,
  load, capture, compare -- and the currently-held model is unloaded first,
  because it is a third. On an 8 GB card the alternative is a comparison that
  only runs when the model fits three times, which excludes every model worth
  comparing.

  Refusals where a number would otherwise be meaningless: different
  tokenisations (a GGUF carries its own tokeniser, and the refusal names the
  first position where they diverge), different vocabulary sizes (a KL over
  different supports is undefined), and the same file on both sides. Missing
  attention is `null` and noted, never 0 -- a zero would read as "identical".

  And it says what it is not: this measures the quantiser through
  HuggingFace's kernels, not llama.cpp's end-to-end damage, which uses its
  own. One prompt is one sample, so the prompt travels with every response and
  the per-position series is returned whole rather than averaged.

- **Run a GGUF through the whole stack, and see what it costs first.** The
  reader could tell you what was inside a GGUF and then nothing could open it.
  `gguf_load.py` dequantises one into an ordinary torch module, so the logit
  lens, the head sweep, the patching grid and attention all work on it.

  The number nobody expects comes with it. A 4-bit GGUF does not load as a
  4-bit model — transformers has no kernels for these types, so it dequantises
  every tensor on the way in. **0.397 GB on disk becomes 1.192 GB of bfloat16
  tensors**, three times the file, because the whole checkpoint is
  materialised as float32 before anything is cast — so asking for bfloat16
  does not avoid the float32 transit.

  The resident figure is arithmetic on the header — `parameters × dtype bytes`
  — and it is exact: 1,192,099,840 bytes predicted, 1,192,099,840 weighed,
  error 0.000000, and the same on SmolLM2-135M. The load report carries that
  comparison rather than asserting it.

  The transit figure is a different kind of number and this entry originally
  blurred them. `parameters × 4` is a *prediction*; the process RSS that
  results is a *measurement*, and they disagree in both directions:

  | | predicted | sampled RSS delta | error |
  |---|---|---|---|
  | Qwen3-0.6B-Q4_K_M | 2.384 GB | 2.30 GB | −3.5% |
  | SmolLM2-135M-Q4_K_M | 0.538 GB | 0.585 GB | +8.6% |

  Opposite signs, so there is no correction factor to fold in — RSS also
  carries the tokeniser and the allocator's own release timing. `Loaded` now
  reports both and their signed error, and `scripts/measure_docs.py --gguf`
  prints them, which it did not before: the measured peak was quoted in prose
  that no command in this repo could reproduce. That is the failure this
  project exists to be about, sitting in the feature about not doing it.

  Everything is still computable before the download from a few hundred
  kilobytes of header, which is the point — a projection good to about ten
  percent is what a refusal needs.

  Worked example of the refusal, on the machine this was written on: Gemma 4
  E2B is 4.63 **billion** raw parameters behind an "E2B" name, so its 2.83 GB
  Q4_0 file wants 9.26 GB resident and an 18.51 GB float32 transit. Against
  16.94 GB of total RAM the answer is "will not fit", and it says *total*
  rather than *free*, because closing other programs cannot change it.

  Refusals by name rather than by stack trace for the things a GGUF repo ships
  beside the model — `mmproj-*` projectors, `mtp-*` speculative heads, split
  `-00001-of-*` shards, architectures transformers has no config for (asked at
  runtime, not hardcoded), and a directory holding several quantisations, which
  is refused rather than guessed at because which file you load is which file
  your measurements describe.

  Every result says so, too: a loaded GGUF is the quantised weights
  dequantised, not the original model, and `quantdiff` measures the gap.

- **A third ablation baseline, and the number that says the baseline is
  deciding.** `ablate.py` has documented since it was written that zero- and
  mean-ablation disagree, and done nothing about it — so every ranking this
  tool has shown was one of several answers with nothing on screen saying the
  others existed. `resample` is the on-distribution third: replace a head with
  what it really computes on a different sentence, eight times, and report the
  median with its spread. Measured on gpt2 layer 0, bf16, "The capital of
  France is", against 8 plain sentences — zero ranks H7, H10, H9; mean ranks
  H1, H8, H2; resample ranks H7, H8, H10. Spearman between pairs runs 0.34 to
  0.47 and the top five disagree on two or three heads in every pair.

  Head 10 in that run scored between **0.0274 and 0.3349** across the eight
  draws, around a median of 0.0355 — a twelvefold spread, so a single donor
  could have reported any number in that range as the head's importance. One
  draw is a coin flip, which is why there are eight, and why the corpus is
  named in every response: the same head scores differently against different
  donors, so a resample number quoted without its corpus cannot be checked.

  Refusals rather than fallbacks throughout — no corpus, a donor shorter than
  the prompt (both lengths named), a donor missing a layer. Padding a short
  donor would score the padding; falling back to `mean` would return a
  different measurement under this one's name.

- **A cost preflight, so an analysis is priced before you pay for it.** One
  probe pass on this machine, multiplied by the pass count the analysis already
  knows. Time multiplies across sequential passes; **peak memory does not** —
  measured on gpt2 over the full 146-pass sweep, the loop's peak was 2.00x one
  pass, not 146x, so multiplying the peak would have refused every analysis this
  tool offers. Projection called 146 passes exactly, 4.90 s against 4.46 s
  actual and 1.1 MB against 1.8 MB. Labelled as one sample, and CPU/MPS report
  what they could not measure rather than a confident zero.

- **A fit calculator that shows its arithmetic and grades itself.** Weight
  bytes from the safetensors header, KV cache as
  `2 x layers x kv_heads x head_dim x seq_len x dtype`, and the eager-attention
  buffer ModelMRI itself forces — every term printed with its formula, and the
  longest context that fits your card by binary search (5,313 tokens for gpt2
  on an 8.6 GB 4060). MLA, sliding-window and hybrid-SSM architectures are
  refused by name rather than approximated; pointed at a real cache it declined
  `gemma-3-270m-it`, which genuinely has a 512-token window.

- **A random-weight control.** The same architecture built from `config.json`
  alone — no weights fetched, works offline — seeded, and run through the
  identical `rank_heads`. Measured on gpt2 layer 0, seed 0: the trained model
  ranks H7 0.898, H10 0.535, H9 0.412 while the untrained twin ranks H3 0.016,
  H0 0.015, H1 0.015. Spearman -0.50, sharing 1 of the top 5, and the twin's
  scores are fifty times smaller and nearly uniform. The ranking survives,
  which is the outcome you want and not the guaranteed one.

- **A telemetry bar, with the cost of being watched broken out.** Tokens/sec
  measured over the streamed tokens, prompt processing kept apart from decode,
  peak allocator memory, and context fullness against the model's real limit.
  Live telemetry is table stakes — TextGen, LM Studio and llama-server all
  have it — so the differentiated line is the other one: **what introspection
  costs.**

  ModelMRI is slower than Ollama for a specific, nameable reason rather than a
  vague one. It forces `attn_implementation="eager"` and asks for
  `output_attentions=True`, which materialises an `n_layers x n_heads x S x S`
  tensor a plain runner never allocates. That figure is computed from the
  shape and shown as its own line, with a warning available *before* a run
  that would not fit rather than an explanation after the allocation fails. At
  4,096 tokens on a 12-layer, 12-head model it is 4.8 GB — larger than the
  weights of most models this runs on.

  Every number is labelled for what it is. Memory reads `allocated by
  PyTorch`, never "VRAM used": the caching allocator's view is not the
  driver's and other processes are invisible to it. The rate travels with the
  prompt length and sequence length it was measured at, because one generation
  is one sample. A cell that could not be measured says so — CPU has no
  allocator to ask, and a 0 in a memory column is a claim that nothing was
  used. `tokenizer.model_max_length` is rejected when it is a sentinel (several
  tokenizers report 1000000000000000019884624838656) rather than turned into a
  confident 0.0% of context.

- **The GGUF reader has a panel now, and the scanner actually lists them.**
  "Click any `.gguf` the scanner already found" was not true: `find_torchscript`
  globbed only `.pt`, `.pth` and `.torchscript`, so the format most people
  running models locally actually have was never in the list at all. It globs
  `*.gguf` too, and `checkpoint_kind` decides by the file's magic bytes rather
  than by extension — a GGUF is not a zip, so the archive logic would have
  called it unreadable.

  Clicking one opens a reader rather than attempting a load: the headline
  numbers, a by-quantisation-type table, the tensors sitting above the
  headline, a sortable and filterable table of all of them, and every metadata
  key behind a disclosure. Read is deliberately a different verb from load —
  transformers cannot run a quantised GGUF, and one button for both would
  promise something that can only refuse.

  Driven in a browser against a real 0.52 GB qwen3-0.6B blob: 311 tensors,
  751,632,384 parameters, 5.499 effective bpw, and the table showing where that
  comes from — Q4_K 4.5 bpw across 155 tensors and 294.0 MB, Q6_K 6.562 across
  15 and 163.8 MB, F16 16 across 28 and 58.7 MB. An unsized tensor renders
  "unknown type", never a dash that could be read as zero.

- **Open a GGUF and read what is inside it.** The scanner has always found
  `.gguf` files — the format most people running models on their own machine
  actually have — and then refused them with a note. It still cannot *run*
  one, but "cannot run" and "cannot tell you anything" are different claims and
  only the first was ever true. Every metadata key, a full tensor table with
  each tensor's ggml type, shape, byte count and file offset, and per-type
  roll-ups.

  **Bits-per-weight is arithmetic, not a label.** Ollama, LM Studio, Jan and
  Open WebUI show a quant preset name and a file size. Measured on this
  machine's own Ollama blobs: a `qwen3-0.6B` whose dominant type is Q4_K reads
  **5.499 bpw effective**, because 164 MB of it sits in Q6_K and 59 MB in F16
  against 294 MB at the 4.5 headline. The tensors above the headline are named
  rather than averaged into it.

  Stdlib only, and deliberately not the `gguf` pip package: nothing here reads
  tensor data, only the length-prefixed table describing it, so there is no
  dependency to add and nothing that a release versioned against llama.cpp can
  break. Reading a 0.82 GB blob's header took 963 ms and the tensor byte counts
  account for 98.2% of the file. An unknown ggml type renders as
  `ggml type N (unknown)` with its size omitted — never bucketed into the
  nearest familiar thing, since a wrong bits-per-weight computed confidently is
  worse than an absent one.

- **Search every recorded step, from a pip install.** Free text plus
  allow-listed filters — `kind:tool_call`, `error:true`, `duration>2000`,
  `name:pytest` — over every trace on the machine. Results are steps rather
  than runs, because what somebody is looking for is the tool call that
  failed, not the hour it happened in, and clicking one opens that run with
  the step selected.

  Backed by SQLite FTS5, which is compiled into essentially every CPython
  build — Langfuse needs ClickHouse for this and Braintrust built a bespoke
  columnar store. "Essentially every" is not "every", so a build without FTS5
  degrades to a substring scan and the response **names the engine that
  answered**, rather than quietly becoming a slower, differently-matching
  feature. Filters are an allow-list of five column names, never string
  interpolation; every value is a bound parameter.

  A filter binds only with no space after the colon, so a pasted log line —
  `error: connection refused`, the single most likely thing anybody types into
  a search box — stays plain text. With loose binding it parsed as
  `error:connection` and was refused. And an unparseable filter is named
  rather than dropped: `error:maybe` silently matching nothing looks exactly
  like a trace with no failures.

- **A recorded agent step opens in the mechanistic panels.** The two halves of
  this tool have sat beside each other doing nothing for one another: a
  timeline of agent steps on one side, attention and ablation and patching on
  the other, and no way to get from a failing step to what the model was doing
  when it produced it. `modelmri_record.instrument_transformers()` now records
  a local `generate()` call's actual token ids, and `adopt_step` re-establishes
  that generation as the current one — so every existing panel reads it
  unchanged, with nothing re-run.

  Demonstrated end to end on gpt2: instrument, generate inside a `trace()`,
  store, reload, adopt, and rank heads on the result — H7, KL 0.898, on a
  sequence nobody typed into the UI.

  This is the one join no hosted platform can build, and the reason is
  structural rather than clever: LangSmith, Langfuse, Phoenix, Braintrust,
  Weave, Opik and Laminar all stop at the API boundary and none of them ever
  holds the weights.

  Four refusals, because every panel downstream reads `last_ids` and none of
  them checks where it came from. A hosted-API step says the weights are not on
  this machine rather than offering a button that can only fail. A step from a
  different model is refused by name — reading one model's ids through
  another's weights produces numbers about nothing and no panel would show
  that it had. Re-tokenising the prompt is checked against the recorded ids and
  a mismatch refuses, naming the likely cause, because adopting near-identical
  ids would point every panel at a sequence the model never saw. And there is
  **no substitute-model path**: replaying a hosted model's prompt through
  whatever happens to be loaded is a machine for confident wrong conclusions,
  however loudly it is labelled.

  `meta` carries ids and numbers only, never prompt or completion text —
  `redact.py` runs over `input` and `output` at delivery and nothing else, so
  text smuggled through `meta` would leave the machine unredacted. One
  consequence worth stating: if redaction rewrites a prompt, the step becomes
  un-adoptable, because re-tokenising the redacted text no longer reproduces
  the recorded ids. That is correct — the model saw the unredacted text and
  this tool should not reconstruct it.

- **Steering for the models that have no SAE, which is almost all of them.**
  Contrastive prompt pairs give a direction without an SAE and without a
  training run — but difference-of-means *always* returns a direction, and
  adding any large vector to a residual stream changes the output, so nothing
  about the result looks different when there was no signal. Every direction is
  therefore scored against its own **label-shuffled null**: refit eight times
  with the labels reassigned, and report a direction that does not beat its
  shuffles as *not measured* rather than as a small finding. Fitted on half the
  pairs and scored on the other half, because a direction scored on its own
  fitting set separates it by construction.

  Measured on gpt2, bf16, twelve sentiment pairs: CAA has 11 of 12 layers beat
  their null (best at layer 9, +3.326 against a null max of 2.185), RepE 11 of
  12 (best at layer 10). Splitting the same 24 sentences at random instead of
  by sentiment: **0 of 12** survive. Layer 0 fails in both, which is correct —
  that is the embedding, before anything has been computed.

  Directions persist with the provenance needed to judge them later — model,
  revision, layer, dtype, hidden size, method, and whether they beat their
  null. Loading one onto a model whose residual stream is a different width is
  refused by name rather than reshaped; loading onto a *different* model of the
  same width warns loudly rather than blocking, because cross-checkpoint
  transfer is a legitimate experiment when the person running it knows that is
  what they are doing.

- **Every logit-lens row reports its own error.** `kl_to_final` is the KL from
  the model's real next-token distribution to that layer's lens distribution.
  On gpt2 with "The Eiffel Tower is located in the city of": layer 0 is 21.58
  nats away reading ' destro', layers 9-11 turn ' Rome' -> ' London' ->
  ' Paris', and 0.96 at layer 11 is the closest the lens gets. Past a stated
  threshold the panel calls the lens unusable instead of rendering a confident
  ranked list that describes nothing.

### Fixed

- **Three more from a second audit of the areas the first one skipped.** The
  first audit named what it had NOT exercised, which is the only reason these
  were findable: `budget.py`'s projection arithmetic, `nullmodel.py`,
  `corpus.py`, the steering null, `spearman`, the lens normed check, and React
  hook ordering.

  - **`nullmodel.teardown` did not free the twin.** `del twin` unbinds the
    function's own parameter; the caller's variable is still a live reference,
    so `gc.collect()` collected nothing and `empty_cache()` had nothing to
    release. Measured on a real gpt2 twin: 255.3 MB allocated, 255.3 MB still
    allocated after teardown returned — while its docstring claimed the memory
    came back immediately. On an 8 GB card that is the difference between the
    next analysis running and refusing. Moves the parameters to CPU first now,
    which frees the CUDA storage however many references survive: 256.2 MB in,
    0.0 MB retained.

  - **The steering gate's false-positive rate is now measured and published.**
    With eight draws the smallest attainable permutation p-value is 1/9, so the
    gate cannot assert better than 0.111 however clean the data. Measured over
    200 trials per method on structureless clouds with no direction in them:
    CAA passes 16.0%, RepE 12.0%, against 50/50 detection of a real 4-sigma
    separation. It is a useful screen and it is not a significance test, and
    that number now appears in the module rather than being left for a user to
    discover. Every direction also reports `p_value` alongside `beats_null`,
    because a boolean hides whether the call was 1/9 or 9/9.

  - **A verdict drawn from nothing.** `nullmodel.verdict` took its
    "mostly the architecture" branch on a high correlation even when no top
    heads had been compared, printing "sharing 0 of the top 0". Unreachable
    from `compare_baselines` as it stands — a ranking short enough to give
    top_k 0 is also too short for Spearman to be defined — so this is a guard
    rather than a live fix.

  Checked and clean, with what was run: `spearman` and `_ranks` against
  scipy over 800 random vectors (exact match on ranks, 5e-5 on rho, and the
  four undefined cases agreeing with scipy's `nan`); `compare_baselines`
  invariants over 300 random cases; the budget projection against real sweeps
  at 1, 3 and 12 layers (pass counts exact, 0.88x-1.02x on time); the lens
  normed check across fp32/bf16/fp16 on CUDA and fp32/bf16 on CPU (correct top
  token and a ~0 floor in all five); and React hook ordering across all four
  panels, checked per component with brace depth rather than by grepping
  returns — the first pass flagged ten violations that were all `useEffect`
  cleanups or a nested component.

- **Nine defects found by an adversarial audit of this branch before it was
  pushed.** Six independent lenses over the diff, every candidate attacked by
  one reviewer trying to reproduce it and one trying to refute it, keeping only
  what survived both. 20 candidates, 10 verified, 9 confirmed. The blocking one:

  - **Trace search returned nothing for every trace you already had.**
    `SELECT count(*) FROM step_fts` does not count index rows — `step_fts` is
    an external-content table, so an unqualified scan reads through to `step`
    and returns the CONTENT count. On any store an earlier version wrote,
    `indexed` therefore equalled `stored`, the backfill never ran, and it never
    ran on any later start either. Search answered `engine: "fts5"` with an
    empty list while a filter-only query (which takes the substring path)
    returned the same traces, so the store visibly contained what the search
    box said did not exist. Now counts `step_fts_docsize` and backfills with
    FTS5's own idempotent `'rebuild'`. Verified on a 40-trace store written in
    the old shape: 0 hits before, 40 after.

  - **Tokens/sec was fabricated and the token count was always one too many.**
    A `TextIteratorStreamer` yields one chunk per token *plus* a final flush
    from `TextStreamer.end()`, and the rate divided that inflated count by a
    window spanning only `n-1` intervals. Measured on gpt2: 8 real tokens
    reported as 9, and at `max_new_tokens=1` a machine doing 31 tok/s reported
    308. The count now comes from `generate`'s own output ids, the rate divides
    by intervals, and a single token reports no rate at all rather than an
    unbounded one.

  - **Trace search promised "newest first" and sorted by offset-within-run.**
    `step.started_ms` is milliseconds since that run's own start, so a step
    nine minutes into last month's run outranked one a second into today's —
    and the LIMIT then dropped today entirely. A full page of stale hits that
    looks complete, which is worse than an empty one. Now orders by
    `trace.started_at`, and every hit carries it.

  - **The GGUF summary averaged the tensors it happened to understand.**
    Element counts are read from `dims` before the ggml type is consulted, so
    they are as known for an unknown type as for F32 — but `parameters` was
    summed over sized tensors only. A 1.44B model whose bulk tensors use a type
    newer than this table (llama.cpp is at 39; shipping gpt-oss GGUFs use
    MXFP4) reported 131,072 parameters, wrong by 11,009x. Parameters now count
    every tensor; byte totals and bits-per-weight are withheld entirely, with
    the reason, when anything could not be sized.

  Five more, each reproduced: `import_trace` retracted from the search index
  *after* `INSERT OR REPLACE` had already cascade-deleted the rows, leaving
  stale terms bound to reused rowids; `adopt_step` cleared every derived cache
  except `_feats`, so the previous generation's SAE activations were served
  against the adopted tokens; the recorder's `result[0]` assumed a bare tensor
  and silently recorded nothing under `return_dict_in_generate=True`;
  `fit.py`'s architecture guards read the top-level config only, so a nested
  multimodal `text_config` escaped the sliding-window refusal and overstated KV
  by 5.8x; and `adopt_step` treated a recorded `n_prompt_tokens` of 0 as
  absent, disabling the id-verification guard it exists for.


- **The logit lens double-normed its final row on every bfloat16 load.** The
  check that decides whether the last hidden state is already normalised
  compared *logits* with `allclose(atol=1e-3, rtol=1e-3)`. Measured on gpt2,
  cuda: in float32 the two vectors differ by 0.00007 and it passed; in bfloat16
  they differ by 0.5 and it failed — but the logits are ~128 and bf16's
  precision there is `128 * 2^-8 = 0.5`, so that IS agreement to the last
  representable digit. The check was reading the dtype, not the model.

  The consequence is the exact failure that block was written to prevent, and
  it had been shipping: the final row read `' the'` where gpt2 actually says
  `' Paris'`, and both `final` and `settled_at` are derived from that row. bf16
  is the default on every current NVIDIA GPU, so this was wrong for most users,
  silently, on the one row a reader can check by eye.

  Now compared as **distributions** rather than logits — softmax is scale-free,
  so bf16 rounding lands near 1e-4 nats while a genuine double-norm measures
  2.12. Found by the new `kl_to_final`: the last row is the model, so its KL is
  an arithmetic floor that has to read ~0, and it read 2.12.

- **`/api/attention/ablate/estimate?baseline=resample` answered 500.** The
  probe built its hook with no donor, so the resample arm indexed `None` inside
  a forward pass. Found by driving the browser, not by a unit test — the
  estimator had only ever been called with the default baseline.

- **A truncated tool output read as a complete one.** `traces._clip` caps
  payloads at 20,000 characters and appends `… [+N]`, and the inspector
  rendered that suffix as if the agent had produced it — so a clipped result
  ended mid-sentence with a bracketed number after it and nothing saying the
  rest existed. Parsed out server-side now and rendered as a marker that names
  how much is missing and why there is a cap at all.

- **`duration_ms` could not say "not recorded".** The column was
  `INTEGER NOT NULL DEFAULT 0`, so a step recorded bare was indistinguishable
  from one that took no measurable time — the same class as the
  `.get(name, 0.0)` above. It is nullable now, absence survives the round
  trip, and the inspector prints "duration not recorded" instead of `0ms`.
  SQLite cannot relax `NOT NULL` with `ALTER TABLE`, so existing stores get a
  real table rebuild.

- **The trace store's lock is now reentrant.** Several helpers touch the
  connection while their caller already holds it, and with a plain `Lock` the
  honest fix — every method takes the lock — self-deadlocks. The alternative
  was a list of remembered exemptions in the test that guards this, which is
  precisely how the 0.10 data race survived review. An `RLock` serialises
  other threads identically and lets the invariant have no exceptions.

- **An empty `t.sqlite` was committed at the repo root.** A trace store left
  behind by a test run; the test itself writes to a temp directory, and nothing
  referenced the root copy. Removed, and `*.sqlite` added to `.gitignore`.

## [0.10.1] — 2026-08-12

### Fixed

- **The model picker's caret was invisible, and the first attempt at that
  enlarged the whole button.** A 12px text glyph in the muted ink, beside a
  12.5px monospace id that out-weighed it — so the one signal that the control
  OPENS something rendered as a smudge. The button's size is restored exactly
  as it was; the caret is drawn rather than typed, at full ink weight, and
  dips on hover. An accent edge makes the control findable without making it
  bigger.


## [0.10.0] — 2026-08-12

### Added

- **Activation patching: where in the model the answer gets decided.** Every
  other ranking here removes something from one prompt. `POST /api/patch`
  takes *two* prompts that differ in one fact, moves an activation from the run
  that knows the answer into the run that does not, at every (layer, position),
  and reports the share of the gap between the two answers that comes back.
  Ablation says "this mattered"; this says "the fact is here".

  Measured on gpt2 float32 with "The Eiffel Tower is located in the city of"
  against "The Colosseum is located in the city of": the shared first token
  scores exactly 0.000 at every layer, the subject tokens carry 0.2-0.44 in
  early-middle layers, and the final token climbs from 0.005 to **0.844** by
  layer 11 — the information moving to where the prediction is made. 350
  forward passes in 9.66 s for a 12x11 grid.

  **The score is signed, so it is not KL** — the one ranking in this tool that
  does not report nats. Patching has a direction and a patch can push the
  answer further away: 5 of 132 sites did, worst -0.157, and an unsigned metric
  cannot tell those from a site that recovered nothing. The two also disagree
  about the ranking, overlapping on 5 of 8.

  **Controls are eight draws, not one.** At a single site the same-norm random
  draws ran from -2.038 to +0.616 against a real recovery of +0.427, so the
  gate moves from 76 of 132 sites on one draw to 20 on all eight. Of the 8
  highest-recovery sites, 3 clear both controls and 5 do not — including +0.435
  at layer 3. Each site also gets a shifted-position control, which asks
  whether it is that position or merely that layer.

  **Most casually-written pairs are refused, and both failures are invisible
  without being told.** The two prompts must tokenize to the same length (2 of
  8 natural minimal pairs did not) and must predict different tokens (2 of 3
  did not, making the denominator exactly 0.000000). Both refusals name what to
  change and print both tokenizations.

  The panel draws the grid with a **diverging** scale — the only heatmap here
  that needs one, because a patch can push the answer further from the clean
  run than doing nothing would. Cells that were tested against chance are
  ringed, and hovering one says whether it beat its own controls or is not
  distinguished from an edit of that size at that layer. The demo refuses it
  rather than baking a grid: the answer depends on the two prompts the reader
  types, so a recorded one would be a fabricated measurement wearing their own
  words.


- **A `.mri` carries the causal result, and `modelmri inspect` reads one
  without a browser.** The format held attention, the logit lens, the prompt
  and the generation — so the one finding in this tool that is *causal* rather
  than correlational, "the answer is decided at layer 15, position 4", was the
  one finding you could not send anybody. Open a recording that carries a
  trace and the patching panel draws it, marked as recorded rather than
  measured on your machine, with the pair it was measured on: a grid without
  its prompts is unreadable. A recording that carries none says so instead of
  offering a button that can only refuse.

  Additive, so the format version does not move and files written before this
  still open. The section is validated like `attention` rather than trusted —
  a `.mri` is meant to be forwarded, so `parse` runs on bytes a stranger sent,
  and a ragged grid, a string where a number belongs, an infinity or a
  40,000 x 40,000 claim all stop there rather than in whoever's browser opened
  it. Malformed is **refused, not dropped**: a damaged file presented as an
  intact one that simply has no patching is the failure that module exists to
  avoid. A 12x8 trace over three components adds under 4 KB.

  `modelmri inspect file.mri` prints the model, the machine it ran on, the
  shape, what was captured, the patching components and the prompt, or
  `--json` for the lot untruncated. Held to the same rule as `open` — no
  torch, no transformers, no server — because 26 seconds of imports to read a
  54 KB file reads as a hang and somebody pressed ctrl-c. Measured at 0.185 s.

- **A download you can watch.** `POST /api/ollama/pull` consumed the daemon's
  progress stream with a loop whose entire body was `last = update`, so a nine
  gigabyte pull showed the word "Pulling…" and nothing else until it finished
  — while exact `completed`/`total` counts arrived the whole time. The data was
  there; nobody published it. The picker now shows bytes, a bar and a time,
  driven against the real daemon: 10.8 to 187.5 of 522.6 MB with the estimate
  converging 166 s to 90 s.

  The estimate is the **average** rate, not an instantaneous one. hf_xet writes
  blobs in large infrequent jumps — 71.6 seconds of silence measured during a
  perfectly healthy download — so an instantaneous figure swings between "12
  seconds" and "four hours" on one transfer. It is withheld entirely until
  there is something to divide: no total, nothing transferred, or under two
  seconds of history. A countdown that starts wrong is worse than one that
  starts late.

  A pull gets its **own** progress slot. Sharing the model loader's was tried
  and reproduced this project's oldest bug: mid-pull of `gemma3:1b`, a page
  that loaded gpt2 made `/api/model/progress` answer with gpt2's name against
  gemma3's byte counts, the pull still running and its updates silently
  dropped. The bar also follows the server rather than the tab that clicked,
  so refreshing the page to check on a download no longer makes it look
  stopped.

- **Every camera in a LeRobot dataset, not the first one.** `next(...)` kept
  one video feature and discarded the rest, so a two-camera SO-100 recording
  or a four-camera ALOHA one was presented as though it had a single view —
  the others were not merely unselectable, they were invisible. All are
  listed, the panel offers a picker when there is a choice, and switching
  re-reads the episode table because the routing is stored per camera.

- **`MODELMRI_MODELS_HOME`** moves where downloaded models land, for anyone
  who does not want them wherever an ambient `HF_HOME` points. Opt-in, and
  only opt-in: an `HF_HOME` somebody configured is theirs, and silently
  relocating a cache shared with transformers and datasets would strand every
  model they have. When it is set, the previous location joins the discovery
  roots, so models already downloaded stay visible rather than appearing to
  have vanished.

- **`modelmri models` and `modelmri traces`.** Two questions people were
  starting a server and opening a browser tab to answer. `models` lists what
  is on this machine **and what will not load, with the reason** — 17 found
  and 6 loadable here, the eleven refusals each naming themselves ("a
  vision-language model, not a causal LM", "no architecture in config.json").
  `traces` lists recorded agent runs, newest first, which is what you want the
  moment `record_demo.py` finishes in another terminal. Both follow the rule
  `open` and `inspect` follow — no torch, no transformers, no server — and
  import in 0.082s. The README now opens with a table of all nine commands.

- **The scanner finds the checkpoints people actually have.** It matched
  exactly one standalone extension, `.gguf`, so a folder holding
  `weights.pth`, `scripted.pt`, `model.onnx` and `epoch3.ckpt` returned **zero
  results** — somebody who trained their own model and pointed this tool at it
  was told, in effect, that it was not there. It now covers `.pt`, `.pth`,
  `.onnx`, `.ckpt`, `.h5`, `.msgpack` and `.pkl`, each with the truthful
  reason it will or will not open, and `loadable: false` wherever the loader
  would refuse. Finding a file is not the same as claiming to support it.

- **A state_dict refusal that looks for the missing half first.** A `.pth` is
  weights with no architecture and cannot be loaded alone — but "write an
  adapter" is a poor answer when the model class is sitting in the same
  folder, which is how people lay a project out. The refusal now reads the
  neighbouring `.py` files, names the `nn.Module` subclass it finds, and
  writes the six-line adapter using *that class, that filename, those
  weights*. Found with `ast`, never by importing: reading a stranger's file to
  see what is in it must not mean executing it.

- **A hero you can touch.** A click drops a travelling ripple, the cursor
  leaves a wake, and a slow diagonal sweep crosses on its own. The
  composition is still seeded from the loaded checkpoint, so every model keeps
  its own stable field.

### Changed

- **Every panel spends the colour it already owns.** Six semantic hues were
  defined and almost nothing used them: each panel carried its accent in a 7px
  dot and a 10.5px caption on a white rectangle. Panels now take that accent
  as a 4px band across the top edge, the measurement rule takes it too, the
  dot is a lit indicator rather than a printed square, and section titles went
  from 10.5px — smaller than the body text beneath them — to 13px. Elevation
  went from 4.5% opacity, which on a cream ground is arithmetically present
  and visually nothing, to something that reads as a plate resting on paper.
  Panels reveal as you scroll to them rather than in a 240ms flash that is
  over before anybody looks, and the ground carries a 40px grid at the same
  pitch as the rules' major ticks.

- **The model picker was 38px tall at 12.5px type** — smaller than the prompt
  under it and quieter than the Generate button above it, for the control
  every panel on the page depends on. 52px at 15px, with an accent edge.

- **The base-model caveat arrives before the generation, not under it.** It
  rendered beneath the output — after the reader had typed a question, waited,
  and read something confidently wrong. Asking gpt2 "whats 2+2" and being told
  to respect each other is not a bug report anybody files as "I used a base
  model". It fires on `instruct === false` only, never on `null`: unknown is a
  third state and announcing "this is a base model" when the tool does not
  know would be a false claim made confidently.

- **The agents panel says what it is.** It said "no traces yet" and gave a
  command, which reads as an empty list of something you already have. It is
  not: that panel has nothing to do with the model loaded above it, and no
  model will ever fill it. It records runs of your own agent code.

- **The lint gate is written down instead of inherited.** There was no
  `[tool.ruff]` section anywhere, so `uv run ruff check .` in CI enforced
  whatever the pinned ruff happened to *default* to — and that default is not
  stable. Measured on one unchanged tree: ruff 0.15.20 (what `uv.lock`
  resolves) reports **0 findings**; ruff 0.16.2 reports **159**. The next
  `uv lock --upgrade`, for any dependency, would have turned `main` red with
  159 findings belonging to nobody's change.

  `select` now names every rule, and `select` replaces the default rather than
  adding to it, so the list is the whole gate on any version. Both installed
  versions resolve it to **the same 160 rules** — settings dumped from each and
  diffed, empty in both directions — and both pass `ruff check .` and
  `ruff format --check .`.

  Every family left out is recorded in the file with its reason and its count,
  so enabling one later is a decision someone makes rather than one a release
  makes for them. `BLE001` is out because **26 of its 63 sites are false
  positives by ruff's own exemption**: ruff does not flag a blind except whose
  body logs the exception, only one that calls a helper which logs it, and this
  codebase deliberately factored that logging into `_internal`. `S110` is in,
  because it makes the opposite claim — that the handler records nothing — for
  the price of 4 directives. 28 `# noqa` directives that had never suppressed
  anything were removed, each keeping its explanatory prose.

  One behaviour change came with it, flagged rather than buried: three `zip()`
  calls gained `strict=True`. Their operands are equal-length by construction,
  and `zip`'s default was to truncate — which in `hub.py` would have marked
  gated models unusable on no evidence, silently.

- **The workbench is a panel, and resting panels use their width.** The model
  picker, the prompt and Generate were a bare fragment, so each became a direct
  child of the page grid and inherited the gap meant to separate *instruments*
  — three of those stacked inside one logical group. It was the region a reader
  uses first and the only one on the page with no card, no header and no
  grouping, sitting above five panels that had all three.

  `max-width: 46ch` on a resting panel is the right measure for prose and the
  wrong one for a panel: three fifths of every resting card sat empty, and
  because it clamped the control rows too, the folder input was too narrow to
  show its own placeholder and a Windows path broke across four lines. The
  measure belongs to the prose now, and the space beside it carries a diagram
  of the *shape* of that panel's output — unlabelled and carrying no numbers on
  purpose, because a decorative chart holding invented data is the worst thing
  this project could put on screen.

### Fixed

- **`modelmri-record` 0.1.3 shipped the trace-parking leak, and the version
  never moved.** The published artefact had no `_undelivered_dir`, did not read
  `MODELMRI_TRACE_DIR`, and parked traces at a bare `modelmri-traces` path
  relative to the working directory — which for somebody instrumenting an
  agent is normally their git repo. Untracked JSON of their conversations, one
  `git add -A` from a public remote. Because the fix landed in source under the
  same version number, `pip install -U` was a no-op for everyone holding the
  leaky build and nobody could tell which one they had. **0.1.4 is published
  and 0.1.3 is yanked.** This is also why the credential-redaction test looked
  order-dependent for a day and a half: it was reading the stale installed
  copy whenever the repo's was not already on `sys.path`.

- **A data race in the trace store.** One sqlite3 connection shared across
  threads with `check_same_thread=False`, which Python permits and does not
  make safe. Every writer serialised it; both readers did not — so a read
  arriving beside any other statement got rows of the wrong width, surfacing
  as an intermittent 500 from `/api/traces`. Before the agents panel had a
  retry, one of those left it empty for the session, indistinguishable from
  "you have not recorded anything". `get_trace` was worse and never raised at
  all: two statements read together, so an interleaving pairs one trace's
  header with another trace's steps.

- **The agents panel fetched once and could never fetch again.** No polling,
  no revalidation, and from the empty branch no reachable path to `setList` —
  so it told you to run `record_demo.py` and then could not display the result
  without a full page reload. It polls briefly and refetches on window focus,
  which is the actual gesture.

- **A `.pth` was filed under a heading reading TORCHSCRIPT.** Those fail in
  completely different ways, and grouping by extension decides nothing —
  `.pt` and `.pth` are the same zip container. The kind is read from the
  archive *index* now (TorchScript writes `constants.pkl` and a `code/` tree;
  `torch.save` writes `data.pkl`), which executes nothing.

- **TorchScript answered 500 instead of refusing.** PyTorch installs a
  generated `fail()` over both hook APIs on `RecursiveScriptModule`, so every
  TorchScript archive on disk is un-instrumentable — all of them, not a corner
  case. The raise sat above the try/finally as a bare `RuntimeError`, so the
  reader got "Something inside ModelMRI failed rather than refusing" for a
  format that simply cannot carry what the panel measures.

- **The custom panel listed our own template, and duplicated yours.**
  `examples/adapter_template.py` ships inside the package as something to
  copy, and it was listed beside real models — once per agent worktree, so
  four rows of a blank form above the one file that mattered. A checkpoint
  under two overlapping scan roots was listed twice. And a module that runs
  more than once in one forward pass — a shared encoder applied to two inputs
  — printed its name twice with nothing to tell the readings apart; rows now
  say `1/2`.

- **The sign-out button was offered for credentials this tool does not own.**
  `sign_out` deletes ModelMRI's own token and nothing else, which is right —
  but `whoami` falls back to `HF_TOKEN` and to huggingface-cli's file, so the
  server truthfully answered `signed_in: true` and the click looked dead. It
  now appears only for a token ModelMRI stored, and otherwise names the owner
  and how to remove it.

- **The keyboard focus ring was a rectangle around rounded controls.** The
  rule said `border-radius: inherit`, meaning "keep the shape you have", and
  `inherit` takes the *parent's* radius — so every control in a flex row had
  its own overwritten with 0.

- **The robot panel showed episode 0's video for every one of 206 episodes.**
  Measured on `lerobot/pusht`: episodes 0, 5 and 20 returned byte-identical
  images, while the state vector printed underneath each one was correctly
  that episode's. The picture and the numbers disagreed and nothing said so —
  and the attention heatmap was then computed on a frame the reader was not
  looking at.

  `episodes()` read `video_from_timestamp`. No LeRobot v3.0 dataset has a
  column by that name — it is `videos/<camera>/from_timestamp`, namespaced per
  camera — and `.get(name, 0.0)` turned the miss into a start time of zero for
  all of them. It reads the real column now and **refuses** when it is absent,
  because a missing routing column means frames cannot be located and
  defaulting is precisely what made this invisible.

  Two more assumptions of the same shape were underneath. The video file was
  `sorted(rglob("*.mp4"))[0]` — the first mp4 anywhere in the snapshot — so
  with two cameras it was whichever key sorts first (the panel could show the
  wrist view while labelling it the overhead one) and with two chunks every
  episode past the first decoded from the wrong file entirely. The row offset
  summed earlier episodes' lengths where the dataset states it outright in
  `dataset_from_index`. Verified across all 206: the stated offsets and the
  summed ones agree, and each video span matches `length / fps` to within
  0.15 s.

- **Six places published a library's own text to the browser.** `Refusal` and
  `BadRequest` carry sentences authored for the reader and are relayed
  deliberately; everything else is a library talking about a machine, and
  library text routinely carries absolute paths and site-packages frames. None
  of the six was on a route anybody had hardened.

  `POST /api/model/load` correctly answers a fixed sentence at 500 — and the
  same failure was written into the progress snapshot, which
  `GET /api/model/progress` returns verbatim at 200 once a second because the
  load meter polls it. Measured: a torch OOM put an absolute weights path and a
  `module.py` frame into that body. `/api/ollama/resolve` returns its error as
  *data* on a 200, so no except arm sanitises it, and an SSL failure published
  the CA bundle's path. `ollama._unreachable` interpolated `err.reason` whole,
  under a docstring arguing a reason "is an errno sentence, never a path from
  this machine" — true of the case it considered, and there was a second one.
  `custom.py` pasted torch's text into the checkpoint reader's refusal, and
  `attribute.py` reported a failed mask check as the exception's full text.

  All six now name the exception's **class** and log the rest, which is the
  rule this codebase already stated one arm away: "only their exception
  CLASSES are interpolated (never their text)". The boundary is written down —
  the reader's *own* adapter code failing is not a leak, it is the entire
  content of "why did my adapter not work", and those four messages still
  relay in full.

- **The focus ring was a hard rectangle around rounded controls.** The rule
  said `border-radius: inherit`, meaning "keep the shape you already have",
  and `inherit` does not mean that — it takes the *parent's* radius, so every
  control in a plain flex row had its own overwritten with 0. `outline` has
  followed the element's own radius since Chrome 94 / Firefox 88 / Safari
  16.4, so the correct value there is none at all.

- **146 of pusht's 206 episodes were unreachable.** The episode dropdown was
  `.slice(0, 60)` and said nothing about it — the same shape as reading only
  the first parquet shard. State and action were printed with `toFixed(0)`,
  which reads fine on pusht (pixel coordinates in the hundreds) and prints
  every value of a normalised dataset as `0`; precision now follows the
  vector's magnitude, chosen once per vector so two axes of one measurement do
  not appear in two formats. Two strings hardcoded "SmolVLA" in a panel whose
  whole point is that any checkpoint with a vision tower works.

- **The load meter reported 5.0 GB of a 2.5 GB model, and a load that stopped
  returning took every later load down with it.** Reported from the app while
  loading `meta-llama/Llama-3.2-1B-Instruct`. Four separate defects, each
  measured before and after:

  - **199.7% on a fully cached model.** The total counted the repo's top-level
    files; the on-disk figure walked the whole tree. Llama-3.2 ships
    `original/consolidated.00.pth` beside `model.safetensors`, both 2.472 GB —
    the same weights in Meta's format, which `from_pretrained` never opens — so
    a complete cache measured 4.955 GB against an expected 2.481 GB. The Hub's
    file list now picks one set of names and **both** sides count exactly that
    set: 2.481 / 2.481 GB, 100.0%. The same rule now sizes the model picker,
    which had been listing that model at 4.96 GB.

  - **The wrong model's numbers under the right model's name.** Watcher threads
    write into one shared snapshot, so a previous load's watcher kept
    publishing after the next load began, and the browser labelled the bar with
    the model the *picker* was showing rather than the one the server was
    loading. Both now name the load that is actually running.

  - **A second load queued behind a wedged one forever.** `load()` took the
    runtime lock with no timeout, so when one load stopped returning, every
    request after it blocked with no message — "nothing loads any more". A
    second load is now a **409** that names what is holding the slot and for
    how long, rather than a thread that never comes back.

  - **A healthy download called stalled.** The 45 s threshold predates
    `hf_xet`, which huggingface_hub 1.x installs by default and which writes
    the blob in large, infrequent jumps. Watching a healthy 324 MB download of
    `EleutherAI/pythia-160m`, the blob sat unchanged for **71.6 s**, then again
    for 59 s. Now 180 s, and the message is scoped to the download stage.

  Added with them: a load that goes quiet in any *other* stage is now
  diagnosed rather than shown as a full bar forever. Measured on the real one —
  `.to(cuda)` never returned, 0.3 CPU-seconds and 0 bytes read over 12 s, while
  the same file read at 295 MB/s and the same GPU took host-to-device copies at
  1266 MB/s from another process — so "no CPU and no bytes" is the evidence the
  meter now reports, and it says stopped rather than slow. The Hub call that
  supplies the denominator also gained a timeout (it was unbounded, and cost
  1502 ms on a model needing no network at all) and now honours
  `HF_HUB_OFFLINE`.

### Added

- **The features panel can now say which features actually changed the
  answer.** It ranked by raw activation, which is what fired, not what
  mattered — the same gap `ablate.py` closed for heads and `attribute.py`
  closed for prompt tokens. `GET /api/features/ablate` removes one feature's
  contribution from the residual stream, runs the model again, and reports how
  far the next-token distribution moved, in the same KL nats every other
  ranking in the tool uses.

  Measured on gpt2 `blocks.8.hook_resid_pre` (SAE
  jbloom/GPT2-Small-SAEs-Reformatted, calibrated `centered+b_dec`), prompt "The
  Eiffel Tower is located in the city of", attributing at token 10 where the
  top token is " Paris" at p=0.06378: the top-8 by activation and the top-8 by
  causal effect share **6 of 8** at that token, and **3 of 8** when each
  feature is removed everywhere it fires across the prompt — because four of
  that ranking's top eight fire only at earlier tokens and reach the answer
  through attention, which the bar chart cannot show at all.

  The scores do not add up, and they miss in the opposite direction from heads:
  the 43 singles sum to 0.66446 while one joint ablation removing all 43 gives
  2.135221, so they **under**-count by 3.2x, where head ablation on gpt2 layer
  0 over-counts 8x. The panel reads the direction off each run rather than
  remembering one.

  It refuses in float32-only. In bfloat16 — which ModelMRI selects for every
  NVIDIA GPU — an edit whose true effect is 4.9e-07 nats reads 0.02836 and
  outranks a feature with 100x its activation, while writing the stream back
  unchanged is still bit-exact and still scores 0.0, so the noise floor cannot
  catch it. The panel now shows that refusal **instead of** the button rather
  than behind it.

### Fixed — in the ranking, before it shipped

Three adversarial passes over the above. Every number below was re-measured
here before anything was changed.

- **The mechanism check was reporting a property 38 rows out of 43
  contradicted.** It re-encoded the edited stream, found the top feature's
  activation was exactly 0.0 of 35.546, and returned `removal_verified: true`
  with a note saying this is "a property of the edit and the SAE, not of each
  feature". Run on every candidate instead of one, it fails on **38 of 43**,
  with the SAE's encoder still reading 10.1% to 60.3% of the original
  activation — and the five that pass do so because relu clamped an
  *overshoot* (feature 5856's pre-activation goes 35.546 to −2.331 for an
  activation of 35.546), not because the removal was clean. The cause is that
  encoder and decoder directions are not dual: `W_enc[:,f] · W_dec[f]` has mean
  0.8387 over d_sae, from −0.3819 to 1.3072.

  `removal_verified` now makes exactly one claim, and it is the one that *is* a
  property of the edit: the stream the model received differs from
  `x − activation × W_dec[f]` by at most `edit_deviation`, measured 0.0 in
  float32. What the SAE still reads afterwards is per row, in
  `encoder_residual`, and costs no forward pass — the re-encode of the cast
  stream agrees with one taken through the model to 3.6e-06.

- **A score was partly the size of the edit, and nothing said so.** A random
  Gaussian direction of the top feature's norm (35.5), subtracted at the same
  token, costs 0.0666–0.1093 nats over five draws against that feature's own
  0.417461. Every scored row is now paired with its own same-norm control
  (`control_kl`, `clears_control`), which is the second forward pass per row
  and why the cost is now `2 × tested + 6`. Measured: **34 of 43** rows clear
  their own control, and two that do not — #22852 and #1288 — are 5th and 6th
  in the bar chart the panel plots. **That 34 is one draw's verdict**, and it
  is corrected below rather than left standing as a property of the features.

- **"34 of 43 rows clear their own control" was one sample, not a finding.**
  The control is a single seeded Gaussian draw per row, and re-measured with 8
  draws per row on the same setup — draw 0 reproducing every shipped `kl` and
  `control_kl` exactly — the same 43 rows give **34 clearing one draw, 21 the
  95th percentile of 8, and 20 all 8**. 21 of the 43 fall between their own
  smallest and largest draw, 14 are called differently by one draw than by all
  8, and 7 of the 9 that fail would have passed on some other draw. The count
  is not device-stable either: 34 on cpu/float32, 36 on cuda/float32 consuming
  the identical draws. `clears_control` is still one draw — paying for 8 costs
  `9 × tested + 6`, 4.27× — but nothing now reads it as more than that: the
  panel says "beat the one random direction of the same size each was given",
  the row tooltip says the control moves, and `control_means` carries the
  8-draw counts.

- **The reconstruction baseline was measured at one token while the edits
  landed at eleven.** At `scope=prompt` the ranking edits every token where a
  tested feature fires, but the "what the decomposition gets wrong" line was
  substituted at the attributed token alone: 0.077530 nats, against 0.221217
  over the window actually edited — 2.85x. Against the first, 2 of 43 features
  clear; against the second, 1 of 256, so a feature was shown as clearing the
  SAE's own error when it did not. The baseline now follows the scope, and
  `residual_share` reports the worst per-position share over that window
  (0.4253 at token 3, against 0.2036 at the attributed one), with
  `residual_share_at_position` kept beside it.

- **The reconstruction caveat compared a squared fraction with a norm
  fraction.** "aggregate FVU 0.0012 — but at this token the SAE fails to model
  20.4% of the stream's norm" put 0.000984 next to 0.203571 under a "but",
  which reads as three orders of magnitude. Like for like it is **7x**: the
  calibration's `rel_err` (0.029397) is the aggregate in the same units, and it
  is now the one the sentence quotes and the one the calibration banner shows.

- **The panel called features "not tested" that it had tested and scored.** At
  `scope=prompt` the client sent no `top_k`, so the server trimmed 256 scored
  rows to 64 while `n_tested` stayed 256, and every plotted feature below
  causal rank 64 got a chip reading "not tested" with a tooltip saying it was
  "not asked, not found unimportant" — the exact inversion of the truth.
  Measured: #18994 scored 0.00031514 at causal rank 72 and was labelled
  untested. The client now requests every scored row, the server's `n_scored`,
  `n_returned` and `rows_note` reach the UI, and the "not tested" wording is
  gated on the response actually carrying every row it scored. The tail count
  under the leaderboard is counted off `n_scored` too — it read "54 more were
  tested and scored lower" a few pixels above the server's own "256 of 494
  firing features were tested".

- **Two docstring claims that were false.** The edit does not leave the rest of
  the decomposition alone: removing feature 5856 drives 33 of the other 42
  firing features to exactly zero, starts 2 silent ones, moves 42.4943 of
  activation outside the target against the 35.546 it removed inside it, and
  grows the unmodelled remainder at that token from 21.3036 to 31.8553. And
  subtracting in the raw and the centered space are not the same subtraction —
  `activation × W_dec[f]` has a non-zero d_model mean (−0.0904 for 5856, 7.05%
  of the edit's norm). The conclusion the second one supported is still right,
  for the other reason: holding `mu` at the value the decomposition was taken
  with is what makes this edit identical to "zero the feature, decode, re-add
  the error".

- **`Reconstruction checked.`** `usable` is `fvu < 1.0`, which only rules out
  an SAE carrying less than a constant vector would — an SAE at FVU 0.85 gets
  the same word as one at 0.001. Verified by scaling a real SAE's decoder until
  it landed under the gate: FVU 0.8482, `usable` true, a fully plotted bar
  chart, and a ranking whose top score (0.00569) is 163x smaller than its own
  reconstruction error's cost (0.9289). The banner now says
  "Reconstruction **measured**", prints the norm fraction beside the variance
  fraction (86.3% for that SAE), and the ranking leads with a red line when not
  one of its scores clears the SAE's own error.

- **`/api/features/ablate` was missing from the error-taxonomy regression
  list**, so nothing would have noticed the 500 arm being widened back to a 409
  carrying torch's own words. Added. `tests/demo_check.py` counted the route as
  "covered" because it falls inside demo.ts's `/api/features/` prefix handler,
  which answers 200 with a *single feature's detail payload* — a fabricated
  ranking rendered as a measurement, and the one failure this project cannot
  ship. It is now exempted with that reason and pinned by a build-artifact
  check, alongside the token-ranking one. The per-row KL annotation is hoisted
  behind the same folded constants as the control it belongs to, which drops
  1,753 bytes of unreachable JavaScript from the demo bundle.

## [0.9.0] — 2026-08-10

The release where several things that were already on screen turned out not to
be true, and the tool learned to say so. If you read one entry, read the sparse
autoencoder one: the features panel was plotting the wrong features.

### Fixed — things that were wrong on screen

- **The SAE was reading activations it was never trained on, so the features
  panel plotted the wrong features.** Two causes, and only one was suspected.
  `saes.py` read `cfg.get("apply_b_dec_to_input", False)`, and the default
  release's `cfg.json` has no such key — so an SAE whose training forward always
  ran `sae_in = x - b_dec` was treated as one that had declined it. Larger, and
  declared nowhere at all: SAELens SAEs are trained on TransformerLens
  activations, which are centered along `d_model`, while HuggingFace's residual
  stream is not — measured `d_model` mean 0.507 on gpt2 layer 8. Feeding an
  uncentered stream to an SAE trained on a centered one does not error; it
  returns features, in the right shape, with plausible magnitudes, for a vector
  the SAE never saw.

  Measured on gpt2 `blocks.8.hook_resid_pre`, prompt "The Eiffel Tower is
  located in the city of", 11 tokens, float32, fraction of variance unexplained:

  | input convention | L0 | FVU |
  |---|---|---|
  | raw, no b_dec — what 0.8.4 shipped | 7491.5 | 13579.24 |
  | raw + b_dec | 2745.4 | 12908.35 |
  | centered, no b_dec | 1344.0 | 0.4219 |
  | centered + b_dec | 60.5 | 0.0010 |

  The top-8 features on the last token — what the violet bar chart draws —
  overlapped the correct top-8 **two of eight**.

  The fix is not a new default, because "always center" would break Gemma Scope
  SAEs, which are trained on raw HuggingFace activations, exactly as badly in
  the other direction. `SAEHandle.calibrate` now runs all four conventions
  against the model the SAE is attached to and keeps the one that reconstructs.
  An SAE is a checkable claim: encode, decode, see how much variance comes back.
  The panel states which convention won, that it was measured rather than read
  from a config, what the config declared, the FVU and the features per token —
  and refuses to plot at all when nothing reconstructs (FVU >= 1 carries less of
  the activation than a constant would).

- **The hosted demo published the machine that baked it.** Reported from a
  phone: the page said it was running CUDA on an RTX 4060 and listed 17
  HuggingFace repositories as "cached, loads offline" on a laptop the visitor
  had never touched. `accelerator`, `discovered`, five `device: "cuda:0"`
  fields and a duplicated sample trace are now synthesised from the recording.
  No model files were ever uploaded; what leaked was metadata. The privacy
  check had missed all of it because it scanned for identifier *shapes* — home
  paths, usernames, drive letters — and a GPU model name is none of those; it
  now asserts the positive form instead.

- **The logit lens served a live model inside a recording.** `/api/lens` is the
  only replay-sensitive route whose guard was not in `runtime.py`, so it never
  passed one. `model is None` covered for it only while nothing was loaded:
  open a `.mri` with your own model still resident and the lens reported the
  live model's layers under a pill reading "recorded, not live".

- **`attention_meta` had one answer for two opposite instructions.** "No model
  loaded" and "nothing generated yet" both arrived as a bare
  `{"available": false}`, so the panel could not tell you to pick a model or to
  press the button in front of you.

- **The documented steering example demonstrated a feature you could not have
  found.** Under the corrected encoding, feature #974 has activation 0.0000 and
  rank 994 of 24,576 — it does not fire, so nobody following "click a feature
  that fired" would reach it. The output still reproduces, because steering
  uses `W_dec` and the calibration bug never touched it. Replaced with #5856,
  the top-firing feature on that token, which flips Paris to London at -40.

- An unclosed `cfg.json` handle in `saes.py`, and `session.py`'s `v != v` NaN
  idiom replaced with `math.isfinite` — same answer, one call, and it no longer
  reads as a typo.

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

[Unreleased]: https://github.com/muhammadmahadazher/ModelMRI/compare/v0.9.0...HEAD
[0.9.0]: https://github.com/muhammadmahadazher/ModelMRI/compare/v0.8.4...v0.9.0
[0.5.0]: https://github.com/muhammadmahadazher/ModelMRI/compare/v0.4.0...v0.5.0
[0.4.0]: https://github.com/muhammadmahadazher/ModelMRI/compare/v0.1.0...v0.4.0
[0.1.0]: https://github.com/muhammadmahadazher/ModelMRI/releases/tag/v0.1.0
