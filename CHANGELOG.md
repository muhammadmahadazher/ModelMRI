# Changelog

Notable changes to `modelmri` and `modelmri-record`. Format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/); versioning is
[semantic](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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

### Fixed

Seventeen portability and path bugs, from a 24-agent audit of code that was
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
