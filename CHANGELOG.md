# Changelog

Notable changes to `modelmri` and `modelmri-record`. Format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/); versioning is
[semantic](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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
