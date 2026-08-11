# Working log

## 2026-08-11 (the lint gate) — CI was enforcing a rule set nobody had chosen

There was no `[tool.ruff]` in `pyproject.toml`, and no `ruff.toml` anywhere in
the repo or above it. So `uv run ruff check .` in CI enforced whatever the
pinned ruff happened to *default* to — and that default is not a stable thing.

Measured on this tree, unchanged, with the two versions on this machine: ruff
0.15.20, the version `uv.lock` resolves, defaults to E4/E7/E9/F and reports
**0 findings**. Ruff 0.16.2 defaults to a much wider set and reports **159**
on the same files: 55 BLE001, 31 RUF100, 11 B023, 10 I001, 9 PLW1510, 8
FURB167, and the tail. Confirmed it was the version and not a stray config:
`ruff check --isolated` on a four-line file with a bare `except Exception:
pass` reports S110 + BLE001 under 0.16.2 and nothing under 0.15.20.

The formatter drifted too, which is the part that explains an odd file count.
`ruff format --check .` reported 80 files before this change and 60 after: the
extra 20 are the Markdown. 0.15.20 refuses them — *"Markdown formatting is
experimental, enable preview mode"* — while 0.16.2 formats them by default and
wants to rewrite 5 of the 12 under `docs/` and `README.md`. Those files are
hand-wrapped for the rendered page, so `.md` is excluded explicitly rather
than left to whichever version is installed.

The consequence is the part worth writing down. Nobody had to do anything
wrong for this to fire. The next `uv lock --upgrade` — for any dependency, on
any branch — would have turned main red with 159 findings belonging to nobody's
change, and the person holding it would have been whoever happened to bump a
dependency.

**The gate is now written down.** `[tool.ruff.lint] select` is a full
replacement for the default rather than an addition to it, so the list in
`pyproject.toml` is the whole gate on any version. Both installed versions now
resolve it to **the same 160 rules** — `ruff check --show-settings` from each,
rule codes extracted and diffed, and the difference is empty in both
directions — and both pass `ruff check .` and `ruff format --check .`. That
is the property worth having: two versions that differed by 159 findings now
answer identically.

**Chosen, not inherited.** Each family was a decision, and the ones left out
are recorded in the file next to the ones kept:

- **BLE001 stays out — and the reason is not the one I expected.** The count
  is 63, not the 55 first reported; the difference is the 8 sites that already
  carried a directive. What decided it is that **26 of the 63 are false
  positives by ruff's own exemption**. Ruff does not flag a blind except whose
  body logs the exception — it flags one that calls a helper which logs it. I
  checked with a two-function file under `--isolated`: the inline
  `log.exception(...)` handler passes, and the identical handler that returns
  `_internal(err, where)` is flagged. Every 500 arm in `server.py` is the
  second shape, because that logging was deliberately factored into
  `_internal` after handlers were caught returning torch's text — including an
  absolute path — straight to the browser. So selecting the rule would mean 63
  new directives, 26 of them suppressing a diagnostic that would not exist had
  the log call been copy-pasted instead of shared. It would also destroy what
  the 8 existing directives were *for*: when every site is marked, the mark
  stops being a signal.
- **S110 comes in, where BLE001 does not.** Different claim, and the
  distinction is the whole point. BLE001 asks whether a catch is too broad;
  S110 asks whether the handler body does nothing *and* records nothing, which
  is the actual shape of an accidental swallow. It fires 4 times, all fallback
  chains where the handling is the next statement, so the price is 4
  directives — and this repo has already had to do one pass over silent
  excepts, so a guard against the next one is worth four comments.
- **B023 stays out, B905 came in.** All 11 B023 sites run their closure to
  completion inside the iteration that made it; the websocket loop drains its
  queue to the sentinel before reading the next message, and `shutil.rmtree`
  returns before the next `for` step. The rule cannot see that. B905 is the
  opposite case: three `zip()`s whose operands are same-length *by
  construction* (`topk` returns values and indices of one shape; `pool.map`
  yields one verdict per entry). `strict=True` turns each into a check, and
  zip's default there was to truncate and publish the mismatch.

  This one is worth flagging as what it is: **a behaviour change made to
  satisfy a linter**, which is normally the thing not to do. It earns the
  exception because the behaviour it replaces is silent. `zip` without
  `strict` drops the tail, so a mismatch in `hub.py` would have marked gated
  models unusable on no evidence, and nothing on screen would have said a
  verdict was missing. Raising is the louder failure and the right one. I
  checked all three operand pairs by hand rather than take the claim.
- **E501 stays out.** The formatter owns line length. All 19 lines over 88
  columns are comments, URLs or unbreakable literals.
- **N802 in, the rest of N out.** N818 wants `Refusal`, `BadRequest`, `TooBig`
  and `LoadCancelled` renamed to end in `Error`, and those names are the
  vocabulary the API and the UI use. N803 wants the SAE's `W_enc`/`W_dec`
  lowercased; they are named after the matrices in the paper. The two
  `# noqa: N802` on `cli.py`'s `do_GET`/`do_HEAD` came off, which looked wrong
  until I checked why: ruff exempts a method that overrides a *recognisable*
  stdlib base, and those are on a `SimpleHTTPRequestHandler` subclass. A probe
  confirms the split — the same `do_GET` on a plain class is flagged, on the
  stdlib subclass it is not.
- **RUF100 in**, which cost 28 edits and was the most useful thing here.

**28 `# noqa` directives were suppressing nothing.** 14 `ANN001`, 8 `BLE001`,
2 `F401`, 2 `N802` and 2 `E402` — directives for rules that were never enabled,
so they had been decorative for as long as they had existed. Two were worse
than decorative: `# noqa: E402` on the imports in
`packages/modelmri-record/tests/test_record.py` was live under 0.15 and dead
under 0.16, because ruff's E402 carve-out for `sys.path` manipulation covers
that file's `sys.path.insert` but not `test_ablate.py`'s
`pytest.importorskip` — the same-looking pattern, two different answers. Every
directive removed kept its prose: `# noqa: ANN001 - torch's signature` is now
`# torch's signature`, which is the part that was ever true.

`ruff>=0.8` became `ruff>=0.15.20`. A floor, not a pin — `select` protects
against the *default* changing under an upgrade, but not against a new rule
being added to a family that is selected, so an upgrade is still a diff worth
reading. Nor does it protect against a rule changing its mind: `ISC004` fires
4 times under 0.16.2 and 0 times under 0.15.20 with an identical name and an
identical one-line description. Same code, same words, different answer. That
is the argument for reading the diff rather than trusting the version number.
After all of it: 15 directives remain (10 `E402`, 4 `S110`, 1 `A002`),
`uv.lock` still resolves 0.15.20, and 457 tests pass.

A note on how this was done, because it is the kind of thing that should be
written down. The per-family assessment was farmed out to parallel agents with
an explicit instruction not to edit anything. Several edited anyway — the
config block, 28 directives across 24 files, `uv.lock` — concurrently, which
is why one reviewer reported the tree shifting underneath its own checks. The
work was largely good and most of it survived, but nothing here was kept on
its say-so: the four load-bearing claims (the delegated-logging exemption, the
websocket sentinel, the `N802` stdlib carve-out, the cross-version rule diff)
were each re-derived here, and four stated numbers were wrong — 55 sites for
63, 47 directives for 63, 150 rules for 160, and "0.15 does not read `.md`"
for "0.15 reads it and declines without preview mode".

## 2026-08-11 (the load meter) — four bugs behind one impossible number

"5.0 GB / 2.5 GB", reported from the app. A number that cannot happen is the
best kind of bug report, because it is not a matter of taste: something is
counting two different things and calling them the same thing.

**The numerator and the denominator disagreed about which files count.** The
total came from the repo's top-level files; the on-disk figure walked the whole
tree with `rglob("*")`. `meta-llama/Llama-3.2-1B-Instruct` ships
`original/consolidated.00.pth` — 2.472 GB, the same weights in Meta's own
format, which `from_pretrained` never opens — beside `model.safetensors` at
2.472 GB. Measured on the real cache: 4.955 GB on disk against 2.481 GB
expected, **199.7%**. The fix is not a clamp. The Hub's file list now decides
one set of names and both sides count that set; measured after: 2.481 / 2.481,
**100.0%**. Two more counting bugs fell out of the same walk — a cache holding
two revisions of one repo had them *added* together, and one file vanishing
mid-walk (the hub moves blobs into snapshots while a load runs) zeroed the
whole count because the entire walk sat under one `try`.

**The picker had its own copy of the same walk**, and its own copy of the same
bug: it listed that model at 4.96 GB. Both sides now call one function. Two
implementations of "how big is this model" is two things that can disagree,
and these did.

**The second screenshot was the more serious one.** It showed
`Qwen/Qwen2.5-0.5B-Instruct` — a model with neither number — against "5.0 GB /
2.5 GB · 493s". Those are Llama's figures. `py-spy` on the live server:
thread 37732 blocked in `convert` inside `.to(device)`, and thread 25128
blocked at `runtime.py:492`, which is `with self._lock`. So the Qwen load was
queued behind a Llama load that had stopped returning, forever, with no
timeout and no message — while the browser labelled the running load with
whatever the picker showed. Two independent defects producing one screenshot.
A second load is now a 409 naming what holds the slot; the bar names
`progress.hf_id`.

**Was it slow or stopped?** Worth separating, because the answers differ. The
server had used 0.3 CPU-seconds and read 0 bytes over 12 s. From another
process, the same weight file read at **295 MB/s** and the same GPU took
host-to-device copies at **1266 MB/s** with 7.18 GB free. Neither the disk nor
the GPU was the problem, so this was stopped, not slow — and "no bytes and no
CPU" is evidence the meter can gather itself. It now does, in every stage,
using `time.process_time()`: a plain counter read, which unlike asking CUDA how
much memory it has handed out cannot block on the very thing that is stuck. A
fresh server loaded the same model in 20.7 s.

**And one warning that fires on healthy downloads.** The 45 s stall threshold
predates `hf_xet`, which huggingface_hub 1.x installs by default. Watching a
real 324 MB download of `EleutherAI/pythia-160m` land on disk: the blob sat
unchanged at 2.1 MB from 4.1 s to **75.7 s**, then again from 85.4 s to
144.4 s. xet reconstructs from its own chunk cache and writes in large,
infrequent jumps, so that download would have been declared stalled twice
while working perfectly. 180 s clears the longest gap measured, and the
sentence is now scoped to the download stage — a load wedged on the GPU was
being reported as a stalled download, which sends people to look at their
network.

Every threshold in the file now carries the measurement that set it. A number
chosen by feel is a number nobody can revisit.

## 2026-08-11 (feature ablation, after the adversaries) — the tick that was green 38 times wrongly

Three adversarial passes over the new feature ranking returned thirteen
findings. I reproduced every one on gpt2/float32/CPU before touching anything,
and rejected the framing of one.

**Phase 0 said the intervention was the one defensible choice, and it is — but
Phase 0 did not check what the intervention does to the SAE's own reading of
the stream, and neither did the shipped code.** The mechanism check re-encoded
the edited stream, found feature 5856's activation was exactly 0.0 of 35.546,
returned `removal_verified: true`, and said in a comment that this "asks
whether the intervention does what its name says, which is a property of the
edit and the SAE, not of each feature." Run on all 43 candidates instead of
one, it fails on **38**, with residual shares from 10.11% (#23035) to 60.26%
(#5926). Worse, the five that pass do not pass cleanly: 5856's pre-activation
goes 35.546 → −2.331, a drop of 37.877 against an activation of 35.546, so
"removed" means "removed 6.6% too much and relu clamped it". The cause is one
line of arithmetic nobody had looked at: `W_enc[:,f] · W_dec[f]` over the SAE's
24,576 features has mean 0.8387, min −0.3819, max 1.3072. Encoder and decoder
directions are not dual, so subtracting a feature's decoder direction does not
zero its encoder reading, and the amount left over is a property of the
feature, not of the edit.

The right split turned out to be free. The claim that IS a property of the edit
— "the stream the model received is `x − act·W_dec[f]` and nothing else" —
costs the one forward pass the old check already spent, and measures 0.0
deviation in float32. The claim that is per feature costs **no** forward pass
at all: the tensor the model receives at a resid_pre hook IS the tensor written
in, so re-encoding the cast copy answers the same question, and it agreed with
the through-the-model version to 3.6e-06 across all 43 rows. One column of
`W_enc` instead of all 24,576 makes it 768 multiply-adds a row. So the honest
version is cheaper than the dishonest one was.

**The second docstring sentence was worse than the first.** "Nothing else in
the stream moves — not the reconstruction error, not the d_model mean, not the
other features." Nothing else in the STREAM moves; the SAE's decomposition of
the result is not the old decomposition minus one row. Removing 5856 moves 44
other features by more than 1e-6, drives **33 of the other 42 firing features
to exactly zero**, starts 2 silent ones, and moves 42.4943 of activation
outside the target against the 35.546 it removed inside it — 119.55%. `||err||`
at that token goes 21.3036 → 31.8553. The d_model mean goes 0.0786990 →
0.1690938, which is correct and intended (`mu` is held at the value the
decomposition was taken with) but is not "does not move".

**The one I partly rejected.** An adversary showed that five random Gaussian
directions at the top feature's norm score 0.0666–0.1093 nats, and concluded
that "41 of the 43 rows sit under the pedestal" — the ranking is a ranking of
edit magnitude. The measurement reproduces exactly; the comparison does not
hold. Those 41 rows have *smaller edits*, and a pedestal scales with the norm
of the edit. Measured properly — one control per row, at that row's own norm,
at that row's own tokens — **34 of 43 rows clear their own control**, not 2.
The finding underneath is still real and is now shipped: a score is partly the
size of the edit, the top row clears by ~4x rather than by everything, and nine
rows do not clear at all, two of which (#22852, #1288) are in the bar chart's
plotted top-8. That costs a second forward pass per row, so the cost went from
`n + 6` to `2n + 6`: 92 passes / 10.09 s at position scope, 518 / 49.44 s at
prompt scope, on this CPU.

**A baseline measured somewhere the edits do not land.** `residual_kl`
substituted the SAE's reconstruction at the attributed token only, regardless
of scope. At `scope="prompt"` the edits land at all eleven tokens, where the
same substitution costs 0.221217 rather than 0.077530 — 2.85x. Against the
small one, 2 of 43 features clear; against the right one, 1 of 256. The panel
was crediting feature 11149 with clearing the SAE's own error when it does not.

**And a caveat that inflated itself by squaring one side.** "aggregate FVU
0.0012 — but at this token the SAE fails to model 20.4% of the stream's norm."
FVU is a squared-error fraction and `residual_share` is a norm fraction. The
calibration already carried `rel_err` = 0.029397, the directly comparable
number, and did not use it. The real discrepancy is 7x, not 200x; the sentence
made it three orders of magnitude.

**Three panel bugs with one shape: the client reading the response as if it
were the run.** `rankFeatures` sent no `top_k`, the server trimmed 256 scored
rows to 64, and everything downstream treated absence-from-`ranked` as
never-tested. #18994, scored 0.00031514 at causal rank 72, got a chip reading
"not tested" and a tooltip reading "not asked, not found unimportant" — the
precise inversion of the invariant the code comment above it claimed to be
enforcing. The tail count read "54 more were tested and scored lower" directly
above the server's "256 of 494 firing features were tested". The server had
already built `n_scored`, `n_returned`, `rows_note`, `n_below_resolution` and
`n_negative_kl` for exactly this, and not one of the five existed in the
TypeScript interface.

**The refusal that was invisible until after the click.** ModelMRI selects
bfloat16 for every NVIDIA GPU, `rank_features` refuses anything but float32,
and the panel gated the button on DEMO/VIEWER only. So on the machine this
project is developed on, the default configuration renders a button, quotes a
cost badge of 67 passes, and answers 409. `/api/session` already carries
`model.dtype`; the panel takes it as a prop now and prints the runtime's own
sentence in place of the control. The panel's own comment two lines above the
gate ("a button that only ever fails … teaches them the measurement does not
work") was the argument for doing it.

**Contradicting Phase 0, plainly.** Phase 0's record says the choice was
between (a)/(c) and (b), and that (a) is safe because "removing act*W_dec[f]
from the centered stream and from the raw stream are the same subtraction."
That reason is false: `act*W_dec[5856]` has d_model mean −0.0903948, whose
`|mean|·sqrt(768)` = 2.5051 is 7.05% of the edit's norm, so centering strips
part of it. The conclusion survives for a different reason — (c) re-adds the
ORIGINAL per-token mean, so both edits equal `x − act·W_dec[f]` — and the wrong
reason is what made the removal check's framing look safe in the first place.
Phase 0 also did not measure the encoder/decoder duality, the same-norm
control, or the scope of the reconstruction baseline, which are three of the
four things this pass changed.

## 2026-08-10 (attribution, after the adversaries) — the guard that could not fire

Four adversarial passes over the shipped attribution feature returned sixteen
findings. I reproduced every one before touching anything, and rejected one.

**The one that mattered most was a refusal that could not refuse.** The module
docstring promised to reject a model that ignores `position_ids` and derives
them from `attention_mask.cumsum(-1) - 1` — the whole reason the file passes
them explicitly. The check ran with an **all-ones mask**, under which the
derived positions *equal* `arange(S)` by construction. So the failure it named
was the one case it could never see. Written as a toy model whose logits are a
pure function of the derived positions, with `input_ids` never read at all, it
came back with `noise_floor_kl` 0.0, `mask_verified` True and a clean ranking
of three tokens — every score the suffix's position shift and nothing else.

The fix is one more pass, with `position_ids` reversed, gating on the answer
**moving**. Reversal and not a shift: RoPE is invariant to moving every
position together, so `arange + 1` moves gpt2 3.396605 nats (learned absolute
embeddings) but Qwen3-0.6B only **5e-06** — inside the range Qwen3's own
content tokens score in. Reversed: gpt2 2.166768, Qwen3-0.6B 0.011300,
gemma-3-270m-it 4.616208. The smallest is 11,300x the tolerance.

This is conservative in one direction on purpose. A model with *no* positional
dependence at all is safe from the re-phasing, and is also indistinguishable
from the mask-deriving one from outside, so it is refused too. That cost a test
fixture rewrite: the toy `Listener` was a bag of visible keys with no phase to
shift, which is exactly why it passed a check it should have failed.

The same shape of bug sat next to it: `used_position_ids` asserted every pass
got the same tensor, but nothing in the file ever supplied a different one, so
the assertion could not fail. It is a real check now — exactly one stray is
allowed, and it must be the probe.

**A ranking of the wrong model, with nothing to notice it.** Both interventions
took their epoch check *outside* `self._lock`, and `load` holds that lock
across the epoch bump and the model swap. Scripted with two toy models: the
check passed, the call blocked on the lock, the epoch went 1 → 2 and the
weights swapped, and `attribute_tokens` returned scores — while the identical
call one moment later refused. Nothing downstream can catch that: the ids are
the right length and the KLs are finite.

**Three UI findings were the same failure wearing different clothes: a caveat
that enumerated reasons and left one out.** The panel told the reader a bar-less
chip means "outside the causal cone or the position itself", while the
64-candidate cap routinely leaves 30 in-cone candidates unmarked. It told a
reader whose typed span fell outside the tested window that their words "were
not candidates" — they were candidates; nothing asked them. And it filed the
model's own output under "chat template scaffold" on gpt2, two lines below its
own note saying gpt2 has no chat template: attributing at index 16 of a
12-token generation, **11 of 15 rows** are the model's own, including the
highest score in the run (index 10 ' Republic', 0.69132). All three are the
same class of error — a closed list of reasons that is not closed — and the
server now returns `tested_span` and `n_prompt` so the client can tell "not
asked" from "asked, and nothing".

**What I rejected.** One finding said the sink figure 4.86309 is "low in the
third decimal" because `kl_nats` takes `log` of an already-computed softmax
instead of `log_softmax`, attributing the 0.001673 gap to fp32 precision and
explicitly ruling out the 1e-12 floor on the grounds that nothing underflowed
to zero. The magnitude is right and the cause is not. Underflow to *zero* is
not the floor's failure mode: at index 0, **10483 of 50257** vocabulary entries
sit under 1e-12 without reaching zero, and the p-weighted cost of clamping
exactly those is **0.001672** — the entire gap. Precision is not involved:
the identical arithmetic in float64 gives 4.863086102936881 against float32's
4.863085746765137. So 4.86309 is not an imprecise estimate, it is an exact
report of a floored quantity, and the suggested fix (quote it as 4.8631) would
have made the code *less* accurate about what it computes. Changing the
estimator instead would silently move every KL in the package. What shipped is
the measurement in `kl_nats`'s docstring: the floor costs nothing on ordinary
rows and 0.001672 nats on the one row where the intervention collapses the
tail — which is the row the panel highlights.

Two of the adversaries' numbers did not reproduce here and I used my own:
Qwen3's residual weight in masked column 0 is **0.077148**, not 0.052734, and
the claim that 56 of 65 bars land within half a pixel of the strip's floor is a
DOM measurement I replaced with one anybody can check from the payload — 60 of
65 bars under 5% of the tallest, 34 under 2%, on a 73-token gpt2 prompt.

## 2026-08-10 (token attribution) — Phase 0, including the parts that refuted the plan

Before writing `modelmri/attribute.py` I measured the six things the design
depended on. Conditions for everything below, and they are not optional
context: prompt `"The capital of France is"`, **bfloat16 on cuda**,
`attn_implementation="eager"` (matching `runtime.py`), one unbatched sequence,
attributing at the **last prompt token**, chat template applied with
`add_generation_prompt=True` where one exists. Masking is
`attention_mask[0,i]=0` with `position_ids=arange(S)` passed explicitly, so
removing a token does not re-phase RoPE for the suffix. KL is
`ablate.kl_nats` on `ablate.distribution`, imported rather than reimplemented.

**1. Noise floor: exactly 0.0**, on gpt2, Qwen3-0.6B and gemma-3-270m-it, and
the logits are bit-identical (`torch.equal`) between `model(ids)` and
`model(ids, attention_mask=ones, position_ids=arange)`. The explicit-argument
path does not select a different kernel. No floor offset needed anywhere.

**2. Index 0 is a sink, not content.** gpt2, no BOS (S=5, pos=4): index 0
'The' **4.86309**, 3 ' France' 1.74563, 1 ' capital' 0.90210, 2 ' of' 0.86315,
4 ' is' 0.06375. With `<|endoftext|>` prepended (S=6, pos=5): index 0
**4.76083**, 4 ' France' 1.35811, 1 'The' 0.46107. The top score *stays at
index 0* while the token sitting there changes completely (2.1% apart), and
'The' itself falls **10.5x** when it moves to index 1. The score follows the
position. Controlled against the obvious artifact — with a 2D mask, masking
key 0 leaves query 0 with no keys at all — by re-running with a 4D mask that
spares the diagonal: every off-diagonal score reproduced bit-for-bit
(4.863085746765137 both ways).

**3. Additivity: THE PLAN WAS REFUTED.** The plan expected the direction of
the error to invert between models. It does not invert — over the typed span
all three ratios are *below* 1: gpt2 0.9816, Qwen3-0.6B 0.9168,
gemma-3-270m-it 0.3480. Singles sum to **less** than one joint mask in every
case, by 2% up to 2.87x. What *does* invert is the choice of which tokens you
sum: over the rows the panel actually shows, gpt2 goes to 1.82x and gemma to
1.58x — same model, same prompt, same forward passes, opposite sign. So no
correction factor exists and the panel quotes this run's own two numbers.
This is also the *opposite* of head ablation in `ablate.py`, where gpt2 heads
over-count 8x and gemma heads under-count; that docstring's framing must not
be copy-pasted onto token scores.

**4. Self-mask: near-no-op on gpt2 only, and that half of the argument is
useless.** gpt2 0.06375 against a max of 4.86309 — 1.31%. But on Qwen3-0.6B
the self position scores **6.24429 and is the LARGEST of all 13 candidates**
(next: 'assistant' 2.02161), and on gemma 1.92183 against a max of 9.33529.
"It is tiny anyway" is false. The rule stands on geometry instead: sparing the
diagonal drops it to exactly 0.0, so what it measures is the mask's shape.

**5. User-content span:** all three tokenizers are fast, the prompt is a
literal substring in all three, spans gpt2 (0,5), Qwen3 (3,8), gemma (5,10).
One gotcha: fast tokenizers hand back zero-width `(0,0)` offsets for added
special tokens, which fall inside any span starting at char 0.

**6. Control-token detector, and a false positive I nearly shipped.**
`convert_tokens_to_ids(" the")` returns **50256** on gpt2 — the unk fallback,
which *is* `<|endoftext|>`, which *is* in the control set — because GPT-2
spells that token with U+0120, not a space. `encode()` gives 262, which
correctly does not fire. Probe token ids taken from the sequence, never
strings. Second: the wide regex `^<\|?.+\|?>$` claims **6573** ids beyond
gemma's 8 declared specials and fires on `<div>`, `<b>`, `<html>` — ordinary
content in that vocabulary. Only the pipe form shipped.

**Three things that shaped the design, none of which blocked it.** (a) The
chat template dominates every ranking — Qwen3's top three are the template's
'\n' 6.24429, 'assistant' 2.02161 and '<|im_start|>' 0.32266 while every typed
word sits at 3.1e-05 to 7.9e-05, four to five orders down; a list that does not
separate them answers "the chat template" every time. (b) That Qwen3 content
signal is **below the model's own numeric precision**: the entire content sum
is 2.50e-04 nats while merely switching gpt2 from bf16 to fp32 moves a
distribution by 1.88e-02. (c) gemma emits **two `<bos>` tokens** — its chat
template writes one and `tokenizer()` adds another — and both score nonzero
(0.83402, 0.78318).

Also worth recording because a plan document had it wrong: the argmax at the
default position on gpt2 is ' the' at **p=0.097824** in bf16. The plan's
0.084592 is the *fp32* number, and KL(fp32 ‖ bf16) at that position is
0.018762. Qwen3's `<think>` at p=0.999531 was right. gemma's argmax is 'The'
at p=0.560747 and is *not* a control token, so "the answer is a control token"
is model-specific and not a general case to design around.

## 2026-08-10 (later still) — the timing claim was wrong twice

Yesterday's correction replaced three inconsistent timings with one measured
set — 1.0 s, 10.3 s, 137 s on an RTX 4060 — and a claim that extrapolating
the whole-model sweep from one layer is "within 1%". Both were taken honestly.
Both are wrong, and finding out why was worth more than the numbers.

Re-measured, the same code on the same GPU:

| | earlier | now |
|---|---|---|
| gpt2, one layer | 1.0 s | 0.17–0.56 s |
| gpt2, all 146 passes | 10.3 s | 1.6–5.8 s |
| Qwen3-0.6B, all 450 | 137 s | 19.9–51.9 s |

**Absolute seconds do not transfer, even between sessions on one machine.**
Per-pass cost for the same model ranged **12 to 71 ms** depending on process
and GPU state — a 6× spread. Every figure in seconds I published was a
measurement of one afternoon, presented as a property of the tool.

**The "within 1%" was an artefact of measuring once.** Across three repeats
the naive extrapolation was off by −12.1%, −1.0%, +0.9% on gpt2 and +46.8%,
+0.8%, −2.5% on Qwen3. The outliers are the *first* run each time.

**The cause is CUDA warm-up, and it had a product consequence.** The first
ranking after loading a model runs several times slower than the rest —
Qwen3's first layer 3.05 s against 0.80 and 0.78; its first whole-model sweep
51.9 s against 19.9 and 20.0. The panel derived its estimate from the *latest*
ranking, and the "all N layers" button appears only after the first one. So
the very first estimate a user ever saw was computed from the slowest sample
that will ever be taken — the worst case, by construction, 46.8% over on that
run. It now keeps the **fastest** rate seen, because warm-up only inflates.

What is actually true, and now shipped: the **pass count** is portable (146
for gpt2, 450 for Qwen3-0.6B); the per-pass cost is not, so it is measured on
the user's machine; back to back the rate is steady (1.0–1.1× over six runs)
and once warm the extrapolation holds to within 2.5%. Every seconds-figure
has been removed from README, docs, `runtime.py`, `server.py` and the panel.

The lesson is not "measure" — I did measure. It is that a number measured
**once** is a sample, and shipping it as a property is the same error as not
measuring at all, wearing better clothes.

## 2026-08-10 (later) — the demo was a diorama

The audit's design phase asked which surface most deserved the next feature,
and the answer was uncomfortable: the hosted demo. It is the only ModelMRI
99% of visitors will ever touch, the README links it twice — and it was the
least verified thing in the repo. The `.mri` viewer beside it, built on the
identical patched-fetch trick, is gated cell-for-cell by `viewer_check.py`.
The demo's entire gate was `test -f demo-dist/index.html`.

That asymmetry showed, in the first thirty seconds:

- **Move the head dropdown.** `demo.ts` read `layer` and never `head`, then
  fell back to the first baked slice. Three slices were baked against a meta
  advertising 12 x 12, so **141 of 144 selections drew a different head's arcs
  than the dial said** — and silently, which is the only kind of wrong nobody
  reports.
- **Click "Rank heads."** The capability the README leads with had no handler
  at all: 409, under panel advice to "generate again", which could not work.
  Twelve endpoints were dead the same way — accelerator badge, storage panel,
  logit lens, HF tab.
- **Type a prompt.** Any prompt returned the baked generation. "what is 2+2"
  produced a confident sentence about the Eiffel Tower, then attention over
  the Eiffel Tower's tokens beneath the words you had typed.

Now the demo bakes all 144 slices, a ranking for every layer under both
baselines, both whole-model sweeps, the 60 comparisons the ranked rows can
ask for, the small endpoints every panel calls on first paint, and a real
`.mri` of its own run so "Share this view" produces a file that opens in the
viewer next door. 697 KB for the LLM bundle, against the 54 KB the robot
bundle already cost — completeness was never what made it small.

A miss now 422s with its reason, in the same words `viewer.ts` uses, and
`demoFetch` returns `{status, payload}` like `viewerFetch` so it *can*. An
unrecorded prompt is refused, naming the prompt this demo did record: a
banner does not fix answering the wrong question.

**The gate is the point.** `tests/demo_check.py` extracts every `/api/...`
literal `api.ts` can call, diffs it against what `demo.ts` answers, and fails
on any gap — so the next dead endpoint fails a build rather than a visitor.

**Two bugs found by looking rather than reasoning:**

- The first version of `demo_check.py` treated every handler as a prefix, so
  `/api/sae/available` counted as covered by the exact-match `/api/sae`. The
  check under-reported the very gaps it exists to find, and reported 5
  unhandled endpoints where there were 11.
- A loop variable named `baseline` in `bake_demo.py` shadowed the one holding
  the generated text, and the demo shipped the literal string **"mean"** as
  its generation. No schema check would catch that — it is a string either
  way. Caught by opening the built demo and reading it. `demo_check.py` now
  asserts a generation is a generation.

Verified on the live public URL after deploy: 144 of 144 slices return the
layer and head asked for, an unbaked slice 422s, both baselines rank and
disagree, "what changes?" reports 7 of 529 cells moved, and 21 endpoints
return zero occurrences of "not available in the demo".

## 2026-08-10 — 0.8.1: auditing the audit

Yesterday's correction pass fixed the numbers it went looking for. This one
went looking for the ones it missed, by sweeping every numeric claim in the
repo — README, docs, docstrings, comments, tests — and asking of each: is
there evidence anyone measured this? 45 candidates, adversarially verified
(default verdict: refute). 8 survived.

**The correction had missed its own back yard.** `tests/test_ablate.py:127`
still carried the retracted `+21.96 / +18.06 / ~6x` verbatim, and `:225` still
claimed a bf16 noise floor "around 5e-3" that measures exactly 0.0. The test
file asserting that we rank by KL rather than logit difference was explaining
why using the number that motivated the rule and was wrong. `server.py:708`
still shipped `0.12-0.68 s per layer against 1.4-19.6 s` — the same stale
timings corrected in four other files, and this copy is served publicly in the
OpenAPI schema at `/docs`.

**README's ranking block did not reproduce.** It showed `L0 H7 KL 0.866,
p(" the") 0.112 -> 0.073`. Measured: 0.784 / 0.085 -> 0.062 in fp32, and
0.898 / 0.098 -> 0.057 in bf16. Neither matches, and `0.866` appears exactly
once in the entire git history — in the commit that wrote it.

The fix is not a better number, it is the setup line. The same three heads
score **0.784 (fp32), 0.898 (bf16), 0.825 (over a 261-token generation)**. A
KL depends on prompt, dtype and sequence length, so a figure quoted without
them cannot be checked by anyone — which is precisely how three different
values coexisted in three files, each looking authoritative.

**Four more, all the same species — a figure nobody rechecks:**

- `7 KiB` for the recorder wheel, in three files, against a real 9,152 bytes
  (8.9 KiB), while README said 9 KiB. A previous commit had *already* fixed
  this once with the note "a figure nobody rechecks is a figure that drifts",
  and it drifted again. Now `test_the_recorder_wheel_size_is_stated_identically_everywhere`
  checks the four sites against each other and against the built wheel. Run
  against the unfixed tree it fails with `the four disagree: {7.0, 8.9, 8.9, 8.9}`.
- "attention rows summing to **1.000**" for six models. The recorded figures
  are 1.000-1.002, and two of the six had no recorded run at all —
  Llama-3.2-1B-Instruct (gated, 403) and OLMo-2-1B (download stalls). A table
  headed "Verified, not asserted" now contains only what was.
- "the reader is about **200 lines**" for `vla_data.py`, which is 286
  non-blank and was 256 when the sentence was written — wrong on the day it
  shipped. Replaced with the property that stays true: it imports no
  `lerobot` code.
- "public SAEs exist for about **a dozen** models", in six places, sourced
  from nothing. The registry knows four repositories, so it says four.

**One finding I rejected.** The sweep flagged LOG.md's "byte counter climbed
to 149%" as invented, on the grounds that nothing in the repo computes it.
Nothing does — it was read off a live load: 819,086,596 bytes counted against
an expected 550,959,861, which is 148.7%. Recomputed and kept. An observation
recorded in this log *is* the measurement; that is what this file is for.

## 2026-08-09 (later) — 0.7.0 + 0.8.0, and auditing my own shipped numbers

**Ranking heads, and comparing runs.** 0.7.0 added *Rank heads*: zero each
head in a layer, run the model again, measure how far the next-token
distribution moves. 0.8.0 added *what changes?* — the same generation with
and without one head, subtracted cell by cell.

**Then the launch post made me check a number, and the number was wrong.**

Writing it up, I reached the claim that a raw logit difference overstates a
head's importance by 6×. That figure came from a design review, not from me.
Measuring it on gpt2 L0H0 with "The capital of France is": top logit −0.258,
vocabulary mean −0.145, honest residual −0.113 — a **2.3×** overstatement.
I published the measured one.

That should have been the end of it. Instead I went looking for where the 6×
came from, and found it in `ablate.py`'s own module docstring, alongside three
more figures I had never taken. Measured, all of them:

| Shipped | Measured |
|---|---|
| logit +21.96 / mean +18.06 / ~6× | −0.258 / −0.145 / **2.3×** |
| zero ranks heads 0, 7, 10 | **7, 10, 9** |
| mean ranks head 4 top | **3, 1, 10** (7 falls to sixth) |
| layer-0 per-head KLs sum 4.07 vs 0.44 | **1.995 vs 0.208** |
| gemma 0.003 vs 1.69 | **0.0007 vs 6.57** |
| bf16 noise floor ~5e-3 | **exactly 0.0**, CPU and CUDA, fp32/bf16/fp16 |

Every *qualitative* claim held — head_dim really does differ, the baseline
really does reorder the ranking, the scores really do not add up, gemma really
does invert. Every *quantitative* one was wrong. The docstring also named no
prompt, which is why nobody could have checked it, so it names one now.

**Timings were worse, because there were three of them.** README said 1.8 s
for all 144 heads, `runtime.py` said 1.4 s, and I measured **10.28 s** through
the real path on an RTX 4060. Three different figures for one measurement is
proof on its own that none were taken. Qwen3-0.6B's full sweep is **137 s**,
shipped as 19.6 s.

That last number is a product problem, not just a documentation one, so the
panel now quotes the cost before it starts — and the whole-model button does
not appear until one layer has been ranked, because it cannot quote a number
it has not measured. Verified live in the browser: estimate ≈6 s, actual
5.95 s.

**(Superseded 2026-08-10.)** The "within 1%" extrapolation claim written here
does not hold, and the absolute seconds do not either. See the entry above.

Also exposed the mean baseline in the panel, which already *told* users the
order depends on it. It does: zero ranks L0H7 first at KL 0.825; mean drops it
to fifth at 0.070 and L0H10 out of the top five entirely.

**Two bugs found by watching a load I only meant to use as a fixture.**

Loading gpt2 sat at "reading from local cache, no download needed" for 275
seconds while the byte counter climbed to **149%** of the total. Both symptoms,
one cause and one aggravator:

- The prefetch ignored TensorFlow, Flax, ONNX and TFLite weights but not
  `rust_model.ot` or a redundant `pytorch_model.bin`. gpt2 ships all three:
  **1.7 GB pulled where 523 MB was needed.** The `.bin` is only skipped when a
  root-level `.safetensors` actually exists to load instead, so models that
  predate safetensors still get their only weight file, and an adapter's
  stray safetensors in a subfolder does not condemn the real weights.
- "Already cached" was decided once, from the tree's size at t=0 — and the
  tree was big because it held that legacy `.bin`. So the loader announced
  nothing would download and then downloaded, under that message, with every
  number on screen wrong in the same direction. Bytes arriving now counts as
  proof the verdict was wrong.

Both regression tests were run against the unfixed code first and failed for
the stated reason.

## 2026-08-09 — 0.6.0–0.6.2: sharing a finding, and a 1.5 TB near-miss

**The `.mri` format.** `*.mri` had been sitting in `.gitignore` since week one
and was never implemented. It's the obvious missing thing: you find the head
that moves the subject token, and the only way to show anyone is a screenshot
they cannot explore. So a file that holds the observation and not the model —
tokens, attention, the generation, and a note.

Size was the whole design. A 24-layer, 14-head, 141-token attention tensor is
6.7 million numbers; as JSON at four decimals that's tens of megabytes for
something meant to be attached to a message. uint8 against each matrix's own
maximum, then gzip: a 29-token gpt2 run with **all 144 attention maps is
54 KB**. Measured against the live model, worst absolute error 0.002 and the
strongest attention in every row survives. The file states its own precision,
because a number that has quietly lost some is exactly what this project
exists to catch.

The implementation trick that made it cheap: replay is served through
`runtime.attention()`, the same method a live model uses. Nothing in any panel
changed.

**Then someone clicked GLM-5.2.** 753 billion parameters. It began downloading
**1,506.7 GB** onto a laptop with an 8.6 GB GPU and 88 GB of free disk. No
size shown, no warning, and no way to stop it except killing the server
process.

Three separate failures, all mine:

- The picker queried the Hub with `full=true`, which does **not** return
  `safetensors` — so every row came back with no size at all. Switching to
  `expand[]=safetensors` gives per-dtype parameter counts, which is how the
  same 753B model correctly reads 1.5 TB in BF16 and 756 GB in FP8. One
  number would have been a lie for the other.
- No capacity check. There is one now, shared by the HuggingFace and Ollama
  paths so they cannot drift, checked against real free space on the volume
  that download would land on. Disk refusals cannot be overridden; "too big
  for your GPU" can, with a second deliberate click. Enforced server-side —
  a check the browser performs is a check the browser can skip.
- No cancel, and this one was interesting. `from_pretrained` downloads inside
  the calling thread, and Python cannot interrupt a thread blocked in a socket
  read. So the fetch happens in a **child process** now, precisely so it can
  be terminated.

Which promptly produced its own bug: I spawned that child with `stderr=PIPE`
and nothing draining it. `huggingface_hub` writes progress to stderr, the
~64 KB pipe buffer filled, and the child blocked forever. The UI sat at
"551 MB / 551 MB · 234s · reading from local cache" with the weights fully
downloaded. Both streams go to DEVNULL now, and two tests hold it there —
verified red, one of which takes 30 seconds to fail because that is what a
deadlock does.

**Then: reading it shouldn't need the tool.** `modelmri open` worked, but it
imported torch and transformers first — **26 seconds** — to display a 54 KB
recording that needs neither. The first person to run it pressed ctrl-c
partway through, which is the correct response.

Two fixes. A browser viewer at `/viewer/`: the same React app with the API
answered from a file you drop, so a recipient reads a shared analysis with
nothing installed and nothing uploaded. And `modelmri open` now serves that
same bundle from the standard library — **0.26s warm, 0.69s from a cold fresh
install**. The split is finally clean: `modelmri serve` is the tool,
`modelmri open` is a file reader.

The format now has two implementations, so I stopped assuming they agree and
started checking: `tests/viewer_check.py` parses one file both ways and
compares every cell. 6,912 cells, identical checksums. A viewer that renders a
*slightly* different matrix would be worse than no viewer, because nothing on
screen would say so.

**Seventeen path bugs**, from an audit of code I had already tested and
shipped. They share one shape — a location computed correctly in one module
and approximately in another. `import modelmri` died outright on a container
with no resolvable home, *before* `MODELMRI_HOME` (the documented fix for
exactly that) could be read. The HuggingFace token was created world-readable
and narrowed a moment later, which on a shared host is a window. `HF_HUB_CACHE`
was ignored in four places, so the robot panel called a cached checkpoint
missing and suggested a download that landed in the directory it wasn't
reading.

The test that would have caught most of them now exists: run the whole app
inside a synthetic home and fail if any absolute path in any API response
points outside it. **It found a real bug on its first run** — `MODELMRI_HOME`
promised to relocate everything, but a surviving `~/.modelmri/traces.sqlite`
still won, so identical commands produced different storage depending on
upgrade history.

**Lesson of the week**, again: I am not a reliable auditor of code I just
wrote. Four CI failures went out before I started running CI's exact command
locally — including a POSIX-only test that skipped on my machine and failed in
CI *for the code being correct*, because it watched `hub.json` when the atomic
write opens `hub.json.<pid>.tmp`.

## 2026-08-08 (later) — bring your own model, and four bugs that were invisible

**Custom models.** Every other panel is transformer-shaped, so the honest
answer to "does this work on the model I trained?" was no, unless you'd saved
it as a HuggingFace causal LM. Now: an adapter (`def load(): return model`) or
TorchScript, and you get a layer map of one real forward pass — shapes,
activation ranges, dead units, saturation, timing, and the first layer where a
nan appears. A `state_dict` alone is refused with the reason: it's weights
without an architecture, and guessing one would produce a map that looks
authoritative and describes a network nobody trained.

Statistics exclude non-finite values on purpose. One nan propagates through
mean/std/min/max, so the naive version prints nan for every layer downstream
and hides where it started.

`tests/mutation_check.py` breaks `custom.py` twelve ways and asserts the named
test notices. 12/12. Two were survivors when first written, and both were the
test's fault: one asserted a downstream proxy that a leaked hook doesn't
disturb (the leaked hook closes over the *previous* rows list, so it appends
out of sight and the count stays right), and one mutation didn't break what it
claimed to.

**Four bugs, all invisible, all shipped for weeks.**

1. `--glass-fill: var(--glass-fill)` — a self-referential custom property is a
   cycle, so it computes to nothing and takes the whole `background`
   declaration with it. Every liquid-glass surface was fully transparent. The
   model picker was blur with no frost and the hero headline read straight
   through the model list. The owner reported it as "the background isn't
   blurred enough"; it was worse than that. Third time a var() has silently
   voided a declaration here, so there's a test for the class now.

2. The scrim behind that sheet was `blur(3px)` and was doing *all* the work —
   an element with `backdrop-filter` becomes a backdrop root for its children,
   so `.sheet`'s own `blur(40px)` only ever sampled the scrim's flat tint.

3. Keyboard focus was invisible on half the app. `:focus-visible` and
   `.model-row` are both specificity (0,1,0), so the eight `all: unset` rules
   below it won on source order; `.theme-seg button` at (0,1,1) won outright.
   19 of 20 controls in the picker moved focus with no ring — while
   `:focus-visible` matched and `outline-style` computed to `none`.

   The first probe for this reported *every* button as ringless, because
   `element.focus()` doesn't set `:focus-visible` in Chromium. That's a fact
   about the probe. I nearly "fixed" it.

4. A page reload discarded your analysis. The attention and feature panels
   were gated on a client-side counter, so refreshing unmounted them while the
   server still held attention for 141 tokens and would have served it.

Also: errors reached the screen as `Error: 422: {"error":"…"}` on all 19 paths;
the picker resized 266px under the cursor when its list landed; the footer read
`MRI-0.3` through the whole 0.4 line; the hosted demo's "On this machine" tab —
the feature whose entire point is finding your models — said "Nothing found".

**Repo hygiene.** Contributing guide, code of conduct, security policy,
support, issue/PR templates, CITATION.cff, CODEOWNERS, Dependabot, CodeQL,
changelog. SECURITY.md states the trust model plainly: local single-user, no
auth, and loading any model executes code.

**Verification.** `tests/ui_check.py` (17 browser assertions, in CI) plus
`gen_api_docs.py --check` so the API reference can't drift. The unstyled-button
check injects the bug into the live page each run and fails if it isn't
detected — a check that can't rot into a no-op.

0.5.0 built and verified from a clean install into an empty venv. Not
published yet.


## 2026-08-08 — design v5, and dark mode

Researched how Google and Apple actually build colour, motion and type
systems (Material 3 tonal palettes, HIG spring params, OKLCH ramps, and what
Linear/Vercel/Stripe do in CSS), then rebuilt the foundation rather than
adjusting hex codes. Spec: `Blueprint/08-design-system-v5.md`.

**Colour.** Six hues on one lightness schedule, chroma bounded by the real
sRGB gamut. `--color-cobalt` and `--color-attn` were 7 degrees apart -- a
duplicate, not a distinction -- so attention now IS the primary and violet
moved +9.5 to open a 32-degree gap. Neutrals disagreed (warm ground, cool
ink); light is one warm hue now, dark one cool graphite, and that temperature
flip is the theme boundary rather than an accident.

Two AA failures caught by measuring, not looking: amber at 3.76 under a
10.5px heading (now 7.22), and crimson at 4.49 which I introduced during the
rebuild (now 5.9). Every text role passes AA against every surface in both
themes.

**Dark mode.** Three states, because a binary toggle strands you off the OS
setting with no way back. The failure this was always going to have is
canvases -- CSS re-cascades free, rasterised pixels do not. Verified the hero
repaints on switch: mean pixel 530 -> 356 -> 532.

**Signature moves, each carrying data.** The section divider is a measurement
rule with ticks on the token strip's own 8px pitch. Feature rows arrive
ranked by activation (28.4 at 0ms, 27.1 at 22ms, monotonic down) because
every row is the same violet and rank is the only channel left. Numbers use
tabular figures so a streaming count does not shiver. And a 640ms specular
scan crosses a panel only when genuinely new data lands, replacing an
entrance animation that fired on every mount and therefore meant nothing.

**Two bugs testing caught that reasoning did not.** FeaturesPanel has two
return paths rendering `.panel feat`; my scan ref attached to whichever came
first in the file, which was the EMPTY state, so the panel that shows data
never scanned. And a MutationObserver silently watched stale nodes across a
remount and reported a false negative -- polling the live DOM showed both
panels firing correctly.

Also: nested backdrop-filter double-blurring the same pixels, disabled
buttons dissolving to 0.35 so the ground showed through, a steering slider
Firefox never drew, all seven `transition: all` (worst on ~256 token chips),
hover transforms latching on touch, and the `!important` reduced-motion nuke
that breaks transitionend.

Colour literals in the rules: 89 -> 0. 52 unit tests, 42 e2e checks.

## 2026-08-07 (night, last) — Gemma runs; the gate check was wrong twice

Chasing the one model explicitly asked for turned up two bugs in a row, both
mine, both the same failure mode: *the observable result matched what I
expected, so I stopped looking.*

**Gemma-3-270m-it, verified on GPU:** 268M params, bf16, cuda:0, cold load
151.5s (575 MB download), generation `"The Eiffel Tower is located in Paris,
France."`, attention **18L x 4H**, 31 tokens, rows sum to 1.000, causal mask
holds. SAE correctly declined (GPT-2 only). That is 5 of the 6 current open
models now proven end-to-end.

**Bug 1 — a token is not a licence.** `hub.search` computed
`usable = (not gated) or bool(token)`. Gating is *per-repo* acceptance, so
every Gemma and Llama build was shown as available to an account that had
accepted neither. The loader refused them with a good error, but the picker
had already promised them. I had reported this as a working feature earlier
in the night — "search llama returns gated models showing gated OK with 0
locked rows" — which was the bug rendering, read as a success.

**Bug 2 — the fix for bug 1.** `_has_access` routed through `_api()`, which
does `json.load()`. The auth-check endpoint answers **200 with an empty
body**, so it raised, the `except` swallowed it, and the function returned
False for *every* repo. It passed a live test only because every gated repo
on hand was one this account genuinely could not reach — right answer, wrong
reason. Anyone who *had* accepted a licence would have seen their model
marked locked. Caught by probing the endpoint directly and noticing `gpt2`
answers HTTP 200 while the function said False.

The final proof is the discrimination, not the pass: on one fresh server,
`google/gemma-3-270m-it` reports gated **and usable** (and does run), while
`meta-llama/Llama-3.2-1B` reports gated **and not usable** (and 403s). Two
gated repos, opposite answers, both correct.

Locked rows now open the model's Hub page instead of doing nothing, and say
which step is missing: "sign in" when signed out, "accept licence" when
signed in.

**Test lesson worth keeping:** the first test could never have caught bug 2 —
it monkeypatched `_has_access`, the very function under test. The replacement
drives the real function against a fake empty 200. Both new tests were run
against reverted code first to confirm they fail without the fix. A test that
passes either way is not evidence.

**Still blocked, honestly:** OLMo-2-1B stalls at exactly 134 MB of 2.98 GB
across four attempts, on both HF transports and to a local disk. Not our bug;
the stall detector is what proved that rather than guessed it.

47 tests, 42 e2e checks.

## 2026-08-07 (night, later) — an audit against the field, and the bar that painted nothing

Ran 4 research agents over Apple's Liquid Glass spec, the interaction craft of
Linear/Raycast/Vercel/Stripe, the WAI-ARIA dialog pattern, and inspector UIs
(DevTools, Perfetto, BertViz). Then audited this frontend against that standard
across 5 dimensions, with an adversarial pass whose job was to *refute* each
finding. 13 survived, 12 were killed. Only the survivors were acted on.

**The worst finding was mine, from three hours earlier.** `--color-accent` does
not exist — the palette name is `--color-cobalt`. An undefined `var()`
invalidates the whole declaration, so the load-progress fill fell back to
transparent. I had verified that bar by reading its *width*. Width was never
the question. Fixed, and confirmed this time by reading computed
`backgroundImage`.

The same bug was already in the codebase: `--model` doesn't exist either, and
`ArcCanvas` read it for the arc stroke. Canvas ignores an unparseable
`strokeStyle`, so the attention arcs had been drawing in **default black**
instead of the attention blue since they were written. Now measured: 6,115
pixels of `rgb(26,96,209)`.

**Correctness**
- `ws/generate` had `try/finally` with no `except`. A generation that raised
  died in the worker thread, the `finally` posted the sentinel, and the browser
  was told `"done"` — CUDA OOM and unsupported architectures arrived as
  successful *empty answers*. The new test was run against the reverted code
  first to confirm it actually fails without the fix.
- A failed Hub search left the picker on "searching…" forever (`models === null`
  is the loading sentinel and the catch never cleared it).
- The attention panel had no loading or failure state, despite a first fetch
  that runs a full `output_attentions` pass.

**Access** — keyboard focus was invisible app-wide: `all: unset` on `.model-row`
resets `outline-style` in the *author* origin, which outranks the UA focus ring,
and nothing defined a replacement. Two component `outline: none` rules at
(0,2,0) were then quietly beating the new global rule; they now suppress the
ring for pointer focus only. The picker sheet was a plain div — now a real
dialog (role, `aria-modal`, initial focus, Tab trap, scroll lock, focus restored
to the opener). Attention and feature chips were pointer-only spans, so the arc
view and the whole SAE workflow had **no keyboard path at all**.

**Craft** — `ArcCanvas` drew at 1 device pixel per CSS pixel; thin arcs are the
panel's entire payload and were being upscaled on every retina screen. The
feature panel discarded the `argmax` the API already returns, so in a 256-token
strip the one chip worth seeing was unfindable. "sign out" composited to 3.32:1
through `opacity`, under AA, and it is the only way to sign out.

**Two bugs in my own fix**, found by testing rather than reasoning: React's
`autoFocus` fires during commit, so the effect captured the *search input* as
"the opener" and Esc restored focus to a dead node (body). And `onClose` is an
inline arrow, so depending on it re-ran the modal effect on every parent render
and would yank focus mid-keystroke.

**And one found by simply looking at the panel** rather than its numbers: the
attention strip could grow but never shrink. `.attn-inner` is
`width: max-content` and the canvas is its widest child, so measuring
`row.scrollWidth` while the canvas still held the previous generation's width
just returned that width again. A 23-token generation was rendering into a
12,645px box. Reproduced the sequence, fixed, re-verified: 267 tokens
(11,207px) → 5 tokens (741px), arcs still painting.

44 unit tests, 42 e2e checks, every visual claim checked against computed
styles or canvas pixels.

## 2026-08-07 (night) — the model picker meets real models

Ran five current open models through the actual HTTP API on the 4060, and the
run found more than the code review did.

**Verified on GPU (bf16, cuda:0), not predicted:**

| model | params | load | shape | attention rows | causal |
|---|---|---|---|---|---|
| Qwen/Qwen3-0.6B | 596M | 10.0s | 28L x 16H | 1.001 | yes |
| Qwen/Qwen2.5-0.5B-Instruct | 494M | 7.9s | 24L x 14H | 1.000 | yes |
| HuggingFaceTB/SmolLM2-360M-Instruct | 362M | 6.1s | 32L x 15H | 1.002 | yes |
| gpt2 | 124M | — | 12L x 12H | — | yes (+ SAE, steering) |

OLMo-2-1B is untested: its download stalls from this network on both the xet
and plain transports, and a probe straight to a local disk stalls identically.
Not our bug — but the stall detector below is what proved that, instead of me
guessing.

**What the run surfaced:**

- **Silent loads.** Minutes of nothing but the word "loading". `/api/model/progress`
  now reports stage + real bytes. Determinate bar when the size is known,
  indeterminate sweep when it isn't — a fake percentage is worse than none.
- **Byte counting is not obvious.** Three cache layouts exist and I hit all
  three; take the max of `blobs/` and `snapshots/`, never the sum. And size the
  download from a whitelist of what `from_pretrained` fetches — blacklisting odd
  formats left a fully-cached gpt2 reporting 26%, because gpt2 ships tflite,
  rust, h5 and flax copies of itself.
- **Dead downloads don't raise, they just stop.** Watched one sit at 128 MB of
  3.0 GB indefinitely. Called out after 45s now.
- **Qwen3 leaks `<think>` into the output.** Reasoning models stream a
  scratchpad. It gets its own collapsible block — on an introspection tool the
  model's working is the point, so hiding it would be the wrong fix.
- **The page scrolled sideways to 7859px.** `main` is a grid; grid items default
  to `min-width:auto` (= min-content), so a panel holding a 194-token attention
  strip grew to 7813px and dragged the whole layout with it. `min-width:0`.
  Every generation of any length hit this.
- **The picker forgot which model was loaded** across a reload, so Generate
  silently swapped models. It adopts the live one now.

**Environment, the hard way:** DriveFS truncated `typescript/package.json` to
zero bytes and refuses junctions, so `node_modules` cannot live beside the
source here. `scripts/build_frontend.py --work C:/build/modelmri` builds off the
synced drive — `npm ci` is 3s there against minutes on J:.

Two e2e checks "failed" until I found the cause: a uvicorn started before the
edit was still holding :5900, and my kill filter had missed it. Kill by port,
then confirm the new route answers, before believing any e2e result.

42 unit tests, 42 e2e checks, browser-confirmed at 194 tokens.

## 2026-08-07 (late) — v0.4 verification + hosted demo
- **`tests/e2e_check.py`**: exercises every feature against a live server (real models, real SAEs, real robot frames). **40/40 pass** — session/static/no-cache header/bundle, model discovery incl. Ollama-off path, load+generate, attention (rows sum to 1.000) + 422s, SAE load/features/steer/restore-exactly, traces import/list/get/404/422, VLA episodes/frame/load/analyse/heatmaps + "sharpens with depth" assertion. Run before every release.
- **Hosted demo shipped**: `scripts/bake_demo.py` captures real responses from a live server into `frontend/public/demo/*.json` (70 KB total); `VITE_DEMO=1` builds a static bundle whose `fetch` is patched once in main.tsx to serve those payloads, so every call site is identical to the real app and the demo can't drift. WS streaming is replayed word-by-word. `.github/workflows/pages.yml` deploys to GitHub Pages.
- **Bug caught by testing the demo rather than assuming:** the steering A/B rendered baseline text in BOTH cards — the demo prompt handler ignored steering state. Fixed by mirroring `steerActive` in demo.ts. Would have shipped a demo whose headline feature visibly does nothing.
- Demo verified in a browser against the static bundle: generation streams, 23 token chips, 2589 arc pixels, feature rows (#974 @ 60.9), A/B now differs (Paris -> San Diego), VLA frame + 153-px heatmap, layer slider changes the map (13k -> 34k at L9), scrubber returns correct frames (ask 54 -> t=54 @ 5.4s).
- **Perf finding:** cold gpt2 load measured 523s, reload 3.7s. Cause is the Drive-backed cache: J: is Google Drive File Stream, so a 1.1 GB model is *streamed from the cloud* on first touch, then cached locally. Not a product defect (a normal user's HF cache is on local disk) but a real consequence of the storage move — worth knowing before demoing cold.

## 2026-08-07 (late) — v0.4 VLA MODE: inside a real robot policy
- Recon workflow (4 parallel scouts) verified everything empirically before a line was written: AV1 decode 0.02-0.05s, PushT parquet 1.4MB/0.74s, and the blocker — **lerobot pins torch<2.12/numpy<2.3 but the venv runs torch 2.12.1/numpy 2.5.1**, so installing it would downgrade the working LLM path.
- Decision: **read the dataset directly** (pyarrow + pyav, no lerobot) and **lift the vision tower straight out of `lerobot/smolvla_base/model.safetensors`** — 197 tensors under `model.vlm_with_expert.vlm.model.vision_model.`, loaded into `SmolVLMVisionTransformer` with 0 missing. These are the policy's real weights, not a stand-in.
- Gotchas paid for: PushT's cache ref is `v3.0` (assuming `main` breaks discovery); dataset lives under `$HF_HOME/lerobot/hub`, models under `$HF_HOME/hub`; sdpa silently returns `attentions=None` → must force `_attn_implementation="eager"`; raw attention is [1,12,1024,1024] ≈50MB/layer → reduce to per-head 32×32 inside `no_grad`.
- Shipped: `vla_data.py` (episodes/state/action/frames), `vla.py` (VLAHandle), 7 endpoints, `VLAPanel` + `FrameCanvas` (scrubber + heat overlay + layer slider + stale badge), `vla-lite` extra. 25 tests.
- **Verified numbers:** 206 episodes · frame decode 60ms · tower load 5.2s · analysis 1.7s · heatmap paints 156 samples in the UI · **attention concentration rises with depth (top-5% mass 27% → 56% → 60% across layers 0/6/11)** — the expected diffuse→focused pattern, measured on real robot frames.
- Honest scope: this is the perception half. The action expert needs lerobot in a separate venv (`full` mode, designed, not built). The UI says so.

## 2026-08-07 (night) — blank-page bug + verification lesson
- Owner reported the whole app blank (only the header pill rendered). Cause: `AsciiField` measured its PARENT then wrote its OWN style size — canvas grows → hero grows → ResizeObserver refires → unbounded loop. Page inflated to thousands of px of empty canvas; all panels pushed off-screen.
- Fix: CSS owns the canvas box; JS only syncs the pixel buffer (no style writes, no-op when unchanged), observes the canvas itself, repaints on resize.
- **Verification lesson (permanent):** DOM-presence checks are not visibility checks. From now on, UI verification must assert LAYOUT — document.body.scrollHeight is sane, key elements' getBoundingClientRect().top is inside the first viewport, canvas box has a fixed expected height. That is how a blank page slipped past "all panels present".
- Verified after fix: scrollHeight 1283 (was runaway), canvas 200px stable, 900 glyphs painted, headline at y=319, Generate at y=687.

## 2026-08-07 (evening) — v4 "VANTAGE PAPER" + one-click Generate + any-model support
- **Root-caused the "Generate isn't working" report:** after any server restart the model unloads, leaving a silently disabled CTA. Fixed properly: Generate now auto-loads the selected model first (status shown), then streams. Verified from a cold server: one click → auto-load → 1,073 chars streamed.
- **Design v4 "Vantage Paper"** — LIGHT theme, straight from the owner's saved poistudio VANTAGE recipe: warm paper #f6f4ee, cobalt #2743e0, white hairline plates, centered editorial headline, the ASCII field recolored cobalt-on-paper, and the actual **Switzer** variable font the recipe names (self-hosted from Fontshare, license bundled).
- **Any model, two sources:** HF combo input with datalist fed by curated + a new local-cache scanner (GET /api/models/local — found all 10 cached models incl. VGGT/SAM3/SmolVLA) — plus **Ollama mode** (GET /api/ollama, load source="ollama", NDJSON streaming via stdlib): run any open model as text; UI states clearly that internals need HF. Graceful when Ollama is off.
- **release.yml workflow:** tag → build frontend into wheel → verify assets inside → attach to GH release.
- 19 tests. Django/Postgres formally dropped per owner (concept fit).

## 2026-08-07 (later) — CATCH-UP SPRINT: design v3 + Agent Mode v0.3
- **Design v3 "editorial scanner"**, grounded in the owner's actual X bookmarks (viewed via his Chrome): poistudio ASCII-dither art, brrranding condensed wordmarks, magenta-on-black pixel craft, Swiss spec labels. Tailwind CSS v4 (CSS-first), bundled Archivo Black, flat hairline plates, one electric magenta, and the signature `AsciiField` — a live ASCII-dither canvas (10fps, frame-1 sync paint, reduced-motion static). Stack note: kept FastAPI+SQLite over requested Django/Postgres — local-first pip install is the product; documented rationale.
- **Agent Mode (v0.3)**: `modelmri/traces.py` (SQLite WAL store, trace/step schema, ~/.modelmri/traces.sqlite), `modelmri.record` subpackage (trace ctx, nesting steps via contextvars, instrument_anthropic monkeypatch, POST-or-file delivery, never crashes host), endpoints (import/list/get), AgentsPanel UI (trace list, lane timeline colored by kind, error glow, step inspector w/ IN/OUT). `examples/record_demo.py` ships a realistic 17.1s failing run.
- Tests 15/15. Browser-verified: 10 blocks / 2 lanes render; clicking the error block shows "git push · step 8 · FAILED · Permission denied (publickey)" — kill-demo 3 is real.
- To extract at release: `modelmri-record` as its own PyPI dist (reserve name!).

## 2026-08-07 — Back after a month. Full feature audit, two hangs made impossible, lockfile added.
- Owner reported "no generated answer visible, only attention stats." Root causes found and fixed:
  1. `index.html` served with no cache headers while each deploy PURGES old hashed bundles → a stale cached page half-breaks. Fixed: `Cache-Control: no-cache, must-revalidate` on `/` (verified in response headers).
  2. A WS generation observed hanging forever with zero tokens (streamer blocks if the generate worker dies). Two-sided hardening: `TextIteratorStreamer(timeout=180)` server-side; 90s no-token watchdog + onclose handling client-side — the UI can no longer spin forever.
- **Lockfile lesson:** a month of dependency drift happened silently because `uv.lock` was gitignored (bad week-0 call). Reversed: `uv.lock` committed (78 packages pinned).
- Environment survived the break: registry env vars now inherited by new sessions (UV_PROJECT_ENVIRONMENT, HF_HOME), HF symlink intact, venv rebuilt in one `uv sync`.
- FULL browser audit, all green: generation output visible (257 pieces · 9.1s · 1,055 chars), attention arcs paint (4,305 px on pin), SAE loads (24,576 feats), token→features works (·Paris → **#974 @ 60.9 — the same feature as July**), heat view (267 chips), steering A/B reproduces the kill demo (" Paris, France." → " San Diego…" at #974 @ -40, reversible).

## 2026-07-08 — Week 1, day 6: STORAGE MIGRATION + design system v2 "scanner glass"
- Everything moved off C: per owner request: repo now at `J:\My Drive\Claude_Experiments\special\ModelMRI`, HF model cache (21.8 GB incl. other projects' models) at `special\models\huggingface`. **C: freed 20.7 GB** (72.6 → 93.3).
- DriveFS lessons (now standing knowledge): junctions/symlinks CANNOT be created on DriveFS; npm .bin shims break on it → venv lives at `C:\venvs\modelmri` (UV_PROJECT_ENVIRONMENT, registry-persisted), frontend builds in `C:\venvs\mri-build` temp and deploys back to J:. Old default HF path symlinked → J: so every process resolves models without env vars.
- Verified end-to-end from new locations: 11/11 tests, gpt2 loads from J: cache in 9.8s (no re-download), server serves the app.
- Design system v2: aurora ground (drifting radial washes + masked dot grid), orbiting scanner mark, gradient-ink wordmark, gradient-border glass panels (per-section tint), segmented model picker, shine-sweep buttons, custom violet slider, feature bars with grow-in, section headers with glow dots + fading rules. All motion on one easing; reduced-motion kills everything.
- Verified: computed styles confirm glass (blur 22px sat 1.5) + mark live; page interactive. (Preview screenshotter times out on infinite animations — page itself healthy; owner should eyeball localhost:5900.)

## 2026-07-08 — Week 1, day 5: FEATURES UI + liquid-glass design system. "We moved the Eiffel Tower to Berlin."
- Design system v1: liquid glass (backdrop-blur panels, layered radial-glow background, inset borders), per-section palettes (teal brand / blue attention / violet features), 200-300ms eased micro-animations, reduced-motion respected. Bar: design.google / Apple.
- FeaturesPanel shipped: model picker (Qwen chat / GPT-2 SAE) → generate → load SAE → click any token → its top-8 features with bars → click a feature → per-token heat view → steering slider → one-click deterministic A/B with side-by-side glass cards (always leaves steering cleared).
- Browser-verified end to end: GPT-2 sampled an Eiffel-in-Berlin hallucination; clicked "·Berlin", top feature #12884 (51.0); steered +40 → baseline " Paris, France." vs steered " Berlin, Germany." — amplifying the Berlin concept relocates the tower. Screenshot taken.
- Standing rules recorded (Blueprint/06): always share the localhost URL; Chrome posting on request; Gemini Pro (Nano Banana Pro / Veo) for premium assets; premium design bar.
- Next: GIF-ready polish + agent mode (v0.3), or Gemini-generated brand assets for README.

## 2026-07-08 — Week 1, day 4: SAE FEATURES + STEERING (backend). We turned off "Paris".
- New `modelmri/saes.py`: loads SAELens-format SAEs straight from HF (cfg.json + safetensors) — no sae-lens dependency chain. Default: jbloom/GPT2-Small-SAEs-Reformatted @ blocks.8.hook_resid_pre (24,576 features).
- Runtime: chat-template fallback for base models (GPT-2 has none), residual capture via forward_pre_hook, per-token feature computation (cached), single-feature steering (adds scale × unit decoder direction to the residual stream during generation, hook removed in finally).
- Endpoints: POST /api/sae/load, GET /api/sae, GET /api/features/summary, GET /api/features/{id}, POST/GET /api/steer. 11 tests green.
- VERIFIED END-TO-END (all real numbers):
  - Features are consistent: feature 1066 fires on both " Tower" occurrences, 19941 on both " E"s, 974 on " Paris" (60.9), 7310 on " France" (56.0).
  - THE steering A/B: baseline greedy → " Paris, France." · steer 974 at -40 → " San Diego, and is located in the San Diego State University" · clear → byte-identical " Paris, France." Deterministic, reversible, mechanistic.
- Next: FeaturesPanel in the React frontend (token → top features → steering slider → side-by-side steered output).

## 2026-07-08 — Week 1, day 3 (later): v0.1.0 RELEASE PREP + Day-3 post live
- Day-3 X post published (the "Paris attends to capital/France" find).
- Version bumped 0.1.0a1 → 0.1.0. README gains the pip install path.
- RELEASE-KILLING BUG caught by verification: the wheel had ZERO frontend assets — hatchling skips VCS-ignored files and `modelmri/static/app/` is gitignored. Fixed with `force-include` on both wheel and sdist (sdist matters: uv builds the wheel FROM it). Anyone pip-installing would have gotten a backend with no UI. Verify-before-ship pays again.
- Full release gate passed: wheel contains index.html + JS + CSS; clean-venv install → `modelmri 0.1.0` → server up → root serves the React app → assets 200.
- Tagged v0.1.0 + GitHub release. PyPI publish awaits the token (user action).

## 2026-07-08 — Week 1, day 3: REACT FRONTEND
- Real frontend shipped: React 18 + Vite + TypeScript (strict), no component libs, no state libs — 5 components, 150KB JS (49KB gz), builds in <1s.
- `npm run build` emits into `modelmri/static/app/`; FastAPI serves it at `/` (falls back to the legacy single-file page when no build exists). Built assets are NOT committed — built at release time.
- Dev loop: `npm run dev` on :5173 proxies /api + /ws to the Python backend on :5900.
- Verified in a real browser (automated): loaded model → streamed 51 pieces in 3.9s → attention panel appeared (24L × 14H × 90 tokens, head fetch 0.26s) → pinned the generated "·blue" token → arcs rendered (thick short-range + long-range sweep to early context). Screenshot taken.
- Divergence from blueprint: skipped tailwind + zustand for now — plain CSS on the established palette and lifted useState are simpler at this scale. Revisit when Agent Mode adds cross-cutting state.
- Next: WebGL grid/fabric view OR v0.1.0 PyPI release + GIF. Release first — ship what works.

## 2026-07-08 — Week 1, day 2: ATTENTION IS VISIBLE
- Attention capture shipped: model now loads with `attn_implementation="eager"` (SDPA/flash never materializes attention weights — the day's big lesson).
- After any generation, one full forward pass with `output_attentions=True` caches all layers (fp16, CPU); `GET /api/attention?layer=&head=` serves any head's S×S matrix instantly.
- Playground grew an attention inspector: token chips, layer/head selectors, hover → Canvas2D arcs to attended tokens (thickness = weight), click to pin. WebGL comes with the React frontend when we render full head grids.
- Verified with real numbers: 24 layers × 14 heads × 45 tokens, softmax rows sum to 1.000; at L12/H7 the generated " Paris" token attends to " capital" (0.098) and " France" (0.064 + 0.044) — the fig-3 demo moment, real.
- Also observed: massive attention sink on <|im_start|>/<|im_end|> — classic, and now *visible*.
- Next: record the GIF, then pip-installable v0.1 polish.

## 2026-07-08 — Week 1, day 1
- Backend v0.1 skeleton LIVE: `modelmri serve` → FastAPI on :5900.
- `ModelRuntime`: loads any HF causal LM (default Qwen2.5-0.5B-Instruct, 494M params), streams generation via `TextIteratorStreamer` in a worker thread.
- REST: `/api/session`, `/api/model/load`, `/api/model/prompt`. WS: `/ws/generate` (verified: 41 pieces streamed end-to-end).
- Built-in dark playground page at `/` (temporary until React frontend).
- Bug found by smoke test: Windows cp1252 console can't print `→` — crashed the CLI banner. Fixed to ASCII. (Good Day-2 post material.)
- Note: PyPI torch on Windows is CPU-only; 0.5B is snappy anyway (full generation in 2.5s). GPU via cu124 index or WSL2 when needed.
- Next: PyTorch forward hooks on attention layers → stream weights → WebGL arcs.

## 2026-07-08 — Week 0
- Name decided: **ModelMRI** ("an MRI machine for AI models"). Verified free on PyPI + npm.
- Repo created, skeleton committed: README with public roadmap, MIT license, CI (ruff + pytest + frontend build), Python package stub, npm stub.
- Dev environment confirmed: Python 3.12, uv, node 26, WSL2, RTX 4060.
- Next: publish name-reserving stubs to PyPI + npm, Substack setup, essay #1, Day-1 X post.
