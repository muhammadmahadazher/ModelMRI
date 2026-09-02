<h1 align="center">ModelMRI</h1>

<p align="center"><strong>Chrome DevTools for AI models and agents.</strong><br>
See inside any local LLM, VLM or robot policy while it runs.</p>

<p align="center">
  <a href="https://pypi.org/project/modelmri/"><img src="https://img.shields.io/pypi/v/modelmri?color=2563eb&label=pypi" alt="PyPI version"></a>
  <a href="https://pypi.org/project/modelmri/"><img src="https://img.shields.io/pypi/dm/modelmri?color=2563eb&label=downloads" alt="PyPI downloads"></a>
  <a href="https://pypi.org/project/modelmri/"><img src="https://img.shields.io/pypi/pyversions/modelmri" alt="Python versions"></a>
  <a href="https://github.com/muhammadmahadazher/ModelMRI/actions/workflows/ci.yml"><img src="https://github.com/muhammadmahadazher/ModelMRI/actions/workflows/ci.yml/badge.svg" alt="CI"></a>
  <a href="LICENSING.md"><img src="https://img.shields.io/badge/license-AGPL--3.0--only-blue" alt="AGPL-3.0-only licence"></a>
</p>

<p align="center">
  <a href="https://muhammadmahadazher.github.io/ModelMRI/"><b>▶ Live demo</b></a> ·
  <a href="https://muhammadmahadazher.github.io/ModelMRI/viewer/"><b>Open a .mri</b></a> ·
  <a href="https://muhammadmahadazher.github.io/ModelMRI/docs/"><b>Docs</b></a> ·
  <a href="https://modelmri.substack.com"><b>Build log</b></a>
</p>

```bash
pip install modelmri && modelmri serve      # → http://localhost:5900
```

<!-- Both themes. The GIFs used to be dark-only, which sold half the product
     to anybody who works in light mode. GitHub swaps these on the reader's
     own theme via the #gh-dark-mode-only / #gh-light-mode-only anchors. -->
<p align="center">
  <img src="docs/media/attention.gif#gh-dark-mode-only" alt="Hovering a token; attention arcs follow the cursor across the strip" width="820">
  <img src="docs/media/light/attention.gif#gh-light-mode-only" alt="Hovering a token; attention arcs follow the cursor across the strip" width="820">
</p>
<p align="center"><em>Hover any token — arcs show what it attended to. Every layer, every head. Light and dark.</em></p>

---

## Why

|  | |
|:--:|---|
| 🔍 | **You cannot debug what you cannot see.** Your model gave a wrong answer. The logs show the prompt and the output, and nothing in between. |
| 🖥️ | **It runs on your machine.** No cloud, no account, no telemetry, no API key. An 8 GB laptop GPU is the target, not a footnote. |
| 🧾 | **Every number carries a receipt.** What was measured, on which model revision, with how many forward passes, and what it does *not* prove. |
| 🚫 | **It refuses rather than guesses.** When a measurement would be misleading, you get a sentence explaining why — not a plausible number. |

---

## What it does

| | | |
|:--:|---|---|
| 👁️ | **Attention** | Every layer, every head, from a live forward pass — with a causal ranking of which heads actually mattered, scored in KL nats. |
| 🎯 | **Activation patching** | Where in the model the answer is decided, on a (layer × position) grid — each site checked against eight same-norm random draws. |
| ✏️ | **Counterfactuals** | The smallest edit to your prompt that makes it predict a token you name — controlled against random edits of the same size, so a flipped answer has to earn the word *finding*. |
| 🧠 | **Concepts** | SAE features, or contrastive steering vectors when no SAE exists. Find one, turn it off, watch the output change. |
| 🔭 | **Lenses** | Logit lens and a tuned lens trained on *your* text, scored on held-out KL and shown side by side. |
| 🤖 | **Robot policies** | What a VLA looked at, and — through a sidecar with its own environment — what it would *do*. |
| 🕵️ | **Agent traces** | The step where your agent died, as a timeline you can click into the model's internals from. |
| 🧩 | **Your own models** | Any `nn.Module`, TorchScript, or GGUF. Nothing hardcoded to one architecture. |
| 🔒 | **Weight scanning** | Looks inside a checkpoint for anything that executes on load, *before* loading it. |

<p align="center">
  <img src="docs/media/patching.gif#gh-dark-mode-only" alt="An activation patching grid filling in, site by site" width="820">
  <img src="docs/media/light/patching.gif#gh-light-mode-only" alt="An activation patching grid filling in, site by site" width="820">
</p>
<p align="center"><em>Activation patching: which (layer, position) actually decides the answer.</em></p>

---

## 60 seconds

```bash
pip install modelmri
modelmri serve                      # the UI, on localhost only
modelmri models                     # what is already on this disk
modelmri scan ./my_model            # is anything in there executable?
modelmri open finding.mri           # someone sent you a result
```

Python 3.10+ · Windows, macOS, Linux · AGPL-3.0-only (Community) · Apache-2.0 SDKs and `.mri` codec — see [LICENSING.md](LICENSING.md).

<p align="center">
  <img src="docs/media/picker.gif#gh-dark-mode-only" alt="The model picker listing models already on disk" width="820">
  <img src="docs/media/light/picker.gif#gh-light-mode-only" alt="The model picker listing models already on disk" width="820">
</p>
<p align="center"><em>It finds the models you already have — HF cache, plain folders, GGUF — before asking you to type anything.</em></p>

---

## What you can actually do with it

### 1. See what a token attended to

Type a prompt, watch it stream, then hover any token — arcs show which earlier tokens it looked at, scaled by attention weight, for any layer and head.

> The generated token `" Paris"` attends back to `" capital"` and `" France"`. The information was always there. Nobody was looking.

### 2. Ask which heads actually mattered

144 heat maps and no reason to open any of them is a browsing tool. **Rank heads** zeroes each head in a layer, runs the model again, and measures how far the answer moves — so the dropdown arrives ordered and the top head is already selected.

```
Qwen3-1.7B · "The capital of France is" · answer " Paris" · zero-ablation · bf16

Rank heads → L0 H3  KL 1.954   p(" Paris") 0.539 → 0.029   changes the answer
             L0 H9  KL 0.096   p(" Paris") 0.539 → 0.345
             L0 H1  KL 0.054   p(" Paris") 0.539 → 0.475
             L0 H5  KL 0.044
             L0 H11 KL 0.035
             18 forward passes · 5.6 s · noise floor 0.0
```

One head in the first layer carries most of it: removing L0 H3 alone takes
`" Paris"` from 0.539 to 0.029 and the model answers something else. The next
head down moves it by a twentieth as much.

The setup line is not decoration. A KL depends on the model, the prompt, the
dtype and the sequence — the same heads on one model give different numbers in
fp32 and over a long generation — so a figure quoted without them cannot be
checked by anyone.

**Three baselines, and how much they disagree is itself a property of the
model.** Zeroing a head is one choice; replacing it with its own mean is
another; replacing it with what it really computes on a different sentence
(`resample`, eight draws) is the only one that keeps the model on its own
distribution. On `Qwen3-1.7B` layer 0 they broadly agree — Spearman 0.81 to
0.91, and the top five differ by at most one head. On other architectures they
disagree far more. The panel reports that number so the choice of baseline is visible rather than
silently deciding the ranking.

Resampling also shows its own spread, because one donor is a coin flip. Head 3
on `Qwen3-1.7B` scored between **3.016 and 5.904** across the eight draws
around a median of 4.540 — a spread wide enough that a single draw could
have reported almost any number in it as the head's importance.

A ranking costs `n_heads + 2` forward passes; the whole model costs `n_layers × n_heads + 2`. That is the part that is portable — Qwen3-1.7B is 450 (28 layers x 16 heads). What a pass costs on *your* machine is not: measured on one RTX 4060 across sessions it moved between 12 and 71 ms for the same model, so the panel measures a layer on your machine and extrapolates from that rather than quoting a number from mine. One layer by default; the whole model only when told, with the estimate shown first.

Then ask **what changes?** on any ranked head and the panel subtracts the two runs — arcs in one colour where the model attends *more* without that head, another where it attends *less*. It opens at layer L+1, because removing a head cannot change its own layer's attention (that layer's input is unchanged), and a zero result says so rather than showing you an empty canvas.

Both sides are forward passes over the **same token sequence**, never two generations — sampling diverges, and chat templates insert 0, 8 or 29 leading tokens depending on the model, so subtracting two generations would align token 5 of one against token 5 of a different sentence.

It reports what it measured and nothing more. These are **not** each head's share of the prediction — the per-head scores do not sum to the whole layer's ablation, in either direction — and the ranking depends on what a removed head is replaced with, so the baseline is named on screen and both are offered. `head_dim` is read from the model rather than computed as `hidden_size // n_heads`, which is wrong by 2× on Qwen3 and would rank half-heads confidently.

### 3. Ask where in the model the answer is decided

Ablation says *what mattered*. It cannot say *where the thing is*. **Patching** takes two prompts that differ in one fact, moves an activation from the run that knows the answer into the run that does not, at every (layer, position), and reports how much of the difference comes back.

<p align="center">
  <img src="docs/media/patching.gif#gh-dark-mode-only" alt="The patching grid filling in row by row, then three tabs — residual stream, attention, MLP — each showing a different map of the same prompt" width="800">
  <img src="docs/media/light/patching.gif#gh-light-mode-only" alt="The patching grid filling in row by row, then three tabs — residual stream, attention, MLP — each showing a different map of the same prompt" width="800">
</p>

<p align="center"><em>Blue recovered the clean answer, red pushed it further away. Ringed cells were tested against chance.</em></p>

```
clean    "The Eiffel Tower is located in the city of"   ->  " Paris"
corrupt  "The Colosseum is located in the city of"      ->  " Rome"
```

Three grids, because *where* and *through what* are different questions — and they disagree. Measured on the same pair across three architectures:

| model | residual | attention | MLP |
|---|---|---|---|
| `Qwen2.5-0.5B-Instruct` | +0.999 · L23 · `of` | +0.478 · L21 · `of` | **+0.721 · L0 · `os`** |
| `gemma-3-270m-it` | +1.010 · L17 · `of` | +0.736 · L12 · `of` | **+0.483 · L3 · `osseum`** |
| `Qwen3-1.7B` | +0.967 · L3 · `el` | +0.651 · L20 · `of` | **+0.444 · L22 · `of`** |

Across the first three, the MLP peak sat on a **subject** token in an early
layer — `um`, `os` and `osseum` are all pieces of "Colosseum" — while the
attention peak sat on the **last** token, late. That was a tidy story, and
`Qwen3-1.7B` breaks it: its MLP peak is at **layer 22 on the final token**, and
its residual peak moved the other way, to **layer 3 on a subject token**.

Three models is not a result. The generalisation is left here with the model
that falsifies it rather than quietly rewritten, because the shape of this
table is the actual finding: *where a fact lives is a property of the
architecture, not of transformers*. Run it on yours — the panel does this on
whatever you have loaded, and the answer is not knowable from the first
three.

The score is **signed**, and it is the one ranking here that is not a KL: a patch can push the answer further away, and 5 of 132 sites did. It is also not capped at 1.0 — a single site can overshoot, and `gemma-3-270m-it` reads 1.010.

Each of the strongest sites is run again against **eight** same-norm random draws at the same site, not one, because one is a coin flip: at a single site the draws ran from −2.038 to +0.616 against a real recovery of +0.427.

Most casually-written pairs are refused, and both failures are invisible unless you are told — the prompts must tokenize to the same length (2 of 8 natural minimal pairs did not) and must predict different tokens (2 of 3 did not, making the denominator exactly 0). Both refusals print what to change.

### 4. Ask what would make it say something else

Three questions about one prompt, and they are not the same question:

| | asks | answers |
|---|---|---|
| **Rank tokens** | mask a word out — how far did the answer move? | necessity |
| **Anchors** | keep only these words, perturb the rest — does it hold? | sufficiency |
| **Counterfactual** | what do I write *instead* to get the answer I name? | reachability |

The first two describe the answer the model already gives. The third is a
recipe, and its output doubles as the corrupt half of a patching pair —
searched for against a named target and controlled, rather than typed.

A first-order estimate proposes substitutions; a real forward pass on every
shortlisted pair decides. The payload publishes **how often the estimate's top
choice actually won its step**, so the screen has to admit when it is not
helping — measured 0 of 4 on one Qwen3-1.7B run.

**A flipped answer is not a finding.** Every edit is scored against random
edits of the same size at the same positions, and at as many positions
anywhere. Measured on Qwen3-1.7B, steering *"The Eiffel Tower is in the city
of"* toward `" Rome"`:

| budget | passes | result |
|---|---|---|
| 3 edits, 24-wide shortlist | 78 | not reached — says which bound it hit |
| 4 edits, 64-wide shortlist | 267 | reached, and beat both controls 0/24 |

The edit that worked was `"The皇家cente虹桥LTR is in the city of"`. A
gradient-guided token search finds **adversarial** substitutions, not
paraphrases — that is a true property of the method and it is printed under
the result rather than hidden by it.

### 5. Find a concept and turn it off

Load a sparse autoencoder and ModelMRI shows the human-interpretable features firing on every token. Click one, drag the slider, and run a deterministic A/B:

```
gpt2 · layer 8 · jbloom/GPT2-Small-SAEs-Reformatted · FVU 0.0010 · 60.5 features/token

prompt                The Eiffel Tower is located in the city of
baseline               Paris, France.
feature #5856 @ -40    London's central London borough.
```

\#5856 is the **top-firing** feature on the final prompt token — activation
35.55, the one the panel already has selected — so this example is reachable by
following the instructions above rather than by knowing which number to type.
Same prompt, greedy decoding, no prompt tricks. We reached into layer 8 and
turned the concept down. Clearing the steer restores the baseline byte for byte.

**The SAE checks itself before it shows you anything.** An SAE fed the wrong
activation convention does not error — it returns features, in the right shape,
with plausible magnitudes, for a vector it never saw. So ModelMRI measures which
convention actually reconstructs (centered along `d_model` or not, `b_dec`
subtracted from the input or not), reports the fraction of variance unexplained,
and refuses to plot anything when no convention reconstructs. On the default SAE
that is the difference between **60.5** features firing per token and **7,491**,
and between an FVU of **0.0010** and **13,579**.

**And then it asks the question FVU cannot.** A reconstruction close to the
activations is not the same as a reconstruction the model can still predict
through: the directions carrying the residual stream's variance are not the
directions the next token depends on. So the panel also runs the model's own
cross-entropy on text you choose, again with the SAE's reconstruction spliced
in, and again with the activation replaced by a floor — and reports how much of
the lost loss came back. The floor is named on the card, because mean-ablation
and zero-ablation give different percentages for the same SAE, and all three
raw losses come with it so the choice can be undone. `3n + 2` forward passes,
quoted before the button runs. On the command line:
`modelmri sae fidelity --model MODEL --corpus notes.txt --floor mean_ablate`.

### 6. Find the step where your agent died

Two lines of `modelmri.record` around any agent run gives you a timeline: LLM calls, tool calls, subagents, each as a block. The failure glows. Click it for the exact input, output, tokens, and error.

```python
from modelmri.record import trace, step

with trace("fix-failing-tests"):
    step("llm_call", name="plan", input=prompt, output=answer, tokens_in=912)
    with step("subagent", name="auth-fixer"):
        step("tool_call", name="pytest", output="17 passed")
```

Or instrument automatically: `modelmri.record.instrument_anthropic()`.

### 7. Look inside a robot policy

This is the part nobody else ships. ModelMRI loads the **vision tower of the real SmolVLA checkpoint** and runs actual robot-camera frames through it, painting each image patch's attention back onto the frame. Scrub an episode, run the policy, drag the layer slider.

Measured on PushT frames — share of attention mass in the top 5% of patches:

| vision layer | concentration |
|---|---|
| 0 | 27% |
| 6 | 56% |
| 11 | 60% |

Early layers look everywhere; deep layers lock on. No robot hardware required — it reads public LeRobot datasets straight from disk.

A sweep can leave as **MCAP** (Foxglove, ROS tooling) or a Rerun **`.rrd`**. Both are optional installs and neither is a ModelMRI dependency. The `.rrd` path refuses while rerun's usage analytics are enabled — they are on by default, this tool has no telemetry, and it will not quietly make that untrue on your behalf; it names `rerun analytics disable` and writes the file once you have run it.

### 8. Debug a model you trained yourself

Everything above is transformer-shaped. This isn't. Point ModelMRI at your own `nn.Module` — an MLP, a small CNN, whatever you're training — and get a layer-by-layer map of one real forward pass.

```python
# my_net_adapter.py — the whole contract
def load():
    model = MyNet()
    model.load_state_dict(torch.load("checkpoints/best.pt", map_location="cpu"))
    return model
```

| layer | type | output | activation | |
|---|---|---|---|---|
| `fc1` | Linear | 8×64 | −31.20 ± 24.21 | |
| `act1` | ReLU | 8×64 | 0.10 ± 0.26 | **80% dead** |
| `fc2` | Linear | 8×32 | −1.02 ± 4.74 | |
| `act2` | Tanh | 8×32 | −0.12 ± 0.90 | **55% saturated** |
| `head` | Linear | 8×3 | −0.13 ± 0.44 | |

Dead units, saturated activations, and **the first layer where a `nan` appears** — statistics exclude non-finite values on purpose, so one bad number can't turn every row below it into `nan` and hide where it started.

A `state_dict` alone is refused, with the reason: it's weights without an architecture, and guessing one would produce a map that looks authoritative and describes a network you never trained.

### 9. Send someone the finding, not the model

You found the head. Now show a colleague — who does not have your GPU, your prompt, or 8 GB of spare disk.

<p align="center">
  <img src="docs/media/share.png#gh-dark-mode-only" alt="The attention panel's share control, with a note reading 'L8 H3 copies the subject token'" width="800">
  <img src="docs/media/light/share.png#gh-light-mode-only" alt="The attention panel's share control, with a note reading 'L8 H3 copies the subject token'" width="800">
</p>

That writes **one 54 KB file** holding the tokens, the attention, the generation and your note. No weights — it's an observation, not a checkpoint.

<p align="center">
  <img src="docs/media/viewer.png#gh-dark-mode-only" alt="The same analysis open in the browser viewer: replay banner, attention arcs from 'Amsterdam' back through the prompt" width="800">
  <img src="docs/media/light/viewer.png#gh-light-mode-only" alt="The same analysis open in the browser viewer: replay banner, attention arcs from 'Amsterdam' back through the prompt" width="800">
</p>

<p align="center"><em>The recipient opens it at <a href="https://muhammadmahadazher.github.io/ModelMRI/viewer/">the viewer</a> — nothing installed, nothing uploaded, the file is read in their browser.</em></p>

Locally it's the same page, served from the package by the standard library:

```bash
modelmri open 0000.mri     # ~0.3s — no torch, no model, no GPU
```

If you were sent several and want to know which is which, `inspect` prints one without opening anything:

```bash
modelmri inspect 0000.mri
```

```
0000.mri — 83.8 KB
  model         Qwen/Qwen3-1.7B
  size          1,721M parameters
  ran on        cuda:0 · bfloat16
  recorded      2026-08-18T08:38:51+00:00 by ModelMRI 0.11.0
  note          sweep row 0: The capital of France is

  tokens        21 (13 prompt)
  shape         28 layers x 16 heads
  attention     448 maps
                every layer and head
  patching      attn, mlp, resid
    clean       The Eiffel Tower is located in the city of
    corrupt     The Colosseum is located in the city of

  prompt        The Eiffel Tower is located in the city of
  answer         Paris
```

`--json` gives the same summary as a document, with the prompt and generation untruncated. Both take the same ~0.2s as `open`: a `.mri` is gzipped JSON, so nothing here needs torch.

A recording carries the **activation-patching trace** too, when one was run — so the causal finding travels with the file, not just the attention. Open a `.mri` that has one and the patching panel draws it, marked as recorded rather than measured on your machine. A recording that carries none says so instead of offering a button that can only refuse: patching means running the model again with an activation replaced, and a `.mri` holds activations rather than weights.

Every panel reads a recording through the same calls it uses for a live model, so the arcs, the layer/head dials and the token strip all behave normally. The status pill says `replay` and the footer says *recorded, not live*, so it can never be mistaken for your own run.

The browser viewer and the Python tool are checked cell-for-cell against the same file on every change ([tests/viewer_check.py](tests/viewer_check.py)) — a viewer that renders a *slightly* different matrix would be worse than no viewer, because nothing on screen would say so.

---

## Every command

Nine of them, and only one needs a GPU. Everything below reads what is already
on your disk — no model is downloaded and no server is started unless the
command says so.

| Command | What it does | Needs |
| --- | --- | --- |
| `pip install modelmri` | Install it. `modelmri[vla]` adds the robot-policy reader; `modelmri[dev]` adds the test suite. | — |
| `modelmri serve` | Start the app and open it at `localhost:5900`. This is the one that loads models. `--port`, `--host`. | a model |
| `modelmri models` | List every model on this machine, **and the ones that will not load, with the reason**. Instant. | — |
| `modelmri traces` | List agent runs recorded here, newest first. Instant. | — |
| `modelmri open FILE.mri` | Open an analysis somebody sent you, in a browser. No model, no GPU, ~0.3s. | — |
| `modelmri inspect FILE.mri` | Print what a `.mri` holds and exit — model, shape, what was captured, the prompt. `--json` for the lot. ~0.2s. | — |
| `modelmri diff A.mri B.mri` | **Compare two analyses of the same prompt** and exit non-zero when something moved. `--fail-over X` for CI. No torch — instant. | — |
| `modelmri sweep PROMPTS` | **Run one measurement over many prompts** and report each head as median, IQR, n and top-k rate instead of one number. `--metric`, `--layer`, `--jsonl`, `--out-dir`. | a model |
| `modelmri verify FILE.mri` | **Re-run the measurements in a `.mri` on this machine** and report, per number, whether it came back the same. `--json` for CI. Loads the model. | the file's model |
| `modelmri doctor` | What this machine can and cannot run, and why. Run it before you file a bug. | — |
| `modelmri where` | Every directory ModelMRI reads or writes, and the variables that move them. | — |
| `modelmri uninstall` | Remove everything ModelMRI has written here. `--models` takes the weights too. Asks first. | — |

**The two you will use most**

```bash
modelmri models        # what have I got, and why won't that one open?
modelmri serve         # look inside one of them
```

**Sending someone a finding**

```bash
modelmri inspect 0000.mri     # what is in this file?
modelmri open 0000.mri        # show me
modelmri verify 0000.mri      # do these numbers come back on my machine?
```

**Checking a finding instead of trusting it**

```bash
modelmri verify 0000.mri
```

```
0000.mri — measured on Qwen/Qwen3-1.7B
  file: bfloat16 on cuda:0    here: bfloat16 on cuda:0
  commit: 607a30d783df  (same weights)

  ✓ generation: reproduced
      greedy decoding produced the same 4 tokens.
  ✓ attention: reproduced
      all 144 stored head maps match. The worst, 6:9, is off by 2.00e-03
      against a 3.92e-03 tolerance.
  ✓ patching: reproduced
      all 3 grids match to 0.00e+00, inside a 0.00e+00 floor measured by
      running the same trace twice here.
  – head ranking: not verifiable
      this file records that a head ranking ran, but the `.mri` carries
      attention, patching and the generation — not the ranking itself.

  3 reproduced · 0 differ · 1 not verifiable
```

Three verdicts and no pass/fail, because bit-exact reproduction across two
machines is not achievable — kernel selection, cuDNN version and TF32 all move
the last digits. **Every tolerance above was measured on the machine running
the command**, never asserted from a constant: each check runs the same
computation twice locally and takes the spread, and for attention the file
supplies a second floor of its own, since it stores each map as uint8 against
that map's maximum. Exit 1 only for a real disagreement — a file this machine
cannot check is not a broken file.

No hosted platform can offer this. It can hand you its own assertion; it can
never hand you the re-run.

**One prompt is an anecdote**

```bash
modelmri sweep prompts.txt --model Qwen/Qwen3-1.7B --layer 0
```

```
heads over 5 prompts on Qwen/Qwen3-1.7B · baseline zero
  5 measured · 0 could not be measured

  head          median       IQR               range    n  top5
  L0H3         0.00492   0.00578  0.00198–0.01077      5  5/5 (100%)
  L0H12        0.00006   0.00004  0.00005–0.00013      5  5/5 (100%)
  L0H7         0.00004   0.00001  0.00004–0.00006      5  5/5 (100%)
  L0H2         0.00001   0.00000  0.00001–0.00002      5  5/5 (100%)
  L0H1         0.00001   0.00000  0.00001–0.00002      5  4/5 (80%)
  L0H0         0.05418   0.04382  0.03673–1.71640      5  3/5 (60%)
```

Read the fifth row. **L0H0 has a median of 0.054 and a maximum of 1.716** — it
carried one prompt almost entirely and did nothing on the rest. That is the
head worth looking at, and it is the head a mean would have buried. This is
what "a number measured once is a sample, not a property" looks like as
behaviour rather than as a line in a readme.

Three rules it enforces rather than documents: never a mean without a spread;
a prompt that could not be measured is a **row** carrying the reason, never a
gap; and a token-position sweep is never aggregated across prompts, because
position 3 is a different word in every one of them.

**Did my quantisation change the model?**

```bash
modelmri diff baseline.mri after-quantising.mri --fail-over 0.05
```

```
baseline.mri → after-quantising.mri

  = generation: same
  ≠ head ranking: changed
      the top 5 changed: L0H4 entered and L0H7 left. 12 of 12 heads moved
      past the 0.00e+00 noise floor.
  = attention: same
  ≠ patching: changed
      3 patching sites changed sign — a site that recovered the clean answer
      and now pushes away from it is a different causal story.
```

Exit 1, in the pull request that did it. Nothing else in the category has a
regression concept for model internals; the state of the art for this question
is a Reddit thread. It imports **no torch** — both sides are already measured
and comparing them is arithmetic — so it is a CI step you would actually add.
[The guide has a workflow you can paste.](docs/guides/regression-ci.md)

It refuses rather than guesses: two different prompts exit 2 instead of being
diffed into numbers that look like a regression, a sampled run is refused
because it differs for reasons that are not the model, and a section one file
lacks reports **"not comparable"**, never "same".

A `.mri` is one analysis without the model — attention, the logit lens, the
generation, and the activation-patching trace if you ran one. It is how you
show a colleague the head you found without asking them to download 8 GB.

**When something is wrong**

```bash
modelmri doctor        # can this machine run it at all?
modelmri where         # where did it put my stuff?
```

### Moving where things go

| Variable | Moves |
| --- | --- |
| `MODELMRI_MODELS_HOME` | where downloaded models land (default: wherever HuggingFace already puts them) |
| `MODELMRI_MODELS_DIR` | extra folders to search for your own models |
| `MODELMRI_HOME` | all of ModelMRI's own state, under one directory |
| `MODELMRI_TRACE_DIR` | where undelivered agent traces are written |
| `MODELMRI_DEVICE` | force `cpu`, `cuda`, `mps`, `xpu` |
| `HF_HOME` / `HF_HUB_CACHE` | HuggingFace's own cache, honoured as-is |

## Install

```bash
pip install modelmri              # core: playground, attention, features, steering, agents
pip install "modelmri[vla-lite]"  # + robot datasets (av, pyarrow, pillow)
pip install modelmri-record       # just the agent recorder — stdlib only, an 10.9 KiB wheel
modelmri doctor                   # what this machine can and cannot run, measured
modelmri serve
```

**Will it run here?** `modelmri doctor` measures your machine and says so — OS,
cores, RAM, free disk, the torch build, the accelerator it found and its
precision, and roughly how large a model fits. It exits non-zero when something
would stop a load, so it is scriptable. `modelmri serve` prints the one-line
version at startup.

```
  accelerator NVIDIA GeForce RTX 4060 Laptop GPU (cuda)   vram 8.6 GB   bfloat16
  Models up to roughly 3.2B parameters should fit.
```

Every figure is read off the machine at the moment you ask, and a number that
cannot be determined says "could not measure" rather than being invented. Note
this is a *first-run* check rather than an install-time one, deliberately: a
wheel is an archive and pip does not execute code from it, so there is nowhere
honest to put a check during `pip install`. It is also the better place for it,
because the same machine can be perfectly able to open a shared `.mri` and
unable to load a 7B model — and those are different questions.

From source:

```bash
git clone https://github.com/muhammadmahadazher/ModelMRI && cd ModelMRI
cd frontend && npm ci && npm run build && cd ..
uv sync && uv run modelmri serve
```

**Models.** Search HuggingFace, pick from what's already cached on your machine, or switch to **Ollama** and pull any model by name. (Ollama gives you text only — internals need a HuggingFace model, and ModelMRI says so rather than pretending.)

**Nothing downloads by surprise.** Every row shows its size before you click, and a download that cannot fit your disk is refused with both numbers rather than started. One that dwarfs your GPU asks first. Whatever is running, **Stop** actually stops it — the fetch happens in a child process precisely so it can be killed, and the half-written blobs are cleaned up. This exists because a click once began fetching 1.5 TB onto an 8 GB laptop with no way out but killing the server.

**GPU when you have one.** NVIDIA, AMD, Intel and Apple silicon are detected automatically and the badge explains its choice — including the common case where torch was installed as a CPU-only build, where it prints the exact command to fix it. CPU works fine too; a 0.5B model streams in a couple of seconds.

## API

The UI is a client of a plain HTTP API — script against it directly.

| | |
|---|---|
| `POST /api/model/load` | `{hf_id, source}` — `"hf"` or `"ollama"` |
| `WS /ws/generate` | stream tokens |
| `GET /api/attention` | `?layer=&head=` → tokens + attention matrix |
| `POST /api/sae/load` · `GET /api/features/summary` | SAE features per token |
| `POST /api/sae/fidelity` | how much of the model's own loss survives that SAE |
| `POST /api/steer` | `{feature_id, scale}` — clamp a concept during generation |
| `POST /api/traces/import` · `GET /api/traces/{id}` | agent traces |
| `POST /api/vla/analyse` · `GET /api/vla/attention` | robot-policy attention |
| `POST /api/custom/load` · `POST /api/custom/run` | inspect a model you trained yourself |

## Status

| | |
|---|---|
| Playground · streaming · any local model · Ollama | ✅ |
| Attention inspector | ✅ |
| Head ranking by ablation | ✅ |
| Compare two runs (signed attention diff) | ✅ |
| SAE feature browser + activation steering | ✅ |
| Agent trace timeline + step inspector | ✅ |
| Robot policy (VLA) attention over real episodes | ✅ perception |
| Custom models — adapters, TorchScript, layer map | ✅ |
| Shareable `.mri` sessions + zero-install browser viewer | ✅ |
| Download size guard + a Stop button that works | ✅ |
| VLA action expert (needs `lerobot`, separate env) | 🏗️ |
| Hosted zero-install demo | ✅ |

## Honest limits

- **Attention needs eager attention.** SDPA and FlashAttention never materialize the weights, so ModelMRI loads models with `attn_implementation="eager"`. Slower, but it's the only way to see anything.
- **SAE features need an SAE that exists.** They are trained per model, and public ones exist for only a handful — this build knows of four repositories — so there is none for most of what you will load, and no amount of code makes one appear. ModelMRI offers the one that matches your model, says plainly when there is none, and falls back to a logit lens, which needs nothing but the model.
- **Custom models get a layer map, not attention.** Attention and SAE features need a transformer; for an arbitrary `nn.Module` ModelMRI shows shapes, activation statistics and pathologies. Loading an adapter runs your Python — see [SECURITY.md](SECURITY.md).
- **VLA mode is the perception half.** SmolVLA's vision tower is real and loaded from the real checkpoint; the action expert needs `lerobot`, whose torch/numpy pins conflict with the core runtime, so it lives behind an opt-in extra rather than degrading everyone's install.

## How it compares

The parts of ModelMRI are not new. Attention visualization, causal ablation,
sparse autoencoders and agent tracing all have good tools already, and most of
them do their one thing better than ModelMRI does. What is unusual here is the
combination — model internals and agent traces, in one GUI, on your hardware,
with no notebook in between.

| | what it is | where ModelMRI differs |
|---|---|---|
| [BertViz](https://github.com/jessevig/bertviz) | attention visualization in a notebook | ModelMRI is a standalone app, and ranks heads by ablating them rather than only drawing them |
| [TransformerLens](https://github.com/TransformerLensOrg/TransformerLens) | a mechanistic-interpretability library: hooks, caching, patching | TransformerLens is more capable and more precise; you write Python. ModelMRI is a UI for the common questions, with no code |
| [nnsight](https://github.com/ndif-team/nnsight) | library access to internals, including remote large models | nnsight reaches models too big for your GPU; ModelMRI only runs what fits on your machine |
| [Neuronpedia](https://www.neuronpedia.org/) | hosted browser for SAE features | Neuronpedia has far richer feature data for the models it covers; ModelMRI runs an SAE against *your* prompt, locally, and steers with it |
| [SAELens](https://github.com/jbloomAus/SAELens) | training and analysing sparse autoencoders | ModelMRI consumes SAEs, it does not train them |
| [Langfuse](https://langfuse.com/) · [Phoenix](https://github.com/Arize-ai/phoenix) · [LangSmith](https://www.langsmith.com/) | LLM application observability — traces, prompts, cost | These are production observability platforms and much stronger at it. ModelMRI records a run so you can open it next to the model's internals |
| [promptfoo ModelAudit](https://www.promptfoo.dev/) | scans model files for malicious payloads | promptfoo scans a file you point it at; `modelmri scan` runs on the load path and **refuses** to load one |
| [Rerun](https://rerun.io/) · [Foxglove](https://foxglove.dev/) | robotics data viewers | Far better timelines and 3-D. Neither can tell you what the policy *attended to*, or what it would do on a frame you choose |

**The one thing nothing else does:** hold the recorder and the weights in one
process. Every observability platform stops at the API boundary; every
interpretability library has no agent traces. Joining a failing agent step to
the heads that moved the token needs both in memory at once.

Use TransformerLens if you are doing research and want precision. Use Langfuse
or Phoenix if you are running an agent in production and need dashboards,
retention and alerting. Reach for ModelMRI when you have a model on your
machine that is doing something you do not understand, and you want to look at
it now.

## Questions people ask

### What is ModelMRI?

An open-source, local-first tool for inspecting what a model is doing
internally while it runs: attention per layer and head, which heads carry the
prediction, which interpretable features fire, what changes when you turn one
off — plus a recorder that makes an agent run inspectable step by step.
`pip install modelmri && modelmri serve`, then open `http://localhost:5900`.

### Does it work with GPT-4, Claude, or Gemini?

Not for internals, and no tool can — closed API models do not expose attention
weights or activations to anyone outside the provider. ModelMRI needs weights
it can run, so internals mean a local HuggingFace model.

The **agent recorder is a different matter**: it wraps whatever your agent
calls, including hosted APIs, so you can record and inspect a run driven
entirely by Claude or GPT-4. `modelmri.record.instrument_anthropic()` does it
in one line.

### Do I need a GPU?

No. CPU works — a 0.5B model streams in a couple of seconds. NVIDIA, AMD,
Intel and Apple silicon are detected automatically when present, and if torch
was installed as a CPU-only build the badge says so and prints the command to
fix it.

### Which models work?

Any causal LM transformers can load with eager attention, from the HuggingFace
cache you already have, a plain folder on disk, or a search-and-download in
the app. The ones with a recorded end-to-end result are **GPT-2, Qwen3-0.6B,
Qwen2.5-0.5B-Instruct, SmolLM2-360M-Instruct and Gemma-3-270m-it** — the
[verified table](https://muhammadmahadazher.github.io/ModelMRI/docs/#verified-not-asserted)
lists what each one actually measured. Others should work and are not claimed
to have been checked: Llama-3.2-1B is deliberately absent because the
`meta-llama` repos are gated and returned 403 for the account used, and an
untested model in a list headed "supported" is just a guess in bold. Ollama
models work for text generation; internals need a HuggingFace model, and
ModelMRI tells you that instead of quietly showing you nothing. Non-transformer
models you trained yourself get a layer map with activation statistics.

### Is any of my data sent anywhere?

No. There is no cloud, no account and no telemetry. Models are downloaded from
HuggingFace if you ask for one you don't have; beyond that, nothing leaves the
machine. A `.mri` file you choose to share contains tokens, attention and your
note — never weights — and the browser viewer reads it client-side without
uploading it.

### What is a `.mri` file?

One ~54 KB file holding the tokens, the attention matrix, the generation and a
note you wrote. It exists so you can send a colleague the *finding* without
sending them 8 GB of weights or asking them to install anything — they open it
at [the viewer](https://muhammadmahadazher.github.io/ModelMRI/viewer/) in a
browser. `modelmri open file.mri` does the same locally in about 0.3s, with no
torch and no GPU.

### How is this different from TransformerLens or BertViz?

BertViz draws attention; ModelMRI also measures which heads matter by removing
them. TransformerLens is a library you write code against and is more precise
and more flexible than this; ModelMRI is a UI for the questions people ask
most, and adds SAE steering, agent traces and robot policies in the same
window. See [How it compares](#how-it-compares).

### Is it production-ready?

No, and the package says so — it is classified alpha. It is a debugging and
research tool, not infrastructure. The measurements it reports are tested, but
the API surface still moves between minor versions; see the
[changelog](CHANGELOG.md).

### Can it check a downloaded model for malicious code?

Yes. `modelmri scan <path>` reads the pickle opcode stream **without
unpickling it** and reports anything that executes on load — `os.system`,
`eval`, decode-then-execute chains, embedded executables, zip bombs. It exits
non-zero on a finding, so it works as a CI gate.

It runs automatically on the load path, so a dangerous file is refused rather
than reported after the fact. Three verdicts: `safe`, `dangerous`, and
`unscanned` — a format it could not read is never called clean.

A `.bin`, `.pt`, `.pth` or `.ckpt` is a pickle, and unpickling is not parsing:
the payload runs before a single tensor is read. `safetensors` has no
mechanism to execute anything, and ModelMRI tells you when a repository
publishes one.

### Can it tell me what a robot policy would *do*, not just what it looked at?

Yes, through a sidecar that holds the action expert in its own process and
virtual environment — because lerobot pins torch and numpy hard enough that
installing it beside ModelMRI breaks ModelMRI.

`modelmri policy install` builds it; `modelmri policy start` runs it. Then you
get predicted-versus-recorded actions across an episode, an instruction-swap
test measured against the policy's own sampling variance, and input-stream
knockout.

It refuses to overlay a policy's actions on a dataset's recorded ones when the
two are in different units — which is the common case, and the plausible-wrong
chart everything else draws.

### Why does it refuse to show me things?

Because a measurement that would be misleading is worse than no measurement.
Some real examples it will refuse:

- an SAE whose activation convention does not reconstruct your model
- a patching pair whose two prompts predict the same token, making the
  denominator zero
- an instruction-swap test on a deterministic policy, where the reference
  spread is exactly zero
- two action curves whose units were never published

Every refusal is a sentence naming what to change.

### How do I cite it?

[CITATION.cff](CITATION.cff) is in the repository root — GitHub's "Cite this
repository" button reads it. Please include the version, because the measured
figures in this README belong to the release that produced them.

## Contributing

Issues and pull requests are welcome. One rule runs the whole repository:
**don't ship a measurement you haven't verified.** A visualization that looks
plausible and is wrong is worse than none, because interpretability is exactly
the domain where nobody has an independent way to notice.

- [Contributing guide](CONTRIBUTING.md) — setup, quality gates, and the three
  bugs that made that rule
- [Code of conduct](CODE_OF_CONDUCT.md)
- [Security policy](SECURITY.md) — trust model, credential handling, and what
  loading a model actually executes
- [Support](SUPPORT.md) · [Changelog](CHANGELOG.md)

## Built in public

Notes, mistakes, and what broke: [modelmri.substack.com](https://modelmri.substack.com)

AGPL-3.0-only © 2026 Muhammad Mahad Azher · SDK packages and the .mri codec Apache-2.0 · [LICENSING.md](LICENSING.md)
