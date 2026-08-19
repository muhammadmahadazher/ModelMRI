---
description: "Install ModelMRI with pip, load a model already on your machine, and run a first attention analysis. Includes the stdlib-only recorder for tracing agents without installing PyTorch."
---

# Getting started

## Install

```bash
pip install modelmri
```

That pulls PyTorch and Transformers, which is a large download. If you only
want to record agent runs and view them elsewhere, install the recorder
instead — it is stdlib only:

```bash
pip install modelmri-record
```

### Optional extras

```bash
pip install "modelmri[vla-lite]"   # robot datasets: av, pyarrow, pillow
```

## Run

```bash
modelmri serve
```

Open <http://127.0.0.1:5900>. Nothing is loaded yet — the first thing to do is
pick a model.

If someone sent you a `.mri` and you just want to look at it, there is nothing
to configure and no model to download:

```bash
modelmri open their-analysis.mri     # opens the viewer
modelmri inspect their-analysis.mri  # prints what it holds, and exits
```

That validates the file and opens it in about a third of a second — it serves
the bundled viewer from the standard library, so it loads no model and imports
no torch. See [sharing what you found](guides/attention.md#sharing-what-you-found).

## Your first look inside a model

1. **Pick a model.** The button already names one: the smallest model on your
   disk that the playground can run, so on a machine with a populated cache
   you can press Generate without opening anything. Open it to choose another
   — the picker starts on **On this machine**, which lists everything already
   on your disk: the HuggingFace cache, any folder with a `config.json` and
   weights, and any `.gguf` file. Nothing there? The button falls back to a
   name to download; switch to **HuggingFace** and search — `Qwen/Qwen3-1.7B`
   is a good first choice at about 1.5 GB.

2. **Type a prompt and press Generate.** The model loads automatically if it
   isn't already. A cold load shows real progress — stage, bytes, and a warning
   if the download stalls.

3. **Look at the attention.** A panel appears under the output. Hover any token
   to see what it attended to; click to pin it. Change layer and head to watch
   attention sharpen with depth.

4. **Load the SAE** in the features panel (GPT-2 only for now) and click a
   token to see which of its 24,576 features fired.

## Hardware

Run `modelmri doctor` before anything else — it is the shortest answer to "will
this work on my machine", and it measures rather than assumes:

```
  os          Windows 11 (AMD64)
  cpu         24 logical cores        ram   16.9 GB
  disk        191.6 GB free
  torch       2.11.0+cu128 (cuda 12.8)
  accelerator NVIDIA GeForce RTX 4060 Laptop GPU (cuda)   vram 8.6 GB
  precision   bfloat16

  Models up to roughly 3.2B parameters should fit.
```

There is no check during `pip install`, and that is deliberate rather than an
omission: a wheel is an archive and pip does not execute code from one. A
first-run check is also the better question to answer, because opening a shared
`.mri` needs neither torch nor a GPU while loading a 7B model needs both.

`MODELMRI_DEVICE=cpu` forces the CPU on any backend, which is the remedy when a
measurement needs float32 and the accelerator has chosen otherwise.


ModelMRI detects your accelerator and says which one it picked and why. The
badge in the top bar is not decoration — if it says CPU when you expected a
GPU, that is the tool telling you something.

| you have | what happens |
|---|---|
| NVIDIA GPU | CUDA, bfloat16 |
| AMD GPU | ROCm, bfloat16 |
| Intel Arc | XPU |
| Apple Silicon | MPS |
| none of the above | CPU, float32 — slower but correct |

If a model is too large for your VRAM, the load falls back to CPU rather than
dying, and tells you it did.

## Where models come from

`HF_HOME` decides where weights are cached. Set it to keep them off your system
drive:

=== "Linux / macOS"

    ```bash
    export HF_HOME=/mnt/big-disk/hf
    modelmri serve
    ```

=== "Windows"

    ```powershell
    $env:HF_HOME = "D:\hf"
    modelmri serve
    ```

To point the **On this machine** scanner somewhere specific:

```bash
export MODELMRI_MODELS_DIR=/mnt/big-disk/models
```

Otherwise it scans the directory you launched from, which is usually what you
want.

## Gated models

Llama and Gemma need a licence accepted per repository. Sign in with a
HuggingFace **read** token in the picker, and rows you cannot use are marked —
clicking one opens the page where the licence is accepted rather than failing
later.

The token is stored owner-only in ModelMRI's config directory — run
`modelmri where` to see exactly which file — and is sent to nowhere except
huggingface.co. ModelMRI never asks for a password. See
[SECURITY.md](https://github.com/muhammadmahadazher/ModelMRI/blob/main/SECURITY.md)
for what "owner-only" means on each platform — the same absolute form the
custom-models guide uses, because SECURITY.md lives outside the docs tree and
a relative link to it fails the strict build.
