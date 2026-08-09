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
modelmri open their-analysis.mri
```

That validates the file, starts the server with the analysis already open, and
opens a tab. See [sharing what you found](guides/attention.md#sharing-what-you-found).

## Your first look inside a model

1. **Pick a model.** The picker opens on **On this machine**, which lists
   everything already on your disk: the HuggingFace cache, any folder with a
   `config.json` and weights, and any `.gguf` file. Nothing there? Switch to
   **HuggingFace** and search — `Qwen/Qwen3-0.6B` is a good first choice at
   about 1.5 GB.

2. **Type a prompt and press Generate.** The model loads automatically if it
   isn't already. A cold load shows real progress — stage, bytes, and a warning
   if the download stalls.

3. **Look at the attention.** A panel appears under the output. Hover any token
   to see what it attended to; click to pin it. Change layer and head to watch
   attention sharpen with depth.

4. **Load the SAE** in the features panel (GPT-2 only for now) and click a
   token to see which of its 24,576 features fired.

## Hardware

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
huggingface.co. ModelMRI never asks for a password. See [SECURITY.md](../SECURITY.md)
for what "owner-only" means on each platform.
