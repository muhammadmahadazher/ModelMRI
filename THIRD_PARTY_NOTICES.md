# Third-party notices

What ModelMRI ships that it did not write, what it loads that it does not
ship, and the rule that keeps the two apart. The project's own licensing is in
[LICENSING.md](LICENSING.md); this page is about everyone else's.

## Dependencies

<!-- generated-section: dependencies
     Written by scripts/third_party_notices.py from uv.lock and
     frontend/package-lock.json. Do not edit by hand; run the script. -->

### Python

106 packages in `uv.lock`, every extra included. Licenses are read from the
installed metadata of the environment the script ran in; a package that was not
installed there says so rather than guessing.

| package | version | license | homepage | relation | shipped? |
|---|---|---|---|---|---|
| `accelerate` | 1.14.0 | Apache Software License | https://github.com/huggingface/accelerate | direct: modelmri | installed with the wheel, not inside it |
| `annotated-doc` | 0.0.4 | MIT | https://github.com/fastapi/annotated-doc | transitive | installed with the wheel, not inside it |
| `annotated-types` | 0.7.0 | MIT License | https://github.com/annotated-types/annotated-types | transitive | installed with the wheel, not inside it |
| `anyio` | 4.14.1 | MIT | https://github.com/agronholm/anyio | transitive | installed with the wheel, not inside it |
| `attrs` | 26.1.0 | MIT (PyPI metadata) | https://pypi.org/project/attrs/ | dev group only | never installed with a wheel (development tooling) |
| `av` | 17.1.0 | BSD-3-Clause (PyPI metadata) | https://github.com/PyAV-Org/PyAV | direct: modelmri | installed with the wheel, not inside it |
| `boolean-py` | 5.0 | BSD-2-Clause (PyPI metadata) | https://github.com/bastikr/boolean.py | dev group only | never installed with a wheel (development tooling) |
| `certifi` | 2026.6.17 | Mozilla Public License 2.0 (MPL 2.0) | https://github.com/certifi/python-certifi | transitive | installed with the wheel, not inside it |
| `charset-normalizer` | 3.5.0 | MIT | https://pypi.org/project/charset-normalizer/ | transitive | installed with the wheel, not inside it |
| `click` | 8.4.2 | BSD-3-Clause | https://github.com/pallets/click/ | transitive | installed with the wheel, not inside it |
| `colorama` | 0.4.6 | BSD License | https://github.com/tartley/colorama | transitive | installed with the wheel, not inside it |
| `cuda-bindings` | 13.3.1 | LicenseRef-NVIDIA-SOFTWARE-LICENSE (PyPI metadata) | https://github.com/NVIDIA/cuda-python | transitive | installed with the wheel, not inside it |
| `cuda-pathfinder` | 1.5.6 | Apache-2.0 (PyPI metadata) | https://github.com/NVIDIA/cuda-python | transitive | installed with the wheel, not inside it |
| `cuda-toolkit` | 13.0.3.0 | no license metadata (PyPI) | https://developer.nvidia.com/cuda-toolkit | transitive | installed with the wheel, not inside it |
| `diffusers` | 0.39.0 | Apache Software License | https://github.com/huggingface/diffusers | direct: modelmri | installed with the wheel, not inside it |
| `exceptiongroup` | 1.3.1 | MIT License (PyPI metadata) | https://github.com/agronholm/exceptiongroup | transitive | installed with the wheel, not inside it |
| `execnet` | 2.1.2 | MIT | https://execnet.readthedocs.io/en/latest/ | dev group only | never installed with a wheel (development tooling) |
| `fastapi` | 0.139.0 | MIT | https://github.com/fastapi/fastapi | direct: modelmri | installed with the wheel, not inside it |
| `filelock` | 3.29.6 | MIT | https://github.com/tox-dev/py-filelock | transitive | installed with the wheel, not inside it |
| `fsspec` | 2026.6.0 | BSD-3-Clause | https://github.com/fsspec/filesystem_spec | transitive | installed with the wheel, not inside it |
| `gguf` | 0.19.0 | MIT License (PyPI metadata) | https://ggml.ai | direct: modelmri | installed with the wheel, not inside it |
| `greenlet` | 3.5.4 | MIT AND PSF-2.0 | https://greenlet.readthedocs.io | dev group only | never installed with a wheel (development tooling) |
| `h11` | 0.16.0 | MIT License | https://github.com/python-hyper/h11 | transitive | installed with the wheel, not inside it |
| `h5py` | 3.16.0 | BSD-3-Clause | https://www.h5py.org/ | direct: modelmri | installed with the wheel, not inside it |
| `hf-xet` | 1.5.1 | Apache-2.0 | https://github.com/huggingface/xet-core | transitive | installed with the wheel, not inside it |
| `httpcore` | 1.0.9 | BSD-3-Clause | https://www.encode.io/httpcore/ | transitive | installed with the wheel, not inside it |
| `httptools` | 0.8.0 | MIT | https://github.com/MagicStack/httptools | transitive | installed with the wheel, not inside it |
| `httpx` | 0.28.1 | BSD License | https://github.com/encode/httpx | transitive | installed with the wheel, not inside it |
| `huggingface-hub` | 1.22.0 | Apache Software License | https://github.com/huggingface/huggingface_hub | direct: modelmri | installed with the wheel, not inside it |
| `idna` | 3.18 | BSD-3-Clause | https://github.com/kjd/idna | transitive | installed with the wheel, not inside it |
| `importlib-metadata` | 9.0.0 | Apache-2.0 | https://github.com/python/importlib_metadata | transitive | installed with the wheel, not inside it |
| `iniconfig` | 2.3.0 | MIT | https://github.com/pytest-dev/iniconfig | dev group only | never installed with a wheel (development tooling) |
| `interegular` | 0.3.3 | MIT License | https://github.com/MegaIng/regex_intersections | transitive | installed with the wheel, not inside it |
| `jinja2` | 3.1.6 | BSD License | https://github.com/pallets/jinja/ | transitive | installed with the wheel, not inside it |
| `license-expression` | 30.4.4 | Apache-2.0 (PyPI metadata) | https://github.com/aboutcode-org/license-expression | dev group only | never installed with a wheel (development tooling) |
| `lm-format-enforcer` | 0.11.3 | MIT License | https://github.com/noamgat/lm-format-enforcer | direct: modelmri | installed with the wheel, not inside it |
| `markdown-it-py` | 4.2.0 | MIT License | https://github.com/executablebooks/markdown-it-py | transitive | installed with the wheel, not inside it |
| `markupsafe` | 3.0.3 | BSD-3-Clause | https://github.com/pallets/markupsafe/ | transitive | installed with the wheel, not inside it |
| `mdurl` | 0.1.2 | MIT License | https://github.com/executablebooks/mdurl | transitive | installed with the wheel, not inside it |
| `mpmath` | 1.3.0 | BSD License | http://mpmath.org/ | transitive | installed with the wheel, not inside it |
| `networkx` | 3.4.2 (from https://pypi.org/simple) | BSD-3-Clause | https://networkx.org/ | transitive | installed with the wheel, not inside it |
| `networkx` | 3.6.1 (from https://pypi.org/simple) | BSD-3-Clause | https://networkx.org/ | transitive | installed with the wheel, not inside it |
| `numpy` | 2.2.6 (from https://pypi.org/simple) | BSD-3-Clause AND 0BSD AND MIT AND Zlib AND CC0-1.0 | https://numpy.org | direct: modelmri | installed with the wheel, not inside it |
| `numpy` | 2.4.6 (from https://pypi.org/simple) | BSD-3-Clause AND 0BSD AND MIT AND Zlib AND CC0-1.0 | https://numpy.org | direct: modelmri | installed with the wheel, not inside it |
| `numpy` | 2.5.1 (from https://pypi.org/simple) | BSD-3-Clause AND 0BSD AND MIT AND Zlib AND CC0-1.0 | https://numpy.org | direct: modelmri | installed with the wheel, not inside it |
| `nvidia-cublas` | 13.1.1.3 | LicenseRef-NVIDIA-Proprietary (PyPI metadata) | https://developer.nvidia.com/cuda-zone | transitive | installed with the wheel, not inside it |
| `nvidia-cuda-cupti` | 13.0.85 | Other/Proprietary License (PyPI metadata) | https://developer.nvidia.com/cuda-zone | transitive | installed with the wheel, not inside it |
| `nvidia-cuda-nvrtc` | 13.0.88 | Other/Proprietary License (PyPI metadata) | https://developer.nvidia.com/cuda-zone | transitive | installed with the wheel, not inside it |
| `nvidia-cuda-runtime` | 13.0.96 | no license metadata (PyPI) | https://developer.nvidia.com/cuda-zone | transitive | installed with the wheel, not inside it |
| `nvidia-cudnn-cu13` | 9.20.0.48 | no license metadata (PyPI) | https://developer.nvidia.com/cuda-zone | transitive | installed with the wheel, not inside it |
| `nvidia-cufft` | 12.0.0.61 | Other/Proprietary License (PyPI metadata) | https://developer.nvidia.com/cuda-zone | transitive | installed with the wheel, not inside it |
| `nvidia-cufile` | 1.15.1.6 | Other/Proprietary License (PyPI metadata) | https://developer.nvidia.com/cuda-zone | transitive | installed with the wheel, not inside it |
| `nvidia-curand` | 10.4.0.35 | Other/Proprietary License (PyPI metadata) | https://developer.nvidia.com/cuda-zone | transitive | installed with the wheel, not inside it |
| `nvidia-cusolver` | 12.0.4.66 | Other/Proprietary License (PyPI metadata) | https://developer.nvidia.com/cuda-zone | transitive | installed with the wheel, not inside it |
| `nvidia-cusparse` | 12.6.3.3 | Other/Proprietary License (PyPI metadata) | https://developer.nvidia.com/cuda-zone | transitive | installed with the wheel, not inside it |
| `nvidia-cusparselt-cu13` | 0.8.1 | NVIDIA Proprietary Software (PyPI metadata) | https://developer.nvidia.com/cusparselt | transitive | installed with the wheel, not inside it |
| `nvidia-nccl-cu13` | 2.29.7 | no license metadata (PyPI) | https://developer.nvidia.com/cuda-zone | transitive | installed with the wheel, not inside it |
| `nvidia-nvjitlink` | 13.0.88 | Other/Proprietary License (PyPI metadata) | https://developer.nvidia.com/cuda-zone | transitive | installed with the wheel, not inside it |
| `nvidia-nvshmem-cu13` | 3.4.5 | no license metadata (PyPI) | https://developer.nvidia.com/cuda-zone | transitive | installed with the wheel, not inside it |
| `nvidia-nvtx` | 13.0.85 | Other/Proprietary License (PyPI metadata) | https://developer.nvidia.com/cuda-zone | transitive | installed with the wheel, not inside it |
| `packaging` | 26.2 | Apache-2.0 OR BSD-2-Clause | https://github.com/pypa/packaging | transitive | installed with the wheel, not inside it |
| `pillow` | 12.3.0 | MIT-CMU | https://python-pillow.github.io | direct: modelmri | installed with the wheel, not inside it |
| `playwright` | 1.62.0 | Apache-2.0 | https://github.com/Microsoft/playwright-python | dev group only | never installed with a wheel (development tooling) |
| `pluggy` | 1.6.0 | MIT License | https://pypi.org/project/pluggy/ | dev group only | never installed with a wheel (development tooling) |
| `psutil` | 7.2.2 | BSD-3-Clause | https://github.com/giampaolo/psutil | transitive | installed with the wheel, not inside it |
| `pyarrow` | 25.0.0 | Apache-2.0 | https://arrow.apache.org/ | direct: modelmri | installed with the wheel, not inside it |
| `pydantic` | 2.13.4 | MIT | https://github.com/pydantic/pydantic | direct: modelmri | installed with the wheel, not inside it |
| `pydantic-core` | 2.46.4 | MIT | https://github.com/pydantic/pydantic | transitive | installed with the wheel, not inside it |
| `pyee` | 13.0.1 | MIT License | https://github.com/jfhbrook/pyee | dev group only | never installed with a wheel (development tooling) |
| `pygments` | 2.20.0 | BSD-2-Clause | https://pygments.org | transitive | installed with the wheel, not inside it |
| `pytest` | 9.1.1 | MIT | https://docs.pytest.org/en/latest/ | dev group only | never installed with a wheel (development tooling) |
| `pytest-xdist` | 3.8.0 | MIT | https://github.com/pytest-dev/pytest-xdist | dev group only | never installed with a wheel (development tooling) |
| `python-debian` | 1.1.1 | GPL-2.0-or-later (PyPI metadata) | https://salsa.debian.org/python-debian-team/python-debian | dev group only | never installed with a wheel (development tooling) |
| `python-dotenv` | 1.2.2 | BSD-3-Clause | https://github.com/theskumar/python-dotenv | transitive | installed with the wheel, not inside it |
| `python-magic` | 0.4.27 | MIT License (PyPI metadata) | http://github.com/ahupp/python-magic | dev group only | never installed with a wheel (development tooling) |
| `pyyaml` | 6.0.3 | MIT License | https://pyyaml.org/ | transitive | installed with the wheel, not inside it |
| `regex` | 2026.6.28 | Apache-2.0 AND CNRI-Python | https://github.com/mrabarnett/mrab-regex | transitive | installed with the wheel, not inside it |
| `requests` | 2.34.2 | Apache Software License | https://github.com/psf/requests | transitive | installed with the wheel, not inside it |
| `reuse` | 6.2.0 | CC0 1.0 Universal (CC0 1.0) Public Domain Dedication; DFSG approved; OSI Approved; Apache Software License; GNU General Public License v3 or later (GPLv3+); Other/Proprietary License (PyPI metadata) | https://reuse.software/ | dev group only | never installed with a wheel (development tooling) |
| `rich` | 15.0.0 | MIT License | https://github.com/Textualize/rich | transitive | installed with the wheel, not inside it |
| `ruff` | 0.15.20 | MIT | https://docs.astral.sh/ruff | dev group only | never installed with a wheel (development tooling) |
| `safetensors` | 0.8.0 | Apache Software License | https://github.com/huggingface/safetensors | direct: modelmri, modelmri-policy | installed with the wheel, not inside it |
| `setuptools` | 81.0.0 | MIT | https://github.com/pypa/setuptools | transitive | installed with the wheel, not inside it |
| `shellingham` | 1.5.4 | ISC License (ISCL) | https://github.com/sarugaku/shellingham | transitive | installed with the wheel, not inside it |
| `starlette` | 1.3.1 | BSD-3-Clause | https://github.com/Kludex/starlette | transitive | installed with the wheel, not inside it |
| `sympy` | 1.14.0 | BSD License | https://sympy.org | transitive | installed with the wheel, not inside it |
| `tokenizers` | 0.22.2 | Apache Software License | https://github.com/huggingface/tokenizers | transitive | installed with the wheel, not inside it |
| `tomli` | 2.4.1 | MIT (PyPI metadata) | https://github.com/hukkin/tomli | dev group only | never installed with a wheel (development tooling) |
| `tomlkit` | 0.15.1 | MIT License (PyPI metadata) | https://github.com/python-poetry/tomlkit | dev group only | never installed with a wheel (development tooling) |
| `torch` | 2.11.0+cu128 (from https://download.pytorch.org/whl/cu128) | BSD-3-Clause | https://pytorch.org | direct: modelmri, modelmri-policy | installed with the wheel, not inside it |
| `torch` | 2.13.0 (from https://pypi.org/simple) | BSD-3-Clause | https://pytorch.org | direct: modelmri, modelmri-policy | installed with the wheel, not inside it |
| `torchvision` | 0.26.0 (from https://pypi.org/simple) | BSD | https://github.com/pytorch/vision | direct: modelmri | installed with the wheel, not inside it |
| `torchvision` | 0.28.0 (from https://pypi.org/simple) | BSD | https://github.com/pytorch/vision | direct: modelmri | installed with the wheel, not inside it |
| `tqdm` | 4.68.4 | MPL-2.0 AND MIT | https://tqdm.github.io | transitive | installed with the wheel, not inside it |
| `transformers` | 5.13.0 | Apache 2.0 License | https://github.com/huggingface/transformers | direct: modelmri | installed with the wheel, not inside it |
| `triton` | 3.7.1 | MIT License (PyPI metadata) | https://github.com/triton-lang/triton/ | transitive | installed with the wheel, not inside it |
| `typer` | 0.26.8 | MIT | https://github.com/fastapi/typer | transitive | installed with the wheel, not inside it |
| `typing-extensions` | 4.16.0 | PSF-2.0 | https://github.com/python/typing_extensions | transitive | installed with the wheel, not inside it |
| `typing-inspection` | 0.4.2 | MIT | https://github.com/pydantic/typing-inspection | transitive | installed with the wheel, not inside it |
| `urllib3` | 2.7.0 | MIT | https://pypi.org/project/urllib3/ | transitive | installed with the wheel, not inside it |
| `uvicorn` | 0.50.2 | BSD-3-Clause | https://uvicorn.dev/ | direct: modelmri | installed with the wheel, not inside it |
| `uvloop` | 0.22.1 | Apache Software License; MIT License (PyPI metadata) | https://pypi.org/project/uvloop/ | transitive | installed with the wheel, not inside it |
| `watchfiles` | 1.2.0 | MIT License | https://github.com/samuelcolvin/watchfiles | transitive | installed with the wheel, not inside it |
| `websockets` | 16.0 | BSD-3-Clause | https://github.com/python-websockets/websockets | transitive | installed with the wheel, not inside it |
| `zipp` | 4.1.0 | MIT | https://github.com/jaraco/zipp | transitive | installed with the wheel, not inside it |
| `lerobot` | not pinned by uv.lock | Apache Software License (PyPI metadata) | https://huggingface.co/lerobot | direct: modelmri-policy (an extra resolved in its own environment) | installed where that extra is installed |

### npm — runtime tree (what the built application can bundle)

| package | version | license | homepage | relation | shipped? |
|---|---|---|---|---|---|
| `@fontsource/archivo-black` | 5.3.0 | OFL-1.1 | https://www.npmjs.com/package/@fontsource/archivo-black | direct | bundled into the built app if imported |
| `react` | 19.2.8 | MIT | https://www.npmjs.com/package/react | direct | bundled into the built app if imported |
| `react-dom` | 19.2.8 | MIT | https://www.npmjs.com/package/react-dom | direct | bundled into the built app if imported |
| `scheduler` | 0.27.0 | MIT | https://www.npmjs.com/package/scheduler | transitive | bundled into the built app if imported |

### npm — build-time tools (112 packages in the lockfile in total; direct entries below, not shipped)

| package | version | license | homepage | relation | shipped? |
|---|---|---|---|---|---|
| `@tailwindcss/vite` | 4.3.3 | MIT | https://www.npmjs.com/package/@tailwindcss/vite | direct (dev) | build-time only, not shipped |
| `@types/react` | 19.2.18 | MIT | https://www.npmjs.com/package/@types/react | direct (dev) | build-time only, not shipped |
| `@types/react-dom` | 19.2.5 | MIT | https://www.npmjs.com/package/@types/react-dom | direct (dev) | build-time only, not shipped |
| `@vitejs/plugin-react` | 6.1.0 | MIT | https://www.npmjs.com/package/@vitejs/plugin-react | direct (dev) | build-time only, not shipped |
| `tailwindcss` | 4.3.3 | MIT | https://www.npmjs.com/package/tailwindcss | direct (dev) | build-time only, not shipped |
| `typescript` | 7.0.2 | Apache-2.0 | https://www.npmjs.com/package/typescript | direct (dev) | build-time only, not shipped |
| `vite` | 8.2.2 | MIT | https://www.npmjs.com/package/vite | direct (dev) | build-time only, not shipped |

<!-- /generated-section -->

## Bundled assets

Shipped inside the built application (and therefore inside the `modelmri`
wheel, which carries the built application):

- **Geist** — `frontend/src/fonts/Geist-Variable.woff2`. Copyright 2024 The
  Geist Project Authors (<https://github.com/vercel/geist-font>). SIL Open
  Font License, Version 1.1; the license text is beside the font,
  `frontend/src/fonts/Geist-OFL.txt`.
- **JetBrains Mono** — `frontend/src/fonts/JetBrainsMono-Variable.woff2`.
  Copyright 2020 The JetBrains Mono Project Authors
  (<https://github.com/JetBrains/JetBrainsMono>). SIL Open Font License,
  Version 1.1; the license text is beside the font,
  `frontend/src/fonts/JetBrainsMono-OFL.txt`.

In the repository but in no build, recorded so that the state is written
down rather than discovered:

- **Switzer** — `frontend/src/fonts/Switzer-Variable.woff2` is committed but
  referenced by no stylesheet rule (only by comments), so it is not part of
  any build. It is an Indian Type Foundry face distributed through Fontshare
  under Fontshare's free-font licence, which is not the OFL, and its licence
  text is not in the repository. The file will be removed, or its licence
  restored beside it; until one of those has happened, this entry is the
  notice.
- **Archivo Black** — `@fontsource/archivo-black` (SIL OFL 1.1) is declared
  in `frontend/package.json` and imported by nothing, so it is in the
  dependency tree and in no build.

## Loaded at runtime, not distributed

Model weights, sparse autoencoders, datasets and robot policies are fetched
when you ask for them, from the publisher, under the publisher's license.
ModelMRI does not redistribute any of them, and where a publisher gates a
download behind accepting terms, ModelMRI links you to that page with your
own account rather than accepting anything for you. In particular:

- the GPT-2 sparse autoencoders from `jbloom/GPT2-Small-SAEs-Reformatted`
  (SAELens format), and the other SAE releases the registry in
  `modelmri/saes.py` knows how to load, each under its own publisher's terms;
- `lerobot/smolvla_base` and `lerobot/pusht` (HuggingFace LeRobot);
- any model you load from the HuggingFace Hub or from a local Ollama daemon.

## Test fixtures

None with an external origin are committed. `tests/` contains Python only;
a test that needs a model, an SAE or a dataset either builds a synthetic one
or fetches the real one from the Hub at run time, under that publisher's
terms.

## Provenance rule

Every method in ModelMRI is implemented independently from the paper that
describes it, and the paper is recorded — in the module's docstring and in
the changelog entry that introduced the method. Sources are for checking
claims, not for lifting designs. No code is copied from a repository that
does not ship a usable license: `modelmri/model_diff.py` declines to port two
crosscoder implementations for exactly that reason and says so in its
docstring, and a clean-room implementation from the papers is the only route
that would be taken. A contribution that brings third-party code with it must
say so, name the source and its license, and pass the same test — see
[CONTRIBUTING.md](CONTRIBUTING.md).
