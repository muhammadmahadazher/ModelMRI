"""Will this image model run on THIS machine, and what will it actually cost?

`capacity.guard` answers a different question — "is this download so much bigger
than the machine that fetching it is a mistake" — with a deliberately generous
ceiling, on the way in. Nothing answered the question a reader asks while
looking at a list of models they already have: *which of these will run here?*

Three things were wrong with the number the image panel had to answer it with.

**It quoted disk bytes as if they were card bytes.** `image_runtime._weights_bytes`
sums `stat().st_size`, and the load converts every float to the accelerator's
preferred dtype. `fit.Weights` says this in its own docstring — an F32
checkpoint loaded bf16 allocates half its file size — and the image side was
making exactly the mistake the language side had already been fixed for.

**It counted the same weights twice.** MEASURED, on the cached
`stabilityai/sd-turbo` in this machine's cache: `vae/` holds both
`diffusion_pytorch_model.safetensors` (335 MB) and
`diffusion_pytorch_model.fp16.safetensors` (167 MB). Those are two copies of
one component and a load reads one of them. Summing every weight file in the
folder over-quoted that component by 50%.

**It never checked whether the load could succeed at all.** The same cached
sd-turbo has `text_encoder/model.fp16.safetensors` and no plain file, so
`from_pretrained` without `variant="fp16"` cannot find weights for a component
the pipeline requires. That is a model which is fully downloaded, correctly
detected, listed as ready — and which fails at the click. Reporting the
variant a checkpoint actually needs is the difference between "not all models
just plug and play" being a surprise and being something the panel says first.

Nothing here is a name list. Components are found by looking for directories
that contain weights, variants by reading the filenames the publisher used, and
the dtype from the device that will do the loading. A checkpoint laid out in a
way nobody anticipated is priced by the same rules as every other.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path

from . import fit as _fit

log = logging.getLogger(__name__)

#: Weight containers, matching `image_runtime.WEIGHT_SUFFIXES`. Pickles are in
#: the set because real pipelines ship them; they are priced differently below,
#: because a pickle has no readable tensor table.
WEIGHT_SUFFIXES = frozenset({".safetensors", ".bin", ".pt", ".pth", ".ckpt"})

#: Only safetensors carries a tensor table we can read without executing the
#: file. Everything else gets its size from the filesystem and is reported as
#: an estimate rather than a measurement.
EXACT_SUFFIX = ".safetensors"

#: Components a pipeline holds weights for on disk but does not keep resident
#: in the configuration this tool loads. `_load_diffusion` passes
#: `safety_checker=None`, so counting its weights would quote megabytes that
#: are never allocated. Excluded components are REPORTED, not dropped quietly —
#: a number that silently omits a folder somebody can see on their disk reads
#: as an arithmetic error.
NOT_RESIDENT = frozenset({"safety_checker"})

#: `diffusion_pytorch_model-00001-of-00002.safetensors` — the shard marker
#: diffusers and transformers both write, stripped before reading the variant.
_SHARD = re.compile(r"-\d{3,}-of-\d{3,}$")

#: What a component folder may be called. Anything else in a `model_index.json`
#: is treated as a corrupt index rather than followed onto the filesystem.
_SAFE_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")

#: The weight filenames diffusers and transformers write. A variant is formed
#: by inserting `.{variant}` after one of THESE — `model.fp16.safetensors` —
#: so a dot only means "variant" when what precedes it is a canonical name.
#:
#: MEASURED: `facebook/sam3.1` ships `sam3.1_multiplex.pt`, and reading the
#: text after the last dot gave it the variant `1_multiplex`. The dot is part
#: of the model's own name. This is the libraries' file convention, the same
#: kind of fact as the shard pattern above — not a list of models.
_CANONICAL_STEMS = frozenset(
    {
        "model",
        "diffusion_pytorch_model",
        "pytorch_model",
        "tf_model",
        "flax_model",
        "adapter_model",
    }
)

#: Room left for everything that is not weights: the latents, the attention
#: maps this tool captures, the autocast copies, allocator fragmentation.
#:
#: MEASURED, on an RTX 4060 Laptop with `nota-ai/bk-sdm-tiny` at 512x512 —
#: `measure_overhead` at the bottom of this module is what produced these, by
#: running the passes this tool actually runs and reading
#: `torch.cuda.max_memory_allocated` above the resting weights:
#:
#:     weights resting              1083 MB
#:     a plain latent trace          141 MB above weights
#:     a cross-attention capture     650 MB above weights
#:
#: Two things that reading settles. The capture costs **4.5x** the plain trace,
#: so the threshold has to be set for the capture — a bound drawn from
#: denoising alone would call a model comfortable and then die during the one
#: measurement people came for.
#:
#: And it does NOT grow with steps: 650, 640 and 640 MB at 6, 20 and 30 steps.
#: The maps do not accumulate on the card, so peak allocation is bounded by one
#: step's working set rather than by the length of the run. That is what makes
#: a single constant honest here instead of something that has to scale.
#:
#: 900 MiB sits above the worst of those with room to spare, which is the right
#: direction for a figure that gates a refusal: the models measured are small,
#: and a full-size SD or SDXL holds larger maps per step. It is REPORTED in
#: every verdict (`activation_headroom`) so a reader can check this arithmetic
#: rather than take it on faith.
ACTIVATION_HEADROOM = 900 * 1024 * 1024


@dataclass
class Component:
    """One weighted part of a pipeline, priced from the files it actually has."""

    name: str
    #: The variant chosen for pricing — "" for the unsuffixed files.
    variant: str
    #: Every variant present on disk, so a reader can see what was NOT chosen.
    variants: list[str]
    files: int
    disk_bytes: int
    #: What this will occupy once loaded at the target dtype. `None` when it
    #: could not be priced — never 0, which would read as "this is free".
    card_bytes: int | None
    #: True when priced from a tensor table, False when from file sizes.
    exact: bool
    note: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class ImageFit:
    """What a checkpoint costs here, and whether here is big enough."""

    path: str
    components: list[Component] = field(default_factory=list)
    #: Components found on disk and deliberately not counted, with the reason.
    excluded: list[str] = field(default_factory=list)
    #: Declared components whose folder is here and CONTRADICTS ITSELF — a
    #: config with no weights, or weights with no config. Non-empty means the
    #: load fails, however well the sizes fit.
    missing: list[str] = field(default_factory=list)
    #: Declared components whose folder is not here at all. Reported, never
    #: blocking: a component can be handed to `from_pretrained` directly, and
    #: this cannot see that from the files on disk.
    absent: list[str] = field(default_factory=list)
    disk_bytes: int = 0
    #: Resident weight bytes at `dtype`. `None` when any component could not be
    #: priced, because a total missing one part is not a total.
    card_bytes: int | None = None
    dtype: str = ""
    device: str = ""
    device_name: str = ""
    free_bytes: int | None = None
    total_bytes: int | None = None
    headroom_bytes: int | None = None
    #: "fits" | "tight" | "over" | "unknown" — and "unknown" is a real answer,
    #: returned whenever the card's free memory could not be read, rather than
    #: a cheerful default.
    verdict: str = "unknown"
    #: The variant `from_pretrained` must be given, or "" when the plain files
    #: are complete. `None` when no variant covers every component, which means
    #: the checkpoint cannot be loaded as it stands.
    variant: str | None = ""
    loadable: bool = True
    exact: bool = True
    activation_headroom: int = ACTIVATION_HEADROOM
    reason: str = ""
    means: str = ""

    def to_dict(self) -> dict:
        out = asdict(self)
        out["components"] = [c.to_dict() for c in self.components]
        return out


def _variant_of(name: str) -> str:
    """Which weight variant a filename belongs to, "" for the plain one.

    diffusers writes the variant between the stem and the extension —
    `diffusion_pytorch_model.fp16.safetensors` — and before the shard marker
    when a component is split. Read from the name the publisher chose rather
    than from a list of variants this happens to know about, so `bf16`, `int8`
    or a variant invented next year is handled like `fp16`.
    """
    stem = Path(name).stem
    stem = _SHARD.sub("", stem)
    head, dot, tail = stem.rpartition(".")
    if not dot or not tail:
        return ""
    if head not in _CANONICAL_STEMS:
        # A dot inside the publisher's own filename, not a variant marker.
        return ""
    return tail


def _stem_of(name: str) -> str:
    """The filename with its shard marker and variant removed.

    `diffusion_pytorch_model.fp16-00001-of-00002.safetensors` and
    `diffusion_pytorch_model.safetensors` both reduce to
    `diffusion_pytorch_model`, which is what decides whether a file is one the
    loaders go looking for.
    """
    stem = _SHARD.sub("", Path(name).stem)
    head, dot, tail = stem.rpartition(".")
    return head if dot and head in _CANONICAL_STEMS else stem


def _weights_in(folder: Path) -> list[Path]:
    try:
        entries = sorted(folder.iterdir())
    except OSError:
        return []
    return [f for f in entries if f.is_file() and f.suffix.lower() in WEIGHT_SUFFIXES]


def _one_format(files: list[Path]) -> tuple[list[Path], list[Path]]:
    """(what a load reads, what it ignores) when one component ships both.

    transformers writes `pytorch_model.bin` and `model.safetensors` for the
    SAME tensors, and `use_safetensors` defaults to preferring the safetensors
    — so a directory holding both had its pickle downloaded and never opened.
    Counting both is the module's founding defect surviving on a second axis:
    it is the same "two copies of one component" error as the fp16 twin, just
    keyed on format instead of precision.

    MEASURED on `google/vit-base-patch16-224` in this machine's cache, which
    ships `model.safetensors` and `pytorch_model.bin` side by side: priced with
    both it came to 346 MB on the card, exactly twice the 173 MB one copy
    actually occupies.

    `image_runtime._one_copy` already states this rule for the DOWNLOAD list.
    The order matters and is the same here — format first, then precision —
    because the two are independent and applying precision first would pick an
    fp16 pickle over a plain safetensors.
    """
    safe = [f for f in files if f.suffix.lower() == EXACT_SUFFIX]
    if not safe or len(safe) == len(files):
        return files, []
    return safe, [f for f in files if f.suffix.lower() != EXACT_SUFFIX]


def _by_variant(files: list[Path]) -> dict[str, list[Path]]:
    groups: dict[str, list[Path]] = {}
    for f in files:
        groups.setdefault(_variant_of(f.name), []).append(f)
    return groups


def _weighted_parts(root: Path) -> tuple[dict[str, list[Path]], list[str]]:
    """Which directories hold weights, found by looking rather than by name.

    A diffusers pipeline keeps each component in its own folder; a transformers
    checkpoint keeps its weights flat. Both are handled by the same rule — a
    directory that contains weight files is a weighted component — so a
    pipeline with a component nobody here has heard of (`image_encoder`,
    `text_encoder_3`, `prior`) is counted, and a folder of tokenizer JSON is
    not.
    """
    parts: dict[str, list[Path]] = {}
    skipped: list[str] = []
    #: A subdirectory that carries a plain `config.json` is a COMPONENT, even
    #: when its weights are missing. Tracked because it decides whether the
    #: flat-checkpoint fallback below is allowed to fire at all.
    component_dirs = 0
    try:
        entries = sorted(root.iterdir())
    except OSError as err:
        raise _fit.Refusal(
            f"{root} could not be read, so what it holds is unknown rather "
            f"than empty ({err.__class__.__name__})."
        ) from err

    for entry in entries:
        if not entry.is_dir():
            continue
        if (entry / "config.json").is_file():
            component_dirs += 1
        found = _weights_in(entry)
        if not found:
            continue
        if entry.name in NOT_RESIDENT:
            skipped.append(
                f"{entry.name} — loaded as None by this tool, so its weights "
                f"are on disk but never on the card"
            )
            continue
        found, ignored = _one_format(found)
        if ignored:
            skipped.append(
                f"{entry.name}/{ignored[0].name} and {len(ignored) - 1} other "
                f"file(s) — a second copy of the same tensors in a format the "
                f"loader does not prefer"
                if len(ignored) > 1
                else f"{entry.name}/{ignored[0].name} — a second copy of the "
                f"same tensors in a format the loader does not prefer"
            )
        parts[entry.name] = found

    if parts:
        return parts, skipped

    # A FLAT checkpoint — but only if nothing here looks like a pipeline.
    #
    # MEASURED on `stabilityai/stable-diffusion-xl-base-1.0` after Drive
    # stripped its weights: every component folder still held its config, none
    # held weights, so this fell through to the root and found
    # `sd_xl_offset_example-lora_1.0.safetensors`. A 7 GB pipeline with nothing
    # loadable in it was priced at 49 MB from a LoRA and published as "fits" —
    # the exact green-badge-then-failed-click this module exists to prevent.
    #
    # So: the fallback is for checkpoints that were never a pipeline, and the
    # root file has to be one the loaders actually look for. A stray adapter or
    # example file sitting beside a pipeline is not the model.
    # A root `config.json` is a transformers checkpoint saying "I am the
    # model", and then whatever weight file sits beside it IS the weights,
    # whatever it is called. MEASURED: `facebook/sam3.1` ships exactly one
    # weight file, `sam3.1_multiplex.pt`, and requiring a canonical stem
    # rejected a perfectly loadable checkpoint — a false refusal, which is the
    # worse error of the two because it hides a model that works.
    #
    # Without that config there is nothing claiming to be a model, so only the
    # filenames the loaders actually go looking for count. That is the SDXL
    # case: `model_index.json` and no root config, where the only root file was
    # `sd_xl_offset_example-lora_1.0.safetensors`.
    if component_dirs:
        return {}, skipped
    flat = _weights_in(root)
    if not (root / "config.json").is_file():
        flat = [f for f in flat if _stem_of(f.name) in _CANONICAL_STEMS]
    if flat:
        kept, ignored = _one_format(flat)
        if ignored:
            skipped.append(
                f"{ignored[0].name} — a second copy of the same tensors in a "
                f"format the loader does not prefer"
            )
        return {"": kept}, skipped
    return {}, skipped


def _declared(root: Path) -> dict[str, str]:
    """Components the pipeline says it is made of, as {name: class}.

    From `model_index.json`, which is the publisher's own statement of what
    `from_pretrained` will try to build. Entries whose class is null are ones
    the pipeline explicitly does not carry — `safety_checker: [null, null]` —
    and a missing folder for those is correct rather than a gap.
    """
    try:
        raw = json.loads((root / "model_index.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    out = {}
    for name, spec in raw.items():
        if name.startswith("_") or not isinstance(spec, list) or len(spec) != 2:
            continue
        module, cls = spec
        if cls:
            out[name] = str(cls)
    return out


def _gaps(root: Path, declared: dict[str, str]) -> tuple[list[str], list[str]]:
    """Which declared components cannot be built from what is on this disk.

    A weight-bearing component writes a plain `config.json` beside its weights;
    a scheduler writes `scheduler_config.json` and a tokenizer
    `tokenizer_config.json`, and neither has weights to miss. So a directory is
    read as a model when it has a plain config OR it has weights — and a model
    needs both. Reading the layout rather than matching class names means a
    component type nobody here anticipated is checked by the same rule.

    MEASURED, on two checkpoints in this machine's cache: `segmind/tiny-sd`
    has `unet/config.json` and no unet weights, and `stabilityai/sd-turbo` has
    `text_encoder/model.fp16.safetensors` and no config for it. Both were
    listed as complete models ready to open, because the old check asked only
    whether ANY weights existed anywhere underneath.

    Returns `(broken, absent)`, and the split decides whether a load is
    REFUSED or merely annotated:

    * **broken** — the folder is here and contradicts itself. A config with no
      weights, or weights with no config, is unambiguous evidence of a
      download that did not finish, and refusing costs the reader nothing they
      had.
    * **absent** — the folder is not here at all. Weaker evidence, and NOT
      grounds to refuse: `from_pretrained` can be handed a component directly
      rather than loading it from disk, and this cannot see that from the
      files. A refusal here would block a pipeline that was going to work,
      which is worse than the diffusers error it was trying to pre-empt.
    """
    problems: list[str] = []
    absent: list[str] = []
    for name in sorted(declared):
        if not _SAFE_NAME.match(name):
            # `name` arrives from a downloaded `model_index.json`, and on
            # Windows `Path(root) / "C:/Windows/System32"` resolves to the
            # ABSOLUTE path — so a malformed or hostile index would have this
            # listing route walk an arbitrary directory and publish a sentence
            # about what it found. `image_catalog.is_hub_id` closes the same
            # class of hole for repo ids.
            problems.append(
                f"{name!r} is not a name a component folder can have, so this "
                f"index cannot be trusted to say what the pipeline is made of"
            )
            continue
        if name in NOT_RESIDENT:
            # Never built, so nothing about it can be missing. `bk-sdm-tiny`
            # declares a real `StableDiffusionSafetyChecker` class and ships
            # only its config; this tool passes `safety_checker=None`, so that
            # is a component it deliberately does not load rather than a
            # broken download.
            continue
        folder = root / name
        if not folder.is_dir():
            absent.append(f"{name} is declared but its folder is not here")
            continue
        has_weights = bool(_weights_in(folder))
        has_config = (folder / "config.json").is_file()
        if not has_weights and not has_config:
            # A scheduler or tokenizer: no weights expected, nothing missing.
            continue
        if has_config and not has_weights:
            problems.append(
                f"{name} has a config and no weight file, so the download of "
                f"that component did not finish"
            )
        elif has_weights and not has_config:
            problems.append(
                f"{name} has weights and no config.json, so there is nothing "
                f"telling the loader what to build them into"
            )
    return problems, absent


def _choose_variant(parts: dict[str, list[Path]]) -> tuple[str | None, str]:
    """The variant a load must ask for, and why.

    diffusers falls back to the unsuffixed file for any component that has no
    copy of the requested variant, so a variant is usable when every component
    has either that variant or a plain file. The plain load asks for nothing
    and therefore needs a plain file EVERYWHERE — which is the case the cached
    sd-turbo fails, because its text encoder ships fp16 only.
    """
    available = {name: set(_by_variant(files)) for name, files in parts.items()}
    if all("" in have for have in available.values()):
        return "", ""

    # `n or "this checkpoint"`: a flat checkpoint's component is named "" until
    # it is renamed to "weights" further down, so the sentence read
    # " ship only variant weights, so this must be loaded with …".
    missing = sorted(
        (n or "this checkpoint") for n, have in available.items() if "" not in have
    )
    every = set().union(*available.values()) if available else set()
    usable = sorted(
        v
        for v in every
        if v and all(v in have or "" in have for have in available.values())
    )
    if not usable:
        return None, (
            "no single set of weight files covers every component: "
            + ", ".join(
                f"{n or 'this checkpoint'} has {sorted(v for v in have) or 'nothing'}"
                for n, have in sorted(available.items())
            )
        )
    pick = usable[0]
    return pick, (
        f"{', '.join(missing)} ship only variant weights, so this must be "
        f"loaded with variant={pick!r} — a plain load cannot find weights for "
        f"it and fails at the click."
    )


#: Widths of the storage classes a torch pickle names, and which of them a
#: `dtype=` load converts. The same distinction `fit.FLOATING` draws for
#: safetensors, in the vocabulary the older format uses.
_STORAGE_BYTES = {
    "torch.DoubleStorage": 8,
    "torch.FloatStorage": 4,
    "torch.HalfStorage": 2,
    "torch.BFloat16Storage": 2,
    "torch.LongStorage": 8,
    "torch.IntStorage": 4,
    "torch.ShortStorage": 2,
    "torch.CharStorage": 1,
    "torch.ByteStorage": 1,
    "torch.BoolStorage": 1,
    "torch.ComplexFloatStorage": 8,
    "torch.ComplexDoubleStorage": 16,
    "torch.Float8_e4m3fnStorage": 1,
    "torch.Float8_e5m2Storage": 1,
}
_STORAGE_FLOATING = {
    "torch.DoubleStorage",
    "torch.FloatStorage",
    "torch.HalfStorage",
    "torch.BFloat16Storage",
    "torch.Float8_e4m3fnStorage",
    "torch.Float8_e5m2Storage",
}


class _Mark:
    """The pickle stack's MARK sentinel. Never unpickled, only compared."""


def _pickle_table(path: Path) -> tuple[dict[str, tuple[str, int]], dict[str, int]]:
    """Every tensor storage in a torch `.bin`, as (storage class, elements).

    A `.bin` is a zip holding `data.pkl` plus one blob per storage, and the
    pickle records each as `('storage', <class>, key, location, numel)`. That
    is exactly the tensor table safetensors publishes in the clear — it is just
    written in a format nobody can read without a pickle machine.

    So this walks the OPCODES with `pickletools.genops` and never calls
    `pickle.load`. No `__reduce__` runs, no import happens, nothing from the
    file is executed: the same reason `image_runtime._scan` exists rather than
    trusting a checkpoint, applied to reading its sizes.

    Only the opcodes needed to rebuild strings, ints and tuples are
    interpreted, and the memo is followed — without it the storage class is
    recovered once and every later reference comes back empty.

    MEASURED on `facebook/DiT-XL-2-256`: 538 storages recovered against 538
    blobs in the zip, and the byte total reconstructed from the table
    (2999.3 MB) matches the zip's own payload to the byte. Priced this way the
    pipeline came to 1.67 GB against 1.70 GB actually allocated; priced from
    the file size, as it was, it came to 3.33 GB — a 96% over-quote that would
    have called a model that fits comfortably a tight squeeze.
    """
    import io
    import pickletools
    import zipfile

    with zipfile.ZipFile(path) as zf:
        names = [n for n in zf.namelist() if n.endswith("data.pkl")]
        if not names:
            raise ValueError("no data.pkl inside — not a torch zip checkpoint")
        blob = zf.read(names[0])
        # The zip's own record of how many bytes each storage occupies. Kept
        # so the numbers the pickle CLAIMS can be checked against the bytes
        # that are actually there — see `_price_pickle`.
        blobs = {
            i.filename.rsplit("/", 1)[-1]: i.file_size
            for i in zf.infolist()
            if "/data/" in i.filename
        }

    stack: list = []
    memo: dict = {}
    # Keyed by STORAGE KEY, not appended per reference. `pickle.Pickler` never
    # memoizes a persistent id, so two state-dict entries sharing one storage
    # — tied embeddings, or two tensors that are views into the same buffer —
    # emit two `BINPERSID`s naming the same `data/N`. Appending counted that
    # storage twice while the zip holds it once.
    out: dict[str, tuple[str, int]] = {}

    for op, arg, _pos in pickletools.genops(io.BytesIO(blob)):
        name = op.name
        if name == "MARK":
            stack.append(_Mark)
        elif name in (
            "SHORT_BINUNICODE",
            "BINUNICODE",
            "UNICODE",
            "BINSTRING",
            "SHORT_BINSTRING",
            "BINBYTES",
            "SHORT_BINBYTES",
            "BININT",
            "BININT1",
            "BININT2",
            "INT",
            "LONG",
            "LONG1",
            "LONG4",
            "BINFLOAT",
            "FLOAT",
        ):
            stack.append(arg)
        elif name == "NONE":
            stack.append(None)
        elif name == "NEWTRUE":
            stack.append(True)
        elif name == "NEWFALSE":
            stack.append(False)
        elif name == "STACK_GLOBAL":
            qual = stack.pop() if stack else ""
            mod = stack.pop() if stack else ""
            stack.append(f"{mod}.{qual}")
        elif name == "GLOBAL":
            stack.append(str(arg).replace(" ", "."))
        elif name in ("BINPUT", "LONG_BINPUT", "PUT"):
            if stack:
                memo[arg] = stack[-1]
        elif name == "MEMOIZE":
            if stack:
                memo[len(memo)] = stack[-1]
        elif name in ("BINGET", "LONG_BINGET", "GET"):
            stack.append(memo.get(arg))
        elif name == "TUPLE":
            cut = len(stack) - 1
            while cut >= 0 and stack[cut] is not _Mark:
                cut -= 1
            items = tuple(stack[cut + 1 :])
            # `max(cut, 0)`: an unmatched TUPLE leaves `cut` at -1, and
            # `del stack[-1:]` removes ONE element where clearing was meant.
            # Unreachable on a valid pickle; a stack that silently keeps
            # growing is not how this should fail on an invalid one.
            del stack[max(cut, 0) :]
            stack.append(items)
        elif name == "TUPLE1":
            stack.append((stack.pop(),))
        elif name == "TUPLE2":
            b = stack.pop()
            stack.append((stack.pop(), b))
        elif name == "TUPLE3":
            c = stack.pop()
            b = stack.pop()
            stack.append((stack.pop(), b, c))
        elif name == "EMPTY_TUPLE":
            stack.append(())
        elif name == "EMPTY_DICT":
            stack.append({})
        elif name in ("EMPTY_LIST", "EMPTY_SET"):
            stack.append([])
        elif name == "BINPERSID":
            pid = stack.pop() if stack else None
            if (
                isinstance(pid, tuple)
                and len(pid) == 5
                and pid[0] == "storage"
                # `isinstance(True, int)` is True in Python, so the bool guard
                # comes FIRST — `numel=True` would otherwise be accepted and
                # counted as one element. Negatives are rejected for the same
                # reason: this is a number read out of a file, not one this
                # module computed.
                and isinstance(pid[4], int)
                and not isinstance(pid[4], bool)
                and pid[4] >= 0
            ):
                key = str(pid[2])
                seen = out.get(key)
                # One entry per storage. On the impossible case of two
                # references disagreeing about the length, the larger is kept:
                # this figure gates a refusal, and under-quoting is the
                # direction that ends in an OOM.
                if seen is None or pid[4] > seen[1]:
                    out[key] = (str(pid[1]), int(pid[4]))
            # The tensor this belongs to is rebuilt by a REDUCE this does not
            # follow; a placeholder keeps the stack the right depth.
            stack.append(None)
    return out, blobs


def _price_pickle(
    path: Path, dtype_bytes: int | None
) -> tuple[int, int | None, bool, str]:
    """One `.bin`, priced from its own tensor table where that can be read."""
    try:
        table, blobs = _pickle_table(path)
    except Exception as err:
        size = path.stat().st_size
        return (
            size,
            size,
            False,
            f"{path.name}'s tensor table could not be read "
            f"({type(err).__name__}), so its size on the card is the file's "
            f"size — an over-quote for any checkpoint stored wider than it "
            f"loads",
        )
    if not table:
        size = path.stat().st_size
        return size, size, False, f"{path.name} lists no tensor storages"

    disk = card = 0
    unknown: set[str] = set()
    for key, (storage, numel) in sorted(table.items()):
        width = _STORAGE_BYTES.get(storage)
        if width is None:
            unknown.add(storage)
            continue
        claimed = numel * width
        # CHECKED AGAINST THE ARCHIVE, not taken on the pickle's word. `numel`
        # is a number read out of a file this module deliberately does not
        # execute, and nothing else bounds it: a corrupt or hostile `.bin`
        # claiming `2**62` elements produced an eight-exabyte figure reported
        # as an exact measurement. The zip already records how many bytes the
        # storage occupies, so the two can simply be compared — and a file
        # whose own two records disagree is one to price conservatively and
        # say so, rather than to believe.
        actual = blobs.get(key)
        if actual is not None and actual != claimed:
            size = path.stat().st_size
            return (
                size,
                None,
                False,
                f"{path.name} describes storage {key} as {numel:,} × {width} "
                f"bytes but the archive stores {actual:,} bytes for it, so "
                f"what it really holds cannot be read from its own table",
            )
        disk += claimed
        card += numel * (
            dtype_bytes
            if dtype_bytes is not None and storage in _STORAGE_FLOATING
            else width
        )
    if unknown:
        size = path.stat().st_size
        return (
            size,
            None,
            False,
            f"{path.name} holds {', '.join(sorted(unknown))}, whose width this "
            f"does not know",
        )
    return disk, card, True, ""


def _price(
    files: list[Path], dtype_bytes: int | None, *, read_pickles: bool = True
) -> tuple[int, int | None, bool, str]:
    """Disk bytes, card bytes, whether exact, and what could not be read."""
    disk = 0
    card: int | None = 0
    exact = True
    trouble: list[str] = []

    for f in files:
        try:
            size = f.stat().st_size
        except OSError:
            trouble.append(f"{f.name} could not be sized")
            exact = False
            card = None
            continue

        if f.suffix.lower() != EXACT_SUFFIX:
            # A pickle DOES have a tensor table — `_pickle_table` reads it from
            # the opcodes without unpickling. Falling back to the file size
            # only when that cannot be read, and saying so when it happens.
            if not read_pickles:
                # Priced from the FILE, and said so. Reading a pickle's table
                # means opening the zip, which means the whole file has to be
                # materialised — measured at 69 s for one 346 MB `.bin` on
                # this machine's Drive-backed cache. A listing route that does
                # that for every cached model is a route that does not return.
                # The exact figure is read on the load path, where the file is
                # about to be opened anyway.
                disk += size
                if card is not None:
                    card += size
                exact = False
                trouble.append(
                    f"{f.name} is priced from its size on disk rather than its "
                    f"tensor table, which over-quotes any checkpoint stored "
                    f"wider than it loads. The exact figure is read when the "
                    f"model is opened"
                )
                continue
            pdisk, pcard, pexact, note = _price_pickle(f, dtype_bytes)
            disk += pdisk
            if pcard is None or card is None:
                card = None
            else:
                card += pcard
            exact = exact and pexact
            if note:
                trouble.append(note)
            continue

        try:
            header = _fit.read_header(f)
        except Exception as err:  # a truncated or unreadable shard
            exact = False
            card = None
            disk += size
            trouble.append(f"{f.name}: {err.__class__.__name__}")
            continue

        # Accumulated PER FILE and committed only once the whole table parsed.
        # Adding straight into `disk` meant a table that broke on its ninth
        # tensor contributed the payload of the first eight and was summed in
        # as though it were the file — a truncation applied to a published
        # number with nothing saying it had happened. `card` already latched
        # `None` on this path; the disk figure quietly did not.
        file_disk = 0
        for name, spec in header.items():
            try:
                dtype = spec["dtype"]
                shape = spec["shape"]
                # The PAYLOAD, the way `fit.weights_bytes` reads it — not
                # `st_size`, which also counts the JSON header and any
                # alignment padding. The pickle branch already reports its
                # payload, and a `disk` figure that means one thing for
                # safetensors and another for `.bin` cannot be compared with
                # `card` to see what a dtype conversion saved.
                start, end = spec["data_offsets"]
                file_disk += int(end) - int(start)
            except (KeyError, TypeError, ValueError):
                exact = False
                card = None
                file_disk = size
                trouble.append(f"{f.name} describes {name!r} in an unknown shape")
                break
            if dtype not in _fit.DTYPE_BYTES:
                exact = False
                card = None
                file_disk = size
                trouble.append(f"{f.name} holds a {dtype} tensor of unknown width")
                break
            count = 1
            for dim in shape:
                count *= int(dim)
            width = (
                dtype_bytes
                if dtype_bytes is not None and dtype in _fit.FLOATING
                else _fit.DTYPE_BYTES[dtype]
            )
            if card is not None:
                card += count * width
        disk += file_disk

    return disk, card, exact, "; ".join(trouble)


def _dtype_bytes(name: str) -> int | None:
    widths = {"float64": 8, "float32": 4, "bfloat16": 2, "float16": 2}
    return widths.get(name)


#: "the caller did not say", which is NOT the same as "the caller said this is
#: unknown". `None` is a real answer here — a Mac's unified memory reports no
#: free figure — and defaulting the parameters to `None` made an explicit
#: unknown indistinguishable from an omission, so a caller passing the honest
#: `None` got the runner's own card read behind its back.
_UNSET = object()


#: Priced results, keyed on what would change them. Bounded, because an
#: unbounded cache on a long-lived server is a leak rather than an optimisation.
_MEMO: dict[tuple, ImageFit] = {}
_MEMO_MAX = 256


def _fingerprint(root: Path) -> tuple:
    """Enough of the files' identity to know a re-read would say the same.

    Sizes and mtimes, never contents — the point is to avoid opening anything.
    A checkpoint whose files changed gets a different key and is priced again.
    """
    stamps = []
    try:
        for f in sorted(root.rglob("*")):
            if f.is_file() and f.suffix.lower() in WEIGHT_SUFFIXES:
                st = f.stat()
                stamps.append((f.name, st.st_size, int(st.st_mtime)))
    except OSError:
        return ()
    return tuple(stamps)


def of(
    path: str | Path,
    *,
    device=None,
    dtype: str = "",
    free_bytes: int | None = _UNSET,  # type: ignore[assignment]
    total_bytes: int | None = _UNSET,  # type: ignore[assignment]
    read_pickles: bool = True,
) -> ImageFit:
    """Price one checkpoint against the accelerator that would load it.

    `device` and the two byte counts are injectable so a caller pricing a whole
    list reads the card ONCE. Free memory is a measurement that changes under
    you; taking it per row would have every row answered against a slightly
    different machine and the totals would not add up.
    """
    from . import devices

    root = Path(path)
    out = ImageFit(path=str(root))
    cached_key = None

    if device is None:
        device = devices.detect()
    if free_bytes is _UNSET and total_bytes is _UNSET:
        free_bytes, total_bytes = devices._free_total(device)
    free_bytes = None if free_bytes is _UNSET else free_bytes
    total_bytes = None if total_bytes is _UNSET else total_bytes

    want = dtype or getattr(device, "dtype", "") or "float32"
    out.dtype = want
    out.device = getattr(device, "torch_device", "cpu")
    out.device_name = getattr(device, "name", "")
    out.free_bytes = free_bytes
    out.total_bytes = total_bytes

    # The memo holds everything read off the DISK. Free memory is a
    # measurement that changes under you, so it is applied fresh below rather
    # than cached — a verdict served from an hour-old reading of the card is
    # exactly the thing `devices._free_total` refuses to let `Device` carry.
    cached_key = (str(root), _fingerprint(root), want, read_pickles)
    hit = _MEMO.get(cached_key)
    if hit is not None:
        out = ImageFit(**{**asdict(hit), "components": list(hit.components)})
        out.free_bytes = free_bytes
        out.total_bytes = total_bytes
        out.device = getattr(device, "torch_device", "cpu")
        out.device_name = getattr(device, "name", "")
        _verdict(out, free_bytes, total_bytes)
        out.means = _sentence(out)
        return out

    try:
        parts, skipped = _weighted_parts(root)
    except _fit.Refusal as err:
        out.reason = err.sentence if hasattr(err, "sentence") else str(err)
        out.loadable = False
        out.verdict = "unknown"
        # NOTHING was read, so nothing here is a measurement. Leaving
        # `exact=True` on a path that priced no files claims a precision that
        # was never achieved, beside a `disk_bytes` of 0 that is unknown.
        out.exact = False
        out.means = out.reason
        return out

    out.excluded = skipped
    # Asked before the sizes, because "some weights are here" is not the same
    # question as "this can be built". A pipeline holding its VAE and nothing
    # else has weights to measure and no way to run.
    broken, absent = _gaps(root, _declared(root))
    # Both are REPORTED; only the unambiguous half refuses. An absent folder
    # is worth saying out loud beside a model and is not worth blocking a load
    # over — see `_gaps`.
    out.missing = broken
    out.absent = absent
    if broken:
        out.loadable = False

    if not parts:
        out.loadable = False
        out.verdict = "unknown"
        out.exact = False
        out.reason = (
            "no weight files anywhere under this directory, so it holds "
            "configuration and nothing to load — an interrupted download "
            "rather than a model that is ready."
        )
        out.means = out.reason
        return out

    variant, why = _choose_variant(parts)
    out.variant = variant
    if variant is None:
        out.loadable = False
        out.reason = why
    elif out.missing:
        out.reason = (
            "this cannot be loaded as it stands: " + "; ".join(out.missing) + "."
        )
    elif why:
        # A variant WAS found, but only because some component ships nothing
        # else. That is a load which succeeds and a default load which does
        # not, and the reader needs the flag either way.
        out.reason = why

    width = _dtype_bytes(want)
    disk_total = 0
    card_total: int | None = 0
    for name, files in sorted(parts.items()):
        groups = _by_variant(files)
        # Price the files a load would actually read. When the chosen variant
        # is absent for this component, diffusers falls back to the plain one.
        chosen = variant if variant is not None and variant in groups else ""
        picked = groups.get(chosen) or groups.get("") or []
        if not picked:
            # Nothing usable here at all — it still has files, so say which.
            chosen = sorted(groups)[0]
            picked = groups[chosen]
        disk, card, exact, trouble = _price(picked, width, read_pickles=read_pickles)
        out.components.append(
            Component(
                # A flat checkpoint has no component folders, and naming it
                # after `root.name` printed the cache's snapshot hash at the
                # reader.
                name=name or "weights",
                variant=chosen,
                variants=sorted(groups),
                files=len(picked),
                disk_bytes=disk,
                card_bytes=card,
                exact=exact,
                note=trouble,
            )
        )
        disk_total += disk
        if card is None or card_total is None:
            card_total = None
        else:
            card_total += card
        out.exact = out.exact and exact

    out.disk_bytes = disk_total
    out.card_bytes = card_total

    _verdict(out, free_bytes, total_bytes)
    out.means = _sentence(out)
    if cached_key is not None and len(_MEMO) < _MEMO_MAX:
        _MEMO[cached_key] = out
    return out


def _verdict(out: ImageFit, free_bytes: int | None, total_bytes: int | None) -> None:
    """Set `verdict` and `headroom_bytes` from a FRESH reading of the card.

    Separate from the pricing on purpose, and the reason is the one
    `devices._free_total` gives for keeping free memory off `Device`: what a
    checkpoint weighs is a property of the files and stays true, but how much
    room there is changes as soon as another process allocates. So the sizes
    can be memoised and this cannot — a cached "fits" is a verdict about a
    machine that no longer exists.
    """
    from . import fmt

    card_total = out.card_bytes
    if card_total is None:
        out.verdict = "unknown"
        out.reason = out.reason or (
            "part of this checkpoint could not be priced, so what it needs on "
            "the card is unknown rather than the sum of the parts that could."
        )
    elif not out.loadable:
        # A size is not a verdict. MEASURED on the stripped
        # `stable-diffusion-xl-base-1.0`: `loadable=False`, every component
        # missing its weights, and this block still answered `fits` from the
        # 49 MB it had managed to price. The picker only escaped publishing it
        # because `FitBadge` happens to bail on `!loadable` — `/api/image/local`
        # served "fits" for a pipeline with no weights in it.
        out.verdict = "unknown"
    elif free_bytes is None and total_bytes is None:
        out.verdict = "unknown"
        out.reason = out.reason or (
            f"{out.device_name or out.device} does not report how much memory "
            f"it has, so whether {fmt.bytes_si(card_total)} fits cannot be "
            f"answered from here."
        )
    else:
        # Free is the honest number when the driver gives it — another process
        # holding half the card is the difference between a load and an OOM.
        # Total is the fallback, and which one was used is reported.
        room = free_bytes if free_bytes is not None else total_bytes
        out.headroom_bytes = room - card_total
        if card_total >= room:
            out.verdict = "over"
        elif out.headroom_bytes < ACTIVATION_HEADROOM:
            out.verdict = "tight"
        else:
            out.verdict = "fits"


def _sentence(f: ImageFit) -> str:
    """What the verdict means, in the words a reader needs to act on it."""
    from . import fmt

    if not f.loadable:
        return f.reason

    size = (
        fmt.bytes_si(f.card_bytes) if f.card_bytes is not None else "an unknown amount"
    )
    about = "" if f.exact else "about "
    at = f"at {f.dtype}"
    where = f.device_name or f.device

    if f.verdict == "unknown":
        return f.reason or f"{about}{size} {at}; whether that fits here is unknown."

    room = f.free_bytes if f.free_bytes is not None else f.total_bytes
    which = "free right now" if f.free_bytes is not None else "in total"
    head = (
        f"{about}{size} of weights {at} against {fmt.bytes_si(room)} {which} on {where}"
    )
    if f.verdict == "over":
        return (
            f"{head} — this does not fit. Loading it will run out of memory "
            f"unless something else on the card is released first."
        )
    if f.verdict == "tight":
        return (
            f"{head}, leaving {fmt.bytes_si(max(f.headroom_bytes or 0, 0))} for "
            f"latents and attention maps. That is under the "
            f"{fmt.bytes_si(f.activation_headroom)} a capture pass typically "
            f"needs, so it may load and still fail during a measurement."
        )
    return f"{head}, leaving {fmt.bytes_si(f.headroom_bytes or 0)} to work in."


def measure_overhead(pipe, *, steps: int = 4, prompt: str = "a red apple") -> dict:
    """Peak allocation above the weights, from a real run on a real card.

    `ACTIVATION_HEADROOM` gates a verdict people plan around, so it is a number
    this project measured rather than one it chose. Kept here beside the
    constant so re-measuring is one call rather than an archaeology exercise.
    """
    import torch

    if not torch.cuda.is_available():
        return {"error": "no CUDA device, so there is nothing to measure here"}

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    resting = torch.cuda.memory_allocated()

    gen = torch.Generator(device="cpu").manual_seed(0)
    with torch.inference_mode():
        pipe(prompt=prompt, num_inference_steps=steps, generator=gen)

    peak = torch.cuda.max_memory_allocated()
    return {
        "weights_bytes": int(resting),
        "peak_bytes": int(peak),
        "overhead_bytes": int(peak - resting),
        "steps": steps,
    }
