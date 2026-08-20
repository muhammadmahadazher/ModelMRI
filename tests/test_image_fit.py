"""Will this image model run here — and the ways that answer goes wrong.

`image_fit` is read beside a Load button, so every number it produces is acted
on rather than noticed. The failures worth testing are not "does it add up";
they are the four ways a plausible-looking total is a lie:

  disk bytes quoted as card bytes   -> an F32 checkpoint loaded bf16 allocates
                                       half its file size, so the disk figure
                                       calls a comfortable model a tight one
  two copies of one component       -> `sd-turbo` ships its VAE twice, plain
                                       and fp16, and summing the folder
                                       over-quotes by 50%
  weights present, load impossible  -> a component with a config and no
                                       weights, or weights and no config, is a
                                       model listed as ready that fails at the
                                       click
  unknown reported as a number      -> "0 GB" and "fits" are both answers, and
                                       neither is available when the card did
                                       not say how much memory it has

Nothing here loads a model or touches a GPU. Checkpoints are built on disk in
the layouts real publishers use, and the device is a stub, so the verdicts are
about the calculator rather than about whoever's laptop runs the suite.
"""

from __future__ import annotations

import io
import json
import pickle
import zipfile
from dataclasses import dataclass

import pytest
from test_fit import write_safetensors  # the same real-file writer

from modelmri import image_fit


@dataclass
class FakeCard:
    """A device with stated memory, so a verdict does not depend on the runner."""

    kind: str = "cuda"
    torch_device: str = "cuda:0"
    name: str = "Test Card"
    vram_gb: float = 8.0
    dtype: str = "bfloat16"
    reason: str = "stubbed"


CARD = FakeCard()
EIGHT_GB = 8_000_000_000


def component(root, name, tensors, *, variant="", config=True):
    """One pipeline component, written the way diffusers writes them."""
    folder = root / name if name else root
    folder.mkdir(parents=True, exist_ok=True)
    if config:
        (folder / "config.json").write_text(json.dumps({"_class_name": "X"}))
    stem = "diffusion_pytorch_model"
    suffix = f".{variant}" if variant else ""
    write_safetensors(folder / f"{stem}{suffix}.safetensors", tensors)
    return folder


def index(root, components):
    """`model_index.json`, the publisher's statement of what will be built."""
    body = {"_class_name": "StableDiffusionPipeline"}
    for name, cls in components.items():
        body[name] = ["diffusers", cls] if cls else [None, None]
    (root / "model_index.json").write_text(json.dumps(body))


def priced(root, **kw):
    return image_fit.of(
        root, device=CARD, free_bytes=EIGHT_GB, total_bytes=EIGHT_GB, **kw
    )


def torch_pickle(path, storages):
    """A real torch `.bin`: a zip of `data.pkl` plus one blob per storage.

    Written by hand rather than with torch, because the point is that
    `_pickle_table` reads the format without importing torch or unpickling —
    a test that needed torch to build the file would not prove that.
    """

    class Storage:
        def __init__(self, kind, key, numel):
            self.kind, self.key, self.numel = kind, key, numel

    class Pickler(pickle.Pickler):
        def persistent_id(self, obj):
            if isinstance(obj, Storage):
                return ("storage", obj.kind, obj.key, "cpu", obj.numel)
            return None

    objs = [Storage(kind, str(i), numel) for i, (kind, numel) in enumerate(storages)]
    buf = io.BytesIO()
    Pickler(buf, protocol=2).dump({f"t{i}": o for i, o in enumerate(objs)})

    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr("archive/data.pkl", buf.getvalue())
        for obj in objs:
            width = image_fit._STORAGE_BYTES[obj.kind]
            zf.writestr(f"archive/data/{obj.key}", b"\0" * (obj.numel * width))


# ----------------------------------------------------- disk is not the card


def test_f32_on_disk_is_half_that_on_a_bf16_card(tmp_path):
    """The mistake this module exists to stop, on the commonest layout there is."""
    component(tmp_path, "unet", {"w": ("F32", [1024, 1024])})
    index(tmp_path, {"unet": "UNet2DConditionModel"})

    got = priced(tmp_path)

    assert got.disk_bytes == 1024 * 1024 * 4
    assert got.card_bytes == 1024 * 1024 * 2
    assert got.exact is True


def test_integer_tensors_are_not_halved_by_a_float_dtype(tmp_path):
    """`dtype=bfloat16` does not touch an int64 buffer, so neither does this."""
    component(tmp_path, "unet", {"w": ("F32", [512, 512]), "ids": ("I64", [128])})
    index(tmp_path, {"unet": "UNet2DConditionModel"})

    got = priced(tmp_path)

    assert got.card_bytes == 512 * 512 * 2 + 128 * 8


def test_a_checkpoint_already_at_the_target_width_does_not_shrink(tmp_path):
    component(tmp_path, "unet", {"w": ("F16", [256, 256])})
    index(tmp_path, {"unet": "UNet2DConditionModel"})

    got = priced(tmp_path)

    assert got.card_bytes == got.disk_bytes


# --------------------------------------------------------- one copy, not two


def test_two_variants_of_one_component_are_counted_once(tmp_path):
    """MEASURED on `sd-turbo`: its VAE folder holds plain AND fp16 files."""
    folder = component(tmp_path, "vae", {"w": ("F32", [1024, 512])})
    write_safetensors(
        folder / "diffusion_pytorch_model.fp16.safetensors", {"w": ("F16", [1024, 512])}
    )
    index(tmp_path, {"vae": "AutoencoderKL"})

    got = priced(tmp_path)
    (vae,) = got.components

    assert sorted(vae.variants) == ["", "fp16"]
    assert vae.variant == "", "the plain files are what a default load reads"
    assert got.card_bytes == 1024 * 512 * 2, "not the sum of both copies"


def test_a_component_shipping_only_a_variant_makes_the_load_ask_for_it(tmp_path):
    """The cached `sd-turbo` case: fp16-only text encoder, plain everything else."""
    component(tmp_path, "unet", {"w": ("F32", [64, 64])})
    component(tmp_path, "text_encoder", {"w": ("F16", [64, 64])}, variant="fp16")
    index(
        tmp_path,
        {"unet": "UNet2DConditionModel", "text_encoder": "CLIPTextModel"},
    )

    got = priced(tmp_path)

    assert got.variant == "fp16", "a plain load cannot find the text encoder"
    assert "text_encoder" in got.reason


def test_a_dot_in_the_publishers_own_filename_is_not_a_variant(tmp_path):
    """MEASURED: `facebook/sam3.1` ships `sam3.1_multiplex.pt`."""
    assert image_fit._variant_of("sam3.1_multiplex.pt") == ""
    assert image_fit._variant_of("model.fp16.safetensors") == "fp16"
    assert image_fit._variant_of("model.safetensors") == ""
    assert (
        image_fit._variant_of("diffusion_pytorch_model.fp16-00001-of-00002.safetensors")
        == "fp16"
    )


# ------------------------------------------------- present is not loadable


def test_a_component_with_a_config_and_no_weights_is_not_loadable(tmp_path):
    """MEASURED on the cached `segmind/tiny-sd`, whose unet lost its weights."""
    component(tmp_path, "vae", {"w": ("F32", [64, 64])})
    (tmp_path / "unet").mkdir()
    (tmp_path / "unet" / "config.json").write_text("{}")
    index(tmp_path, {"unet": "UNet2DConditionModel", "vae": "AutoencoderKL"})

    got = priced(tmp_path)

    assert got.loadable is False
    assert any("unet" in m for m in got.missing)
    assert got.card_bytes, "the parts that ARE here are still priced"


def test_a_component_with_weights_and_no_config_is_not_loadable(tmp_path):
    """MEASURED on the cached `sd-turbo` after Drive moved its text encoder config."""
    component(tmp_path, "unet", {"w": ("F32", [64, 64])})
    component(tmp_path, "text_encoder", {"w": ("F32", [64, 64])}, config=False)
    index(tmp_path, {"unet": "UNet2DConditionModel", "text_encoder": "CLIPTextModel"})

    got = priced(tmp_path)

    assert got.loadable is False
    assert any("text_encoder" in m and "config" in m for m in got.missing)


def test_a_component_the_loader_never_builds_is_not_a_gap(tmp_path):
    """`bk-sdm-tiny` declares a safety checker and ships only its config."""
    component(tmp_path, "unet", {"w": ("F32", [64, 64])})
    (tmp_path / "safety_checker").mkdir()
    (tmp_path / "safety_checker" / "config.json").write_text("{}")
    index(
        tmp_path,
        {
            "unet": "UNet2DConditionModel",
            "safety_checker": "StableDiffusionSafetyChecker",
        },
    )

    got = priced(tmp_path)

    assert got.loadable is True
    assert got.missing == []


def test_a_scheduler_folder_is_not_reported_as_missing_weights(tmp_path):
    component(tmp_path, "unet", {"w": ("F32", [64, 64])})
    (tmp_path / "scheduler").mkdir()
    (tmp_path / "scheduler" / "scheduler_config.json").write_text("{}")
    index(tmp_path, {"unet": "UNet2DConditionModel", "scheduler": "DDIMScheduler"})

    got = priced(tmp_path)

    assert got.missing == []


def test_a_directory_holding_only_configs_says_so(tmp_path):
    (tmp_path / "unet").mkdir()
    (tmp_path / "unet" / "config.json").write_text("{}")
    index(tmp_path, {"unet": "UNet2DConditionModel"})

    got = priced(tmp_path)

    assert got.loadable is False
    assert got.card_bytes is None, "unknown, never 0"
    assert got.verdict == "unknown"


# --------------------------------------------------------- pickles, exactly


def test_a_torch_pickle_is_priced_from_its_own_tensor_table(tmp_path):
    """MEASURED on `DiT-XL-2-256`: 538 storages, byte-exact against the zip."""
    folder = tmp_path / "transformer"
    folder.mkdir()
    (folder / "config.json").write_text("{}")
    torch_pickle(
        folder / "diffusion_pytorch_model.bin",
        [("torch.FloatStorage", 100_000), ("torch.FloatStorage", 50_000)],
    )
    index(tmp_path, {"transformer": "Transformer2DModel"})

    got = priced(tmp_path)

    assert got.exact is True, "a pickle has a tensor table too"
    assert got.disk_bytes == 150_000 * 4
    assert got.card_bytes == 150_000 * 2, "halved, like any other F32 checkpoint"


def test_a_pickles_integer_storages_are_left_alone(tmp_path):
    folder = tmp_path / "transformer"
    folder.mkdir()
    (folder / "config.json").write_text("{}")
    torch_pickle(
        folder / "diffusion_pytorch_model.bin",
        [("torch.FloatStorage", 1000), ("torch.LongStorage", 10)],
    )
    index(tmp_path, {"transformer": "Transformer2DModel"})

    got = priced(tmp_path)

    assert got.card_bytes == 1000 * 2 + 10 * 8


def test_an_unreadable_pickle_falls_back_and_says_it_did(tmp_path):
    folder = tmp_path / "transformer"
    folder.mkdir()
    (folder / "config.json").write_text("{}")
    (folder / "diffusion_pytorch_model.bin").write_bytes(b"not a zip at all")
    index(tmp_path, {"transformer": "Transformer2DModel"})

    got = priced(tmp_path)

    assert got.exact is False
    assert got.card_bytes == 16, "the file's own size, which over-quotes"
    (part,) = got.components
    assert "tensor table could not be read" in part.note


def test_reading_a_pickle_never_unpickles_it(tmp_path):
    """The whole point of opcode-walking: a hostile `__reduce__` must not run."""

    class Boom:
        def __reduce__(self):
            return (exec, ("raise SystemExit('a pickle ran')",))

    buf = io.BytesIO()
    pickle.Pickler(buf, protocol=2).dump({"payload": Boom()})
    path = tmp_path / "hostile.bin"
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr("archive/data.pkl", buf.getvalue())

    # No storages to find, and above all no execution.
    table, blobs = image_fit._pickle_table(path)
    assert table == {}
    assert blobs == {}


# ------------------------------------------------------ unknown is an answer


def test_a_card_that_does_not_report_its_memory_gets_no_verdict(tmp_path):
    component(tmp_path, "unet", {"w": ("F32", [64, 64])})
    index(tmp_path, {"unet": "UNet2DConditionModel"})

    got = image_fit.of(tmp_path, device=CARD, free_bytes=None, total_bytes=None)

    assert got.verdict == "unknown"
    assert got.card_bytes, "what it COSTS is still known"
    assert "does not report" in got.reason


def test_a_model_larger_than_the_card_is_refused_not_rounded(tmp_path):
    component(tmp_path, "unet", {"w": ("F32", [4096, 4096])})
    index(tmp_path, {"unet": "UNet2DConditionModel"})

    got = image_fit.of(
        tmp_path, device=CARD, free_bytes=8_000_000, total_bytes=EIGHT_GB
    )

    assert got.verdict == "over"
    assert "does not fit" in got.means


def test_weights_that_just_fit_are_called_tight_not_fine(tmp_path):
    """Room for the weights is not room for the latents and attention maps."""
    component(tmp_path, "unet", {"w": ("F16", [1024, 1024])})  # 2 MB
    index(tmp_path, {"unet": "UNet2DConditionModel"})

    free = 2 * 1024 * 1024 + image_fit.ACTIVATION_HEADROOM // 2
    got = image_fit.of(tmp_path, device=CARD, free_bytes=free, total_bytes=EIGHT_GB)

    assert got.verdict == "tight"
    assert "may load and still fail" in got.means


def test_free_memory_is_preferred_over_total(tmp_path):
    """Another process holding the card is the difference between load and OOM."""
    component(tmp_path, "unet", {"w": ("F32", [2048, 2048])})  # 8 MB card
    index(tmp_path, {"unet": "UNet2DConditionModel"})

    got = image_fit.of(
        tmp_path, device=CARD, free_bytes=4_000_000, total_bytes=EIGHT_GB
    )

    assert got.verdict == "over", "the card is big and it is not free"
    assert "free right now" in got.means


def test_the_headroom_it_used_is_reported_not_just_applied(tmp_path):
    component(tmp_path, "unet", {"w": ("F32", [64, 64])})
    index(tmp_path, {"unet": "UNet2DConditionModel"})

    got = priced(tmp_path)

    assert got.activation_headroom == image_fit.ACTIVATION_HEADROOM
    assert got.to_dict()["activation_headroom"] == image_fit.ACTIVATION_HEADROOM


# ------------------------------------------------------------ shape of a row


def test_a_flat_checkpoint_is_not_named_after_its_snapshot_hash(tmp_path):
    """A transformers model has no component folders; the row still needs a name."""
    write_safetensors(tmp_path / "model.safetensors", {"w": ("F32", [64, 64])})
    (tmp_path / "config.json").write_text("{}")

    got = priced(tmp_path)
    (only,) = got.components

    assert only.name == "weights"
    assert got.card_bytes == 64 * 64 * 2


def test_an_unknown_component_name_is_counted_like_any_other(tmp_path):
    """Nothing here is a name list, so a component nobody anticipated counts."""
    component(tmp_path, "unet", {"w": ("F32", [64, 64])})
    component(tmp_path, "prior_posterior_resampler", {"w": ("F32", [64, 64])})
    index(tmp_path, {"unet": "UNet2DConditionModel"})

    got = priced(tmp_path)

    assert {c.name for c in got.components} == {"unet", "prior_posterior_resampler"}
    assert got.card_bytes == 2 * 64 * 64 * 2


def test_an_excluded_component_is_reported_rather_than_dropped(tmp_path):
    component(tmp_path, "unet", {"w": ("F32", [64, 64])})
    component(tmp_path, "safety_checker", {"w": ("F32", [1024, 1024])})
    index(tmp_path, {"unet": "UNet2DConditionModel"})

    got = priced(tmp_path)

    assert {c.name for c in got.components} == {"unet"}
    assert any("safety_checker" in e for e in got.excluded)


def test_an_empty_directory_is_unknown_rather_than_free(tmp_path):
    got = priced(tmp_path)

    assert got.loadable is False
    assert got.card_bytes is None
    assert got.verdict == "unknown"


def test_the_dtype_asked_for_beats_the_devices_preference(tmp_path):
    component(tmp_path, "unet", {"w": ("F32", [64, 64])})
    index(tmp_path, {"unet": "UNet2DConditionModel"})

    got = priced(tmp_path, dtype="float32")

    assert got.dtype == "float32"
    assert got.card_bytes == got.disk_bytes


def test_two_formats_of_one_component_are_counted_once(tmp_path):
    """MEASURED on `google/vit-base-patch16-224`, which ships `model.safetensors`
    AND `pytorch_model.bin` for the same tensors. Priced with both it came to
    346 MB on the card — exactly twice the 173 MB one copy occupies. The fp16
    twin was already handled; this is the same defect on the format axis."""
    folder = component(tmp_path, "unet", {"w": ("F32", [1024, 512])})
    torch_pickle(folder / "pytorch_model.bin", [("torch.FloatStorage", 1024 * 512)])
    index(tmp_path, {"unet": "UNet2DConditionModel"})

    got = priced(tmp_path)
    (unet,) = got.components

    assert unet.files == 1, "safetensors wins, matching `use_safetensors`"
    assert got.card_bytes == 1024 * 512 * 2, "not the sum of both formats"
    assert any("pytorch_model.bin" in e for e in got.excluded), (
        "the copy that was NOT counted is reported, not dropped silently"
    )


def test_a_storage_two_tensors_share_is_counted_once(tmp_path):
    """Tied embeddings put one storage under two state-dict keys, and
    `pickle.Pickler` never memoizes a persistent id — so the walker saw two
    `BINPERSID`s naming one `data/N` and counted those bytes twice."""
    folder = tmp_path / "transformer"
    folder.mkdir()
    (folder / "config.json").write_text("{}")

    class Shared:
        pass

    class Pickler(pickle.Pickler):
        def persistent_id(self, obj):
            if isinstance(obj, Shared):
                return ("storage", "torch.FloatStorage", "0", "cpu", 1000)
            return None

    one = Shared()
    buf = io.BytesIO()
    Pickler(buf, protocol=2).dump({"embed": one, "lm_head": one})
    path = folder / "diffusion_pytorch_model.bin"
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr("archive/data.pkl", buf.getvalue())
        zf.writestr("archive/data/0", b"\0" * 4000)
    index(tmp_path, {"transformer": "Transformer2DModel"})

    table, _ = image_fit._pickle_table(path)
    assert table == {"0": ("torch.FloatStorage", 1000)}, "one entry, not two"
    assert priced(tmp_path).card_bytes == 1000 * 2


def test_a_numel_the_archive_contradicts_is_not_reported_as_exact(tmp_path):
    """`numel` is read out of a file this deliberately does not execute, and
    nothing else bounded it: a pickle claiming 2**62 elements produced an
    eight-exabyte figure flagged as a measurement. The zip already records how
    many bytes the storage holds, so the two are simply compared."""
    folder = tmp_path / "transformer"
    folder.mkdir()
    (folder / "config.json").write_text("{}")

    class Liar:
        pass

    class Pickler(pickle.Pickler):
        def persistent_id(self, obj):
            if isinstance(obj, Liar):
                return ("storage", "torch.FloatStorage", "0", "cpu", 2**62)
            return None

    buf = io.BytesIO()
    Pickler(buf, protocol=2).dump({"w": Liar()})
    path = folder / "diffusion_pytorch_model.bin"
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr("archive/data.pkl", buf.getvalue())
        zf.writestr("archive/data/0", b"\0" * 400)
    index(tmp_path, {"transformer": "Transformer2DModel"})

    got = priced(tmp_path)

    assert got.exact is False
    assert got.card_bytes is None, "unknown, not eight exabytes"
    assert any("archive stores" in c.note for c in got.components)


def test_a_boolean_numel_is_not_one_element(tmp_path):
    """`isinstance(True, int)` is True in Python, so the bool guard has to come
    first or a corrupt table quietly contributes an element."""

    class Flag:
        pass

    class Pickler(pickle.Pickler):
        def persistent_id(self, obj):
            if isinstance(obj, Flag):
                return ("storage", "torch.FloatStorage", "0", "cpu", True)
            return None

    buf = io.BytesIO()
    Pickler(buf, protocol=2).dump({"w": Flag()})
    path = tmp_path / "odd.bin"
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr("archive/data.pkl", buf.getvalue())

    table, _ = image_fit._pickle_table(path)
    assert table == {}, "a bool is not a length"


def test_a_pipeline_that_cannot_load_never_reports_that_it_fits(tmp_path):
    """MEASURED on the stripped `stable-diffusion-xl-base-1.0`: every component
    had lost its weights, and the verdict block still answered `fits` from the
    49 MB it had priced elsewhere. The badge hid it by luck; the route
    published it."""
    component(tmp_path, "vae", {"w": ("F32", [64, 64])})
    (tmp_path / "unet").mkdir()
    (tmp_path / "unet" / "config.json").write_text("{}")
    index(tmp_path, {"unet": "UNet2DConditionModel", "vae": "AutoencoderKL"})

    got = priced(tmp_path)

    assert got.loadable is False
    assert got.verdict == "unknown", "never `fits` for something that cannot open"


def test_a_stray_adapter_beside_a_pipeline_is_not_the_model(tmp_path):
    """MEASURED on SDXL after Drive stripped it: the flat-checkpoint fallback
    found `sd_xl_offset_example-lora_1.0.safetensors` in the root and priced a
    7 GB pipeline at 49 MB, `loadable: true`, `verdict: "fits"`. A green badge
    on a click that cannot work is what this module exists to prevent."""
    for part in ("unet", "vae", "text_encoder"):
        (tmp_path / part).mkdir()
        (tmp_path / part / "config.json").write_text("{}")
    write_safetensors(
        tmp_path / "sd_xl_offset_example-lora_1.0.safetensors", {"w": ("F32", [64, 64])}
    )
    index(
        tmp_path,
        {
            "unet": "UNet2DConditionModel",
            "vae": "AutoencoderKL",
            "text_encoder": "CLIPTextModel",
        },
    )

    got = priced(tmp_path)

    assert got.loadable is False
    assert got.verdict != "fits"
    assert got.components == [], "a LoRA in the root is not the pipeline"


def test_a_weight_file_the_loader_never_opens_is_not_the_model(tmp_path):
    """MEASURED on the two SAM checkpoints in this machine's cache.

    `facebook/sam3` ships `model.safetensors` and loads. `facebook/sam3.1`
    ships only `sam3.1_multiplex.pt` and fails with OSError — `from_pretrained`
    opens `model.safetensors`, `pytorch_model.bin` or a shard index, and does
    not search a folder for whatever looks like tensors.

    This rule was briefly relaxed on the theory that a root `config.json` made
    any weight file beside it the model, because refusing sam3.1 looked like a
    false refusal hiding a working checkpoint. It was not working. The
    relaxation put a green badge and a 3.5 GB size on a click that cannot
    succeed, which is the outcome this module exists to prevent.
    """
    (tmp_path / "config.json").write_text(
        json.dumps({"architectures": ["Sam3VideoModel"]})
    )
    write_safetensors(tmp_path / "sam3.1_multiplex.pt", {"w": ("F32", [64, 64])})

    got = priced(tmp_path)

    assert got.loadable is False
    assert got.verdict != "fits"
    assert got.card_bytes is None, "unknown, not a size for something unloadable"
    assert "sam3.1_multiplex.pt" in got.reason, "name the file that is here"
    assert "model.safetensors" in got.reason, "and the name the loader wants"


def test_the_same_checkpoint_with_a_canonical_name_does_load(tmp_path):
    """The other half, so the rule above cannot become "refuse everything flat"."""
    (tmp_path / "config.json").write_text(
        json.dumps({"architectures": ["Sam3VideoModel"]})
    )
    write_safetensors(tmp_path / "model.safetensors", {"w": ("F32", [64, 64])})

    got = priced(tmp_path)

    assert got.loadable is True
    assert got.card_bytes == 64 * 64 * 2


def test_a_stray_file_beside_canonical_weights_is_ignored_not_added(tmp_path):
    """`facebook/sam3` ships `sam3.pt` beside `model.safetensors`. One load
    reads one of them, and counting both was the 2x over-quote that took its
    published size from 3.45 GB to a true 1.72 GB."""
    (tmp_path / "config.json").write_text("{}")
    write_safetensors(tmp_path / "model.safetensors", {"w": ("F32", [1024, 512])})
    write_safetensors(tmp_path / "sam3.pt", {"w": ("F32", [1024, 512])})

    got = priced(tmp_path)

    assert got.loadable is True
    assert got.card_bytes == 1024 * 512 * 2, "one copy, not two"


def test_a_component_name_from_the_index_cannot_steer_the_walk(tmp_path):
    """`model_index.json` is downloaded, and on Windows `root / "C:/Windows"`
    resolves to the ABSOLUTE path — so a malformed index would have a listing
    route walk an arbitrary directory and publish what it found there."""
    component(tmp_path, "unet", {"w": ("F32", [64, 64])})
    index(tmp_path, {"unet": "UNet2DConditionModel"})
    body = json.loads((tmp_path / "model_index.json").read_text())
    body["../../../etc"] = ["diffusers", "Evil"]
    (tmp_path / "model_index.json").write_text(json.dumps(body))

    got = priced(tmp_path)

    assert got.loadable is False
    assert any("not a name a component folder can have" in m for m in got.missing)


def test_a_variant_that_covers_nothing_blocks_the_row(tmp_path):
    """The second way to be unopenable, and it leaves `missing` EMPTY: no
    single variant covers every component. Gated on that list, such a row came
    back pickable and badge-less — an ordinary healthy model until the click."""
    component(tmp_path, "unet", {"w": ("F16", [64, 64])}, variant="fp16")
    component(tmp_path, "vae", {"w": ("F16", [64, 64])}, variant="bf16")
    index(tmp_path, {"unet": "UNet2DConditionModel", "vae": "AutoencoderKL"})

    got = priced(tmp_path)

    assert got.variant is None
    assert got.loadable is False
    assert got.missing == [], "this is the empty-missing case, on purpose"
    assert got.reason, "so the reason has to carry it"
    assert got.verdict != "fits"


def test_pricing_can_skip_pickles_and_says_when_it_did(tmp_path):
    """The listing route reads dozens of models per page load, and opening a
    `.bin`'s zip materialises the whole file — 69 s for one 346 MB pickle on a
    Drive-backed cache. Skipped, the figure must be marked inexact and say so
    rather than looking like a measurement."""
    folder = tmp_path / "transformer"
    folder.mkdir()
    (folder / "config.json").write_text("{}")
    torch_pickle(folder / "diffusion_pytorch_model.bin", [("torch.FloatStorage", 1000)])
    index(tmp_path, {"transformer": "Transformer2DModel"})

    walked = priced(tmp_path)
    skipped = priced(tmp_path, read_pickles=False)

    assert walked.exact is True and walked.card_bytes == 1000 * 2
    assert skipped.exact is False
    assert skipped.card_bytes > walked.card_bytes, "the file size over-quotes"
    assert any("priced from its size" in c.note for c in skipped.components)


@pytest.mark.parametrize("verdict", ["fits", "tight", "over", "unknown"])
def test_every_verdict_comes_with_a_sentence(tmp_path, verdict):
    """A word in a badge is not an explanation, and this one is read under stress."""
    component(tmp_path, "unet", {"w": ("F32", [512, 512])})
    index(tmp_path, {"unet": "UNet2DConditionModel"})
    room = {
        "fits": EIGHT_GB,
        "tight": 512 * 512 * 2 + 1,
        "over": 1000,
        "unknown": None,
    }[verdict]

    got = image_fit.of(
        tmp_path, device=CARD, free_bytes=room, total_bytes=room and EIGHT_GB
    )

    assert got.verdict == verdict
    assert len(got.means) > 30, got.means
