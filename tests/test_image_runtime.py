"""Holding one diffusion pipeline, and refusing before it costs anything.

The ORDER of the refusals is the design: identify from JSON, scan opcodes,
price from real bytes, and only then load. Each step is cheaper than the one
after it, so a checkpoint that will not work is refused before the expensive
part happens rather than twenty minutes into a download.

Two tests here exist because real data broke the first version. The pipeline
cached on the machine this was written on ships `.bin` pickles rather than
safetensors, and the byte counter did not see them at all — so it read 1.7 GB
as 0.00 GB, `capacity.guard` correctly treated that as "nothing published to
go on", and the refusal that exists to prevent an OOM would never have fired
on a real model.
"""

from __future__ import annotations

import json

import pytest

from modelmri import image_runtime as ir
from modelmri.errors import BadRequest, Refusal


def _pipeline(tmp_path, *, suffix=".safetensors", mb=4):
    """A diffusers pipeline directory with weights of a stated size."""
    root = tmp_path / "pipe"
    root.mkdir(parents=True, exist_ok=True)
    (root / "model_index.json").write_text(
        json.dumps(
            {
                "_class_name": "StableDiffusionPipeline",
                "unet": ["diffusers", "UNet2DConditionModel"],
                "vae": ["diffusers", "AutoencoderKL"],
                "text_encoder": ["transformers", "CLIPTextModel"],
            }
        ),
        encoding="utf-8",
    )
    unet = root / "unet"
    unet.mkdir(exist_ok=True)
    (unet / "config.json").write_text(
        json.dumps({"_class_name": "UNet2DConditionModel", "cross_attention_dim": 768}),
        encoding="utf-8",
    )
    (unet / f"diffusion_pytorch_model{suffix}").write_bytes(b"\x00" * (mb * 1_000_000))
    return root


def _stub_load(monkeypatch):
    """Stop before the real `from_pretrained`; the sequence is under test."""
    monkeypatch.setattr(
        ir, "_load_pipeline", lambda *a, **k: (object(), "cpu", "float32", 0.0)
    )


# ----------------------------------------------------- nothing held yet


def test_an_empty_handle_says_what_to_do_rather_than_just_false():
    h = ir.ImageHandle()
    assert h.status().loaded is False
    assert "Nothing has been loaded yet" in h.status().means()


def test_asking_for_the_pipeline_before_loading_is_a_refusal():
    with pytest.raises(ir.NotLoaded, match="every measurement here runs the real"):
        ir.ImageHandle().require()


# ----------------------------------------------- the refusals, in cost order


def test_an_unnamed_model_is_refused_rather_than_defaulted():
    """A default checkpoint would silently decide which panels apply."""
    with pytest.raises(BadRequest, match="no default worth guessing"):
        ir.ImageHandle().load("   ")


def test_an_unidentifiable_checkpoint_is_refused_before_anything_is_read(tmp_path):
    """First and cheapest: JSON only. A panel drawn for the wrong family is a
    picture of something that does not exist."""
    empty = tmp_path / "nothing"
    empty.mkdir()
    with pytest.raises(Refusal, match="does not exist"):
        ir.ImageHandle().load(str(empty))


def test_a_pipeline_carrying_an_executable_payload_is_refused_before_loading(tmp_path):
    """`from_pretrained` unpickles a `.bin` without asking, and a pipeline is a
    directory of them. That window is the whole reason the scan is in the
    sequence rather than something the user is trusted to remember."""
    import pickle

    from modelmri import weights_scan

    root = _pipeline(tmp_path, suffix=".bin", mb=0)

    class _RunsAShellCommand:
        def __reduce__(self):
            import os

            return (os.system, ("echo pwned",))

    (root / "unet" / "diffusion_pytorch_model.bin").write_bytes(
        pickle.dumps(_RunsAShellCommand())
    )
    with pytest.raises(weights_scan.Unsafe, match="executes something"):
        ir.ImageHandle().load(str(root))


def test_a_pipeline_beside_a_resident_model_is_refused_with_both_numbers(tmp_path):
    """One process, two sets of weights. Unlike a single oversized model,
    neither of these can be offloaded to rescue the other."""
    from modelmri import capacity

    root = _pipeline(tmp_path, mb=4)
    with pytest.raises(capacity.TooBig) as caught:
        ir.ImageHandle().load(str(root), already_held_bytes=6_000_000_000)
    said = str(caught.value)
    assert "6.0 GB" in said
    assert "unload the other one first" in said


def test_confirming_lets_a_second_set_of_weights_through(tmp_path, monkeypatch):
    """Their machine, their decision — but it has to be said out loud."""
    _stub_load(monkeypatch)
    status = ir.ImageHandle().load(
        str(_pipeline(tmp_path, mb=4)),
        already_held_bytes=6_000_000_000,
        confirm=True,
    )
    assert status.loaded


# --------------------------------------------------- the bug real data found


def test_bin_weights_are_counted_and_not_reported_as_zero(tmp_path):
    """THE bug. The pipeline cached on the development machine ships `.bin`
    rather than safetensors, and counting only safetensors read 1.7 GB as
    0.00 GB — so the guard saw "nothing published to go on", allowed it
    through as unknown, and never fired."""
    assert ir._weights_bytes(_pipeline(tmp_path, suffix=".bin", mb=6)) >= 6_000_000


def test_safetensors_and_bin_are_counted_the_same_way(tmp_path):
    def weights(root, suffix):
        unet = root / "unet"
        unet.mkdir(parents=True, exist_ok=True)
        (unet / f"diffusion_pytorch_model{suffix}").write_bytes(b"\x00" * 3_000_000)
        return root

    assert ir._weights_bytes(weights(tmp_path / "a", ".safetensors")) == (
        ir._weights_bytes(weights(tmp_path / "b", ".bin"))
    )


def test_the_download_asks_for_bin_as_well_as_safetensors():
    """Excluding `.bin` fetched a directory of configs with no weights in it,
    and `from_pretrained` then failed about a missing file rather than saying
    the honest thing."""
    import inspect

    source = inspect.getsource(ir._resolve)
    assert '"*.bin"' in source
    assert '"*/*.bin"' in source


def test_a_component_holding_only_json_does_not_break_the_count(tmp_path):
    """`scheduler/` and `tokenizer/` hold JSON only, and a pipeline is mostly
    those."""
    root = _pipeline(tmp_path, mb=4)
    (root / "scheduler").mkdir()
    (root / "scheduler" / "scheduler_config.json").write_text("{}", encoding="utf-8")
    assert ir._weights_bytes(root) >= 4_000_000


# ------------------------------------------------------------- unloading


def test_unload_reports_what_it_dropped(tmp_path, monkeypatch):
    _stub_load(monkeypatch)
    h = ir.ImageHandle()
    h.load(str(_pipeline(tmp_path, mb=1)))
    status = h.unload()
    assert status.loaded is False
    assert "unloaded and its memory handed back" in status.reason
    assert h.pipe is None


def test_unload_on_an_empty_handle_is_not_an_error():
    assert ir.ImageHandle().unload().loaded is False


# ------------------------------------------------------ what the status says


def test_an_unconditional_pipeline_says_there_are_no_word_maps(tmp_path, monkeypatch):
    """`cross_attention_dim: 0` means no prompt attention exists, and drawing
    a word-to-pixel map anyway would be inventing it."""
    root = _pipeline(tmp_path, mb=1)
    (root / "unet" / "config.json").write_text(
        json.dumps({"_class_name": "UNet2DModel", "cross_attention_dim": 0}),
        encoding="utf-8",
    )
    _stub_load(monkeypatch)
    assert "UNCONDITIONAL" in ir.ImageHandle().load(str(root)).means()


def test_the_status_carries_what_can_be_measured(tmp_path, monkeypatch):
    """A panel asks rather than infers, and an unknown family offers nothing."""
    _stub_load(monkeypatch)
    status = ir.ImageHandle().load(str(_pipeline(tmp_path, mb=1)))
    assert "cross_attention" in status.capabilities
    assert status.to_dict()["capabilities"] == status.capabilities


# -------------------------------------------- one loader per family it names


def test_every_family_imaging_can_name_has_a_loader():
    """The defect this locks: `imaging.detect` named seven families and
    `_load_pipeline` knew ONE of them.

    Every ViT, CLIP, detector, segmenter and VLM went through
    `DiffusionPipeline.from_pretrained` and came back as a raw `diffusers`
    OSError about a missing `model_index.json` — a sentence about a file the
    user never heard of, for a checkpoint that is not a pipeline and was
    never going to have one. Measured on a real one: `facebook/sam3` spent
    fifteen minutes downloading and then said that.

    Structural on purpose. Adding a family to `imaging` without a loader has
    to fail here rather than at the end of somebody's download.
    """
    from modelmri import imaging

    named = {
        imaging.UNET_DIFFUSION,
        imaging.DIT_DIFFUSION,
        imaging.VIT,
        imaging.CLIP,
        imaging.DETECTION,
        imaging.SEGMENTATION,
        imaging.VLM,
    }
    covered = set(ir._TRANSFORMERS_LOADERS) | set(ir._DIFFUSION_FAMILIES)
    assert named <= covered, f"no loader for {named - covered}"
    # UNKNOWN must NOT be covered: it is refused before a loader is chosen,
    # and giving it one would be giving a guess a way to run.
    assert imaging.UNKNOWN not in covered


def test_a_family_with_no_loader_is_named_rather_than_guessed_at(monkeypatch):
    """Unreachable through `load`, which refuses an unknown family first.
    Stated anyway — a family added to `imaging` and not to the table must say
    so rather than fall through to whichever loader is written first."""
    with pytest.raises(Refusal) as caught:
        ir._load_pipeline(
            __import__("pathlib").Path("."),
            family="a_family_nobody_wrote_a_loader_for",
            device="cpu",
            dtype="float32",
        )
    said = str(caught.value)
    assert "a_family_nobody_wrote_a_loader_for" in said
    assert "no loader" in said
    assert "nothing was loaded" in said


def test_the_configs_come_down_before_any_weight_does(monkeypatch):
    """The order in `load` claims each step refuses before the next one costs
    anything, and step zero used to break that promise: `_resolve` pulled the
    whole repository, and only then did `imaging.detect` get to say the family
    was one nothing here can open.

    Measured after the fix on a real uncached repo: 1,083 bytes fetched, zero
    weight files on disk, refused in 3.5s.
    """
    asked = []

    def _spy(repo, allow, *, local_ok=True):
        asked.append(list(allow))
        return __import__("pathlib").Path(".")

    monkeypatch.setattr(ir, "_snapshot", _spy)
    ir._resolve_configs("owner/name")
    assert asked == [["*.json", "*/*.json"]]
    assert not any(p.endswith((".safetensors", ".bin")) for p in asked[0]), (
        "a config fetch must not name a weight pattern"
    )


def test_the_weight_fetch_is_a_superset_of_the_config_fetch(monkeypatch):
    """Otherwise the second call re-downloads JSON the first already has, or
    worse, arrives at a directory missing the config that named the family."""
    asked = []
    monkeypatch.setattr(
        ir,
        "_snapshot",
        lambda repo, allow, *, local_ok=True: (
            asked.append(list(allow)),
            __import__("pathlib").Path("."),
        )[1],
    )
    ir._resolve_configs("owner/name")
    ir._resolve("owner/name")
    configs, weights = asked
    assert set(configs) <= set(weights)
    assert "*.safetensors" in weights


# ------------------------------------------- the preprocessor, and why not


def test_a_missing_package_is_not_reported_as_a_missing_file(monkeypatch):
    """The refusal that lied.

    `facebook/sam3` publishes a preprocessor. Loading it raised `ImportError`
    because torchvision was not installed, a broad `except` swallowed that,
    and the refusal said the checkpoint "did not publish an image
    preprocessor" — a claim about the model, for a fact about the machine. It
    sent a reader looking for a file that is right there.

    One is fixable in a single command and the other is not fixable at all,
    so they must not share a sentence.
    """

    class _Boom:
        @staticmethod
        def from_pretrained(*_a, **_k):
            raise ImportError(
                "Sam3VideoProcessor requires the Torchvision library but it "
                "was not found in your environment."
            )

    # Dotted string: transformers is a lazy module, so setting an attribute on
    # the module object only lands once something else has materialised it.
    for name in ir._PROCESSOR_CLASSES:
        monkeypatch.setattr(f"transformers.{name}", _Boom)

    found, why = ir._load_processor(__import__("pathlib").Path("."))
    assert found is None
    assert "torchvision" in why
    assert "not installed" in why
    assert "published one" in why, "it must not read as the checkpoint's fault"


def test_a_checkpoint_with_no_processor_says_that_instead(monkeypatch):
    """The other branch, so the two never collapse into one another."""

    class _None:
        @staticmethod
        def from_pretrained(*_a, **_k):
            raise OSError("no preprocessor_config.json here")

    for name in ir._PROCESSOR_CLASSES:
        monkeypatch.setattr(f"transformers.{name}", _None)

    found, why = ir._load_processor(__import__("pathlib").Path("."))
    assert found is None
    assert "torchvision" not in why
    assert "OSError" in why, "the class, so a reader knows what kind of failure"
    # And never the library's own text, which carries paths from this machine.
    assert "preprocessor_config.json here" not in why


def test_a_composite_processor_yields_its_image_half(monkeypatch):
    """A multimodal checkpoint publishes ONE object holding a tokenizer and an
    image processor. `AutoImageProcessor` does not load it, which read
    `facebook/sam3` as having no preprocessor at all — and handing the
    composite through unchanged would give the sweep a tokenizer where it
    expects something that turns a picture into a tensor."""

    class _Image:
        pass

    inner = _Image()

    class _Composite:
        image_processor = inner

    class _Auto:
        @staticmethod
        def from_pretrained(*_a, **_k):
            return _Composite()

    monkeypatch.setattr("transformers.AutoImageProcessor", _Auto)
    found, why = ir._load_processor(__import__("pathlib").Path("."))
    assert found is inner
    assert why == ""


def test_the_processor_is_dropped_with_the_model_it_belongs_to():
    """A processor outliving its model is the previous checkpoint's
    normalisation applied to the next one's input — the exact wrong-tensor
    failure, arriving through the back door and looking like a real answer."""
    handle = ir.ImageHandle()
    handle.processor = object()
    handle.processor_reason = "stale"
    handle.unload()
    assert handle.processor is None
    assert handle.processor_reason == ""


def test_a_measurement_needing_a_processor_refuses_with_the_reason(monkeypatch):
    """Not "nothing is loaded" — something IS loaded. The two failures need
    two different actions, so they get two different sentences."""
    handle = ir.ImageHandle()
    handle.pipe = object()
    handle.status_ = ir.ImageStatus(loaded=True, repo="owner/name")
    handle.processor_reason = "reading its preprocessor needs `torchvision`"

    with pytest.raises(Refusal) as caught:
        handle.require_processor()
    said = str(caught.value)
    assert "owner/name" in said
    assert "torchvision" in said
    assert "No image model is loaded" not in said


# ------------------------------------------- the bomb, and where it is caught


def test_a_decompression_bomb_is_refused_from_the_header(monkeypatch):
    """The bound existed and ran AFTER `picture.load()`, so the expansion it
    refuses had already happened and the refusal was a message about memory
    that was already gone.

    `Image.open` is lazy — it reads the header and gives `.size` without
    decoding a pixel — so the check belongs between `open` and `load`.

    Measured: 9,000x9,000 (81M pixels) from 78,702 compressed bytes is refused
    in 0.001s with a 0.3 MB peak allocation.
    """
    import base64
    import io as _io
    import tracemalloc

    from PIL import Image

    from modelmri import image_input

    big = Image.new("L", (9000, 9000), 0)
    buf = _io.BytesIO()
    big.save(buf, "PNG", optimize=True)
    url = "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()

    tracemalloc.start()
    try:
        with pytest.raises(image_input.BadImage) as caught:
            image_input.decode(url)
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()

    said = str(caught.value)
    assert "81,000,000 pixels" in said
    assert "before anything is decompressed" in said
    # 81M pixels at one byte each would be 81 MB. Nothing near that was
    # allocated, which is the whole claim.
    assert peak < 20_000_000, f"the image was decompressed anyway: {peak:,} bytes"


def test_the_imaging_library_s_own_bound_gets_an_honest_sentence():
    """Pillow has its own limit and it can fire first. Without a dedicated arm
    the refusal said "those bytes are not an image this can decode", which is
    FALSE — it is a perfectly good image that is too large — and it sends
    somebody to re-export a file whose only problem is its size."""
    import base64
    import io as _io

    from PIL import Image

    from modelmri import image_input

    huge = Image.MAX_IMAGE_PIXELS or 0
    side = int((huge * 2) ** 0.5) + 1
    big = Image.new("L", (side, side), 0)
    buf = _io.BytesIO()
    big.save(buf, "PNG", optimize=True)
    url = "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()

    with pytest.raises(image_input.BadImage) as caught:
        image_input.decode(url)
    said = str(caught.value)
    assert "too large to decode safely" in said
    assert "not an image this can decode" not in said, (
        "a too-big image was reported as an unreadable one"
    )
