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
