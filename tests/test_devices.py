"""Which card the numbers on screen are about.

`_cuda_like` already records this failure for the DEFAULT path, in its own
words: torch's current device is not always device 0, and reading device 0's
properties "then reported card 0's name and VRAM while loading onto a
different card. Every number on screen would describe hardware the model was
not running on, which is worse than no number at all."

The explicit path reintroduced it. `detect("cuda:1")` probed whichever card
was current, kept that card's name and VRAM, and overwrote the device STRING
alone — so a two-GPU box asking for the big card was told it had the small
one. `vram_gb` is what `capacity.guard` refuses against and what the fit
calculator plans with, so the wrong card's memory is not cosmetic.

MEASURED on a simulated pair (RTX 4060 8.6 GB at index 0, A100 80 GB at
index 1): `cuda:1` reported "NVIDIA RTX 4060 Laptop GPU, 8.6 GB" before and
"NVIDIA A100-SXM4-80GB, 80.0 GB" after.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from modelmri import devices  # noqa: E402

CARDS = {0: ("NVIDIA RTX 4060 Laptop GPU", 8.6e9), 1: ("NVIDIA A100-SXM4-80GB", 80e9)}


@pytest.fixture
def two_gpus(monkeypatch):
    """A machine with two different cards, current device 0."""

    class Props:
        def __init__(self, index):
            self.name, self.total_memory = CARDS[int(index)]

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(torch.cuda, "get_device_properties", Props)
    monkeypatch.setattr(torch.cuda, "is_bf16_supported", lambda: True)
    return CARDS


def test_an_explicit_index_reports_that_cards_name_and_memory(two_gpus):
    found = devices.detect("cuda:1")

    assert found.torch_device == "cuda:1"
    assert found.name == "NVIDIA A100-SXM4-80GB"
    assert found.vram_gb == pytest.approx(80.0)


def test_the_default_still_reads_the_current_device(two_gpus):
    found = devices.detect("cuda")

    assert found.torch_device == "cuda:0"
    assert found.name == "NVIDIA RTX 4060 Laptop GPU"
    assert found.vram_gb == pytest.approx(8.6)


def test_an_index_that_does_not_exist_is_not_answered_with_card_zero(two_gpus):
    """It used to return card 0's identity under the device string `cuda:7`,
    so a typo produced a confident description of hardware that is not there
    and a load onto a device torch would then reject."""
    found = devices.detect("cuda:7")

    assert found.kind == "cpu"
    assert "index 7" in found.reason
    assert found.name != "NVIDIA RTX 4060 Laptop GPU"


def test_the_reason_names_the_device_actually_chosen(two_gpus):
    """ "requested explicitly (cuda:1)" has to be about the card that was
    read, not about the string that was asked for."""
    assert "cuda:1" in devices.detect("cuda:1").reason
