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


# ---------------------------------------------------------------------------
# Offering the CHOICE.
#
# `detect()` answers "where should this go", which is the right question right
# up until somebody wants to answer it themselves. A picker needs every device
# that exists, each with its own name, memory and free space — and a machine
# with two cards has two of each, so collapsing them into one "cuda" row hides
# the entire reason the list was opened.
# ---------------------------------------------------------------------------


def test_every_card_is_listed_separately_with_its_own_memory(two_gpus, monkeypatch):
    monkeypatch.setattr(
        torch.cuda, "mem_get_info", lambda i: (int(1e9 * (i + 1)), int(CARDS[i][1]))
    )

    rows = devices.available()

    cuda = [r for r in rows if r["kind"] == "cuda"]
    assert [r["id"] for r in cuda] == ["cuda:0", "cuda:1"]
    # Each row carries ITS OWN card, not whichever was current.
    assert cuda[0]["name"] == CARDS[0][0]
    assert cuda[1]["name"] == CARDS[1][0]
    assert cuda[0]["free_bytes"] == int(1e9)
    assert cuda[1]["free_bytes"] == int(2e9)


def test_the_cpu_is_always_offered(two_gpus, monkeypatch):
    """A list with no CPU row cannot express "run this on the CPU deliberately",
    which is a real thing people do to compare against a GPU result."""
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda i: (1, int(CARDS[i][1])))

    rows = devices.available()

    assert [r["id"] for r in rows if r["kind"] == "cpu"] == ["cpu"]


def test_exactly_one_row_is_the_default(two_gpus, monkeypatch):
    """A list where nothing is the default cannot explain what happens when you
    choose nothing."""
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda i: (1, int(CARDS[i][1])))

    rows = devices.available()

    assert sum(1 for r in rows if r["is_default"]) == 1
    assert next(r["id"] for r in rows if r["is_default"]) == "cuda:0"


def test_free_memory_that_cannot_be_read_is_none_not_zero(two_gpus, monkeypatch):
    """0 free says the machine is out of memory. Nobody asked it."""

    def refuses(index):
        raise RuntimeError("driver does not implement mem_get_info")

    monkeypatch.setattr(torch.cuda, "mem_get_info", refuses)

    rows = devices.available()

    cuda = [r for r in rows if r["kind"] == "cuda"]
    assert all(r["free_bytes"] is None for r in cuda)
    # Total is still known — it came off the properties read, not off this.
    assert all(r["total_bytes"] for r in cuda)


def test_the_cpu_row_never_claims_a_free_figure_it_did_not_measure():
    """ "Available" RAM on an OS with a page cache is a judgement, not a
    reading, and this project does not take the dependency that would make one.
    Total is honest; free stays unknown."""
    rows = devices.available()

    cpu = next(r for r in rows if r["kind"] == "cpu")
    assert cpu["free_bytes"] is None


def test_a_list_is_returned_even_with_no_torch_at_all(monkeypatch):
    """An empty picker reads as "this machine has no devices", which is never
    true."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    rows = devices.available()

    assert rows, "the list must never come back empty"
    assert any(r["is_default"] for r in rows)


def test_a_named_device_does_not_outlive_the_load_that_named_it(two_gpus, monkeypatch):
    """MEASURED, not hypothesised: load with device="cpu", then load again
    naming nothing, and the second one also went to the CPU on a machine with
    a working GPU. `self.accel` was mutated by the explicit load and nothing
    reset it, so "let the tool choose" quietly meant "keep the last choice" —
    and every load for the rest of the session was slower with no way to tell
    why. Empty means detect."""
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda i: (1, int(CARDS[i][1])))

    forced = devices.detect(prefer="cpu")
    assert forced.kind == "cpu"

    # The call the runtime makes when nothing was named. It must re-detect
    # rather than return whatever the previous call settled on.
    after = devices.detect(prefer="" or "auto")

    assert after.torch_device == "cuda:0"
    assert after.kind == "cuda"
