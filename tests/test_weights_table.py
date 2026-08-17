"""A per-tensor table, and the one confusion it must never permit.

The table half is arithmetic and the health half is a count, and both of them
have the same failure mode: a number that arrives looking like a measurement
when nothing measured it. A NaN count of 0 for a tensor nobody opened, a
parameter total over the 512 rows that fit rather than the 30,000 there were, a
mean of 0.0 for a tensor that is entirely NaN, a tied weight counted twice.

So most of these tests are about that boundary, and each one names the wrong
conclusion it exists to prevent.

The models are real `nn.Module`s built here, with real NaN and real infinities
put into them on purpose. The pricing tests need no torch at all — that half is
pure arithmetic, which is what lets a caller see the cost of a scan on a
machine with no accelerator.
"""

from __future__ import annotations

import json
import math
import struct

import pytest

from modelmri import weights_table as wt
from modelmri.errors import BadRequest

torch = pytest.importorskip("torch")


# ------------------------------------------------------------ real modules


class Tiny(torch.nn.Module):
    """Small, real, and tied — the three properties every test here needs.

    `head.weight` is assigned the embedding's own Parameter, which is what
    weight tying is in every transformers model, so the shared-storage path is
    exercised by the same construction the real case uses.
    """

    def __init__(self, *, tie: bool = True):
        super().__init__()
        self.embed = torch.nn.Embedding(16, 8)
        self.block = torch.nn.Linear(8, 8)
        self.head = torch.nn.Linear(8, 16, bias=False)
        if tie:
            self.head.weight = self.embed.weight
        self.register_buffer("scale", torch.ones(8))


class Wide(torch.nn.Module):
    """Several large clean tensors and one tiny broken one.

    The tiny one is the point: it is the smallest tensor in the module, so a
    table capped by size would drop it, and a corrupted bias vector is exactly
    the small tensor somebody is looking for.
    """

    def __init__(self, *, n: int = 6, width: int = 32):
        super().__init__()
        self.layers = torch.nn.ModuleList(
            torch.nn.Linear(width, width, bias=False) for _ in range(n)
        )
        self.tiny = torch.nn.Parameter(torch.zeros(4))
        with torch.no_grad():
            self.tiny[2] = float("nan")


def _with_nan() -> Tiny:
    model = Tiny()
    with torch.no_grad():
        model.block.weight[0, 0] = float("nan")
    return model


def _with_infinities() -> Tiny:
    model = Tiny()
    with torch.no_grad():
        model.block.weight[0, 0] = float("inf")
        model.block.weight[1, 0] = float("-inf")
        model.block.weight[1, 1] = float("-inf")
    return model


def _row(table, name):
    return next(r for r in table.rows if r.name == name)


# --------------------------------------------------------------- the table


def test_a_tied_weight_is_listed_twice_and_counted_once():
    """Two wrong conclusions, one on each side. torch's own default hides the
    second name, so a reader is never told `head.weight` IS the embedding;
    listing both and summing both says the model has twice the parameters it
    has."""
    table = wt.table(Tiny())
    names = [r.name for r in table.rows]
    assert "embed.weight" in names and "head.weight" in names
    alias = _row(table, "head.weight")
    assert alias.shared_with == "embed.weight"
    assert table.shared_tensors == 1
    # 16x8 embedding + 8x8 block weight + 8 block bias + 8 scale buffer.
    assert table.elements_total == 128 + 64 + 8 + 8
    assert "SAME STORAGE" in alias.means()


def test_an_untied_model_reports_no_sharing_rather_than_defaulting_to_it():
    """The detector must be reading identity, not assuming a name pattern."""
    table = wt.table(Tiny(tie=False))
    assert table.shared_tensors == 0
    assert all(not r.shared_with for r in table.rows)
    assert table.elements_total == 128 + 64 + 8 + 128 + 8


def test_the_totals_are_over_every_tensor_even_when_the_rows_are_capped():
    """A capped table read as the whole model is the failure this cap could
    cause. The rows are cut; the arithmetic is not."""
    full = wt.table(Wide())
    capped = wt.table(Wide(), limit=2)
    assert capped.tensors_shown == 2
    assert capped.tensors_total == full.tensors_total
    assert capped.elements_total == full.elements_total
    assert capped.bytes_total == full.bytes_total


def test_a_capped_table_says_what_it_dropped_and_the_pieces_add_up():
    """A silent cap reads as "this is every tensor"."""
    table = wt.table(Wide(), limit=2)
    assert table.tensors_dropped == table.tensors_total - table.tensors_shown
    shown = sum(r.elements for r in table.rows if not r.shared_with)
    assert shown + table.dropped_elements == table.elements_total
    assert "OF" in table.means() and "not shown" in table.means()


def test_buffers_are_listed_beside_parameters_rather_than_omitted():
    """named_parameters() alone misses every buffer, and a NaN in a rotary
    cache kills a forward pass exactly as dead as one in a weight."""
    table = wt.table(Tiny())
    assert table.buffers == 1
    assert _row(table, "scale").kind == "buffer"
    assert wt.table(Tiny(), include_buffers=False).buffers == 0


def test_a_buffer_is_not_reported_as_a_frozen_parameter():
    """False would put every buffer in the frozen-weights column. The question
    does not apply to a buffer, and None is the answer that says so."""
    assert _row(wt.table(Tiny()), "scale").trainable is None


def test_frozen_and_trainable_parameters_are_separate_numbers():
    model = Tiny(tie=False)
    model.block.requires_grad_(False)
    table = wt.table(model)
    assert table.frozen_elements == 64 + 8
    assert table.trainable_elements == 128 + 128
    # The `scale` buffer is in NEITHER, and that is correct: a buffer is not a
    # parameter with `requires_grad=False`, it is not a parameter at all. This
    # assertion used to be `frozen + trainable == elements_total` and failed by
    # exactly the buffer's 8 elements — the arithmetic was right and the
    # invariant was wrong, so `buffer_elements` now makes the third category
    # visible instead of leaving a silent gap.
    assert table.buffer_elements == 8
    assert (
        table.frozen_elements + table.trainable_elements + table.buffer_elements
        == table.elements_total
    )


def test_the_three_categories_account_for_every_element():
    """Trainable, frozen, buffer — and nothing outside them.

    A reader summing two of the three gets a number short of the total with
    nothing saying why, and a BatchNorm-heavy model hides a real share of its
    weights in that gap."""
    for model in (Tiny(), Tiny(tie=False), Wide()):
        table = wt.table(model)
        assert (
            table.trainable_elements + table.frozen_elements + table.buffer_elements
            == table.elements_total
        ), type(model).__name__


def test_bytes_are_read_from_the_tensor_rather_than_a_dtype_width_table():
    """A width table is a thing that can be wrong about a dtype it has not
    seen. bfloat16 is two bytes because torch says the element size is two."""
    table = wt.table(Tiny().to(torch.bfloat16))
    row = _row(table, "block.weight")
    assert row.dtype == "bfloat16"
    assert row.bytes == 64 * 2
    assert table.bytes_total == table.elements_total * 2


def test_the_per_dtype_breakdown_covers_the_same_tensors_as_the_total():
    """A breakdown that does not sum to the total is describing a different
    set of tensors from the one the headline number came from."""
    table = wt.table(Tiny())
    assert sum(v["elements"] for v in table.by_dtype.values()) == table.elements_total
    assert sum(v["bytes"] for v in table.by_dtype.values()) == table.bytes_total
    assert sum(v["tensors"] for v in table.by_dtype.values()) == table.tensors_unique


def test_the_module_path_is_split_structurally_not_matched_by_name():
    """`imaging.py` was written after `vla.py` had to be corrected for reading
    meaning out of tensor-name prefixes."""
    assert wt.split_name("model.layers.0.self_attn.q_proj.weight") == (
        "model.layers.0.self_attn.q_proj",
        "weight",
    )
    assert wt.split_name("weight") == ("", "weight")


def test_the_owning_module_class_is_read_from_the_model():
    """Netron's op-type column, and it comes from named_modules() rather than
    from guessing what `q_proj` usually is."""
    assert _row(wt.table(Tiny()), "block.weight").module_type == "Linear"
    assert _row(wt.table(Tiny()), "embed.weight").module_type == "Embedding"


def test_a_boolean_limit_is_refused_rather_than_quietly_meaning_one_row():
    """isinstance(True, int) is True, so `limit=True` is a one-row table and
    nothing would have said so."""
    with pytest.raises(BadRequest, match="quietly mean"):
        wt.table(Tiny(), limit=True)


def test_a_zero_limit_is_refused_rather_than_returning_an_empty_table():
    with pytest.raises(BadRequest, match="at least 1"):
        wt.table(Tiny(), limit=0)


def test_a_module_holding_no_tensors_is_refused_by_name():
    """An empty table reads as "this model has no weights"."""
    with pytest.raises(wt.NotMeasured, match="no parameters and no buffers"):
        wt.table(torch.nn.Identity())


def test_something_that_is_not_a_module_is_told_what_to_use_instead():
    with pytest.raises(BadRequest, match="table_from_safetensors"):
        wt.table("Qwen/Qwen3-1.7B")


# -------------------------------------------------- zero versus not checked


def test_a_table_without_a_health_scan_says_nothing_was_counted():
    """The headline confusion. A shapes-and-sizes table that is silent about
    health reads as a health report with nothing wrong in it."""
    table = wt.table(Tiny())
    assert table.health_checked is False
    assert all(r.health is None for r in table.rows)
    assert all(r.health_reason for r in table.rows)
    assert "NO HEALTH SCAN WAS RUN" in table.means()
    assert "not zero" in table.means()


def test_an_unscanned_row_never_presents_itself_as_a_nan_count_of_zero():
    row = _row(wt.table(Tiny()), "block.weight")
    assert row.to_dict()["health"] is None
    assert "NOT READ" in row.means()
    assert "not a NaN count of zero" in row.means()


def test_a_scanned_clean_tensor_and_an_unscanned_one_are_distinguishable():
    """The two must not produce the same shape of answer, or the distinction
    is a docstring rather than a guarantee."""
    unscanned = _row(wt.table(Tiny()), "block.weight")
    scanned = _row(wt.table(Tiny(), health=True, exhaustive=True), "block.weight")
    assert unscanned.health is None and unscanned.health_reason
    assert scanned.health is not None and scanned.health_reason == ""
    assert scanned.health.nan == 0
    assert scanned.health.all_finite is True


# ------------------------------------------------------------ real findings


def test_an_injected_nan_is_found_and_the_tensor_is_named():
    table = wt.table(_with_nan(), health=True, exhaustive=True)
    assert table.nan_total == 1
    assert table.unhealthy == 1
    assert "block.weight" in table.unhealthy_names
    assert _row(table, "block.weight").health.all_finite is False
    assert "NON-FINITE" in table.means()


def test_the_two_infinities_are_counted_apart():
    """ "It blew up upward" and "it blew up downward" are different failures,
    and one signed count cannot tell them apart."""
    table = wt.table(_with_infinities(), health=True, exhaustive=True)
    health = _row(table, "block.weight").health
    assert health.pos_inf == 1
    assert health.neg_inf == 2
    assert health.nan == 0
    assert health.nonfinite == 3


def test_a_clean_model_is_only_called_clean_when_every_element_was_read():
    table = wt.table(Tiny(), health=True, exhaustive=True)
    assert table.unhealthy == 0
    assert table.unproven == 0
    assert "No NaN and no infinity anywhere" in table.means()


def test_a_tensor_that_is_entirely_nan_has_no_mean_rather_than_a_mean_of_zero():
    """0.0 is a number some tensor really holds. A tensor with no finite part
    has no mean at all, and the two must not arrive as the same value."""
    health, reason = wt.scan_tensor(torch.full((4, 4), float("nan")))
    assert reason == ""
    assert health.finite == 0
    assert health.minimum is None
    assert health.maximum is None
    assert health.mean is None
    assert health.std is None
    assert "None rather than 0" in health.means()


def test_the_standard_deviation_of_one_element_is_none_not_zero():
    """0.0 there reads as "this tensor is constant", which is a real finding
    this must not manufacture out of a sample size of one."""
    health, _ = wt.scan_tensor(torch.tensor([3.5]))
    assert health.mean == pytest.approx(3.5)
    assert health.std is None
    health_two, _ = wt.scan_tensor(torch.tensor([3.5, 3.5]))
    assert health_two.std == 0.0, "two identical elements really do have zero spread"


def test_the_statistics_are_never_rounded_into_zero():
    """A bf16 weight of 1e-12 rounded to nine places is 0.0, and a module whose
    job is to notice numbers going wrong must not be the thing that zeroes
    one."""
    health, _ = wt.scan_tensor(torch.full((32,), 1e-12))
    assert health.mean != 0.0
    assert health.minimum != 0.0
    assert health.zeros == 0


def test_counts_account_for_every_element_that_was_read():
    """A count that does not add up means some elements fell into no bucket,
    and a NaN could be one of them."""
    model = _with_infinities()
    with torch.no_grad():
        model.block.weight[2, 2] = float("nan")
    health = wt.table(model, health=True, exhaustive=True)
    row = _row(health, "block.weight").health
    assert row.nan + row.pos_inf + row.neg_inf + row.finite == row.scanned
    assert row.scanned == row.elements == 64
    assert row.zeros <= row.finite


def test_an_all_zero_tensor_is_reported_as_a_finding_not_as_a_clean_scan():
    """A dead layer and a fresh bias look identical here; this counts them and
    refuses to decide which."""
    health, _ = wt.scan_tensor(torch.zeros(64))
    assert health.all_zero is True
    assert health.all_finite is True
    assert health.zeros == 64
    assert wt.scan_tensor(torch.ones(64))[0].all_zero is False


# ---------------------------------------------------- sampling versus proof


def test_a_sampled_scan_that_finds_nothing_says_unproven_rather_than_clean():
    """The whole reason `all_finite` is three-valued. "None in the tenth of it
    I read" is not "none"."""
    health, _ = wt.scan_tensor(torch.zeros(100_000), allowance=wt.MIN_ALLOWANCE)
    assert health.complete is False
    assert health.nan == 0
    assert health.all_finite is None, "a partial read must not prove cleanliness"
    assert "UNPROVEN" in health.means() or "not the same claim" in health.means()


def test_a_finding_survives_sampling_because_seeing_one_proves_one():
    """The asymmetry is real and the code has to encode it: absence of evidence
    is not evidence of absence, but evidence is evidence."""
    tensor = torch.full((100_000,), float("nan"))
    health, _ = wt.scan_tensor(tensor, allowance=wt.MIN_ALLOWANCE)
    assert health.complete is False
    assert health.nan > 0
    assert health.all_finite is False


def test_a_sampled_all_zero_tensor_is_unknown_rather_than_all_zero():
    health, _ = wt.scan_tensor(torch.zeros(100_000), allowance=wt.MIN_ALLOWANCE)
    assert health.zeros == health.scanned
    assert health.all_zero is None


def test_a_sampled_scan_states_its_stride_and_says_the_counts_are_the_sample():
    health, _ = wt.scan_tensor(torch.zeros(100_000), allowance=wt.MIN_ALLOWANCE)
    assert health.stride > 1
    assert health.scanned < health.elements
    assert "COUNT OVER" in health.means()


def test_a_run_that_sampled_anything_does_not_claim_the_model_is_clean():
    model = Wide(n=2, width=64)
    with torch.no_grad():
        model.tiny.zero_()
    table = wt.table(model, health=True, per_tensor_elements=wt.MIN_ALLOWANCE)
    assert table.unhealthy == 0
    assert table.unproven >= 1
    assert "UNPROVEN" in table.means()


def test_an_integer_tensor_is_finite_by_construction_not_by_counting():
    """int8 has no bit pattern for NaN. Saying "we looked and found none" for
    a dtype that cannot hold one is a weaker claim than the truth."""
    health, _ = wt.scan_tensor(
        torch.arange(100_000, dtype=torch.int32), allowance=wt.MIN_ALLOWANCE
    )
    assert health.nonfinite_impossible is True
    assert health.complete is False
    assert health.all_finite is True, "true by construction, sampling or not"
    assert "by construction" in health.means()


def test_a_boolean_tensor_counts_its_false_entries_as_zeros():
    health, _ = wt.scan_tensor(torch.tensor([True, False, True, False]))
    assert health.zeros == 2
    assert health.all_finite is True


# ------------------------------------------------- the tensors nobody reads


def test_a_meta_tensor_is_unscanned_with_a_reason_not_scanned_as_zeros():
    """A meta tensor is a shape and a dtype with no values behind it. Counting
    zero NaN in one would be counting nothing at all."""
    with torch.device("meta"):
        model = Tiny(tie=False)
    table = wt.table(model, health=True, source="meta model")
    row = _row(table, "block.weight")
    assert row.device == "meta"
    assert row.health is None
    assert "meta" in row.health_reason
    assert table.meta_tensors == table.tensors_total
    assert "no values at all" in table.means()


def test_a_meta_tensor_still_reports_the_bytes_it_would_take():
    """The size question is answerable without the values, and refusing it
    would make this useless for exactly the models it is needed for."""
    with torch.device("meta"):
        model = Tiny(tie=False)
    assert _row(wt.table(model), "block.weight").bytes == 64 * 4


def test_an_empty_tensor_is_not_reported_as_checked_and_clean():
    health, reason = wt.scan_tensor(torch.zeros(0))
    assert health is None
    assert "no elements" in reason


def test_a_complex_tensor_is_refused_rather_than_half_measured():
    health, reason = wt.scan_tensor(torch.zeros(4, dtype=torch.complex64))
    assert health is None
    assert "no ordering" in reason


def test_a_tied_alias_points_at_the_row_that_was_read_instead_of_being_reread():
    """Two identical health blocks read as two independent confirmations, when
    there is only one tensor and it was checked once."""
    table = wt.table(Tiny(), health=True, exhaustive=True)
    alias = _row(table, "head.weight")
    assert alias.health is None
    assert "same storage as `embed.weight`" in alias.health_reason
    assert _row(table, "embed.weight").health is not None
    assert table.tensors_unscanned == 1


def test_every_tensor_that_could_be_read_was_read():
    """A scan that covered the first few tensors and stopped would still be
    able to say "no NaN found"."""
    table = wt.table(Wide(), health=True)
    scannable = [r for r in table.rows if not r.shared_with and r.elements]
    assert scannable
    assert all(r.health is not None for r in scannable)


# ------------------------------------------------------- the broken and small


def test_a_broken_tensor_is_never_dropped_by_the_row_cap():
    """Dropping the small rows is exactly how this table would hide the thing
    it exists to find: a corrupted bias vector is the smallest tensor there
    is."""
    table = wt.table(Wide(), limit=2, health=True, exhaustive=True)
    names = [r.name for r in table.rows]
    assert "tiny" in names, "the only broken tensor was cut to fit the cap"
    assert table.unhealthy == 1
    assert table.tensors_shown <= 3


def test_the_unhealthy_count_is_complete_even_when_the_rows_are_not():
    """The rows are a view; the finding is a fact."""

    class ManyBroken(torch.nn.Module):
        def __init__(self):
            super().__init__()
            for i in range(6):
                self.register_parameter(
                    f"p{i}", torch.nn.Parameter(torch.full((4,), float("nan")))
                )

    table = wt.table(ManyBroken(), limit=2, health=True, exhaustive=True)
    assert table.unhealthy == 6
    assert table.tensors_shown == 2
    assert "did not fit" in " ".join(table.notes)


# ------------------------------------------------------- chunking arithmetic


def test_a_scan_spanning_several_chunks_agrees_with_one_that_does_not():
    """The variance is merged across chunk boundaries, and a merge that is
    wrong gives a standard deviation nobody can tell is wrong by looking."""
    torch.manual_seed(0)
    tensor = torch.randn(wt.CHUNK_ELEMENTS * 2 + 12_345) * 3.0 + 7.0
    # `allowance` raised past the tensor, because this test is about the CHUNK
    # merge and not about the sampling policy. Those are two different
    # mechanisms that both trigger on size: `CHUNK_ELEMENTS` is the working
    # window a scan materialises, `SAMPLE_ELEMENTS` is how much of a tensor it
    # will read at all — and they are the same number, so a tensor built to
    # span three chunks is also a tensor that gets strided.
    #
    # Without this the assertions below compared exact statistics against a
    # stride-3 sample and failed, which read as a broken merge and was a test
    # measuring the wrong thing.
    health, _ = wt.scan_tensor(tensor, allowance=len(tensor))
    assert health.complete
    assert health.stride == 1
    var, mean = torch.var_mean(tensor.double(), correction=0)
    assert health.mean == pytest.approx(float(mean), rel=1e-9)
    assert health.std == pytest.approx(math.sqrt(float(var)), rel=1e-9)
    assert health.minimum == pytest.approx(float(tensor.min()))
    assert health.maximum == pytest.approx(float(tensor.max()))


def test_the_finite_statistics_ignore_the_non_finite_elements():
    """A minimum of -inf is not a minimum, and a mean including a NaN is a
    NaN — which would arrive looking like a broken tool rather than a broken
    tensor."""
    tensor = torch.tensor([1.0, 2.0, 3.0, float("nan"), float("-inf"), float("inf")])
    health, _ = wt.scan_tensor(tensor)
    assert health.minimum == pytest.approx(1.0)
    assert health.maximum == pytest.approx(3.0)
    assert health.mean == pytest.approx(2.0)
    assert health.finite == 3


# ----------------------------------------------------------- pricing, no torch


def test_a_tensor_under_the_allowance_is_read_whole():
    assert wt.plan_scan(1000, 4096) == (1000, 1)


def test_the_stride_spreads_the_sample_over_the_whole_tensor():
    """Taking the first 1M elements of an embedding reads the first few hundred
    rows of the vocabulary, which is a question about those tokens."""
    scanned, stride = wt.plan_scan(311_164_928, 1 << 20)
    assert stride > 1
    assert scanned <= 1 << 20
    assert scanned * stride >= 311_164_928 - stride


def test_the_predicted_scan_size_is_what_a_strided_view_really_holds():
    """A predicted count that differs from the count actually read would make
    every "over N elements" sentence in this module a small lie."""
    for elements, allowance in ((1000, 4096), (100_000, 1024), (12_345, 1000)):
        scanned, stride = wt.plan_scan(elements, allowance)
        view = torch.zeros(elements)[::stride]
        assert int(view.numel()) == scanned


def test_a_budget_that_divides_below_the_floor_is_refused_with_the_arithmetic():
    """Scanning 40 elements of each of a million tensors describes nothing, and
    reporting it as a health scan would be worse than not running one."""
    with pytest.raises(wt.NotMeasured) as caught:
        wt.allowance_for(1_000_000, max_scan_elements=1 << 20)
    said = str(caught.value)
    assert "1,000,000 tensors" in said
    assert "Raise `max_scan_elements`" in said


def test_the_allowance_is_shared_evenly_so_coverage_does_not_tail_off():
    """Scanning in model order until the budget runs out reads the embedding
    and the first blocks and never opens the last forty layers, then reports
    "no NaN found"."""
    assert (
        wt.allowance_for(256, per_tensor_elements=1 << 20, max_scan_elements=1 << 20)
        == 4096
    )
    assert (
        wt.allowance_for(2, per_tensor_elements=4096, max_scan_elements=1 << 30) == 4096
    )


def test_the_cost_of_a_scan_is_knowable_before_an_element_is_read():
    """`scan_cost` takes element counts, which the table half produces without
    touching a weight."""
    cost = wt.scan_cost([311_164_928] + [4_194_304] * 100)
    assert cost["tensors"] == 101
    assert cost["scanned"] < cost["elements"]
    assert cost["sampled"] >= 1
    assert "would read" in cost["means"]
    assert "nothing else" in cost["means"]


def test_an_exhaustive_price_says_the_answer_would_cover_the_whole_model():
    cost = wt.scan_cost([1_000_000, 2_000_000], exhaustive=True)
    assert cost["scanned"] == cost["elements"] == 3_000_000
    assert cost["sampled"] == 0
    assert "whole model" in cost["means"]


def test_a_boolean_budget_is_refused_like_every_other_integer_knob():
    with pytest.raises(BadRequest, match="quietly mean"):
        wt.allowance_for(4, max_scan_elements=True)


def test_pricing_needs_no_torch_and_no_model():
    """The half that decides whether to spend the other half must run on a
    machine that cannot afford the other half."""
    import subprocess
    import sys

    done = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; from modelmri import weights_table as w; "
            "print(w.scan_cost([10, 20, 30])['scanned']); "
            "print('torch' in sys.modules)",
        ],
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert done.returncode == 0, done.stderr
    assert done.stdout.split() == ["60", "False"]


# ------------------------------------------------------- the header, no load


def _write_safetensors(path, tensors):
    """A real safetensors file, written by the reference implementation.

    Written with `safetensors.torch.save_file` rather than by hand: a file this
    test constructed to match this reader would prove the reader agrees with
    itself.
    """
    from safetensors.torch import save_file

    save_file(tensors, str(path))
    return path


def test_a_header_gives_exact_shapes_and_carries_no_health_at_all(tmp_path):
    """The case the whole zero-versus-unchecked rule was written for. The file
    below really does hold 64 NaN, and a table read from its header must not
    imply anything either way."""
    file = _write_safetensors(
        tmp_path / "model.safetensors",
        {
            "embed.weight": torch.zeros(16, 8),
            "block.weight": torch.full((8, 8), float("nan")),
        },
    )
    table = wt.table_from_safetensors(file)
    assert table.elements_total == 128 + 64
    assert _row(table, "embed.weight").shape == [16, 8]
    assert _row(table, "embed.weight").bytes == 128 * 4
    assert table.health_checked is False
    assert all(r.health is None for r in table.rows)
    assert table.nan_total == 0, "no scan ran, so the total is a count of nothing"
    assert "NO HEALTH SCAN WAS RUN" in table.means()


def test_a_header_table_says_the_numbers_were_never_read(tmp_path):
    """The file above really does contain 64 NaN. Nothing here may imply it
    does not."""
    file = _write_safetensors(
        tmp_path / "model.safetensors",
        {"block.weight": torch.full((8, 8), float("nan"))},
    )
    row = _row(wt.table_from_safetensors(file), "block.weight")
    assert "never opened past its header" in row.health_reason
    assert "NOT READ" in row.means()


def test_the_two_dtype_vocabularies_are_not_translated_into_each_other(tmp_path):
    """A translation table is a thing that can be wrong. Saying which
    vocabulary a table is written in costs nothing and cannot be."""
    file = _write_safetensors(
        tmp_path / "m.safetensors", {"w": torch.zeros(4, dtype=torch.bfloat16)}
    )
    header = wt.table_from_safetensors(file)
    live = wt.table(Tiny().to(torch.bfloat16))
    assert header.dtype_naming == "safetensors"
    assert live.dtype_naming == "torch"
    assert set(header.by_dtype) == {"BF16"}
    assert set(live.by_dtype) == {"bfloat16"}


def test_the_header_byte_span_is_the_payload_not_the_file_size(tmp_path):
    """`stat().st_size` also counts the header and its alignment padding, and
    quoting that as the weight size overstates a small checkpoint badly."""
    file = _write_safetensors(tmp_path / "m.safetensors", {"w": torch.zeros(10)})
    table = wt.table_from_safetensors(file)
    assert table.bytes_total == 40
    assert file.stat().st_size > 40


def test_a_directory_of_shards_is_read_as_one_table(tmp_path):
    _write_safetensors(tmp_path / "a.safetensors", {"one": torch.zeros(4)})
    _write_safetensors(tmp_path / "b.safetensors", {"two": torch.zeros(6)})
    table = wt.table_from_safetensors(tmp_path)
    assert table.tensors_total == 2
    assert table.elements_total == 10
    assert "2 shards" in " ".join(table.notes)


def test_a_tensor_declared_by_two_shards_is_reported_rather_than_overwritten(
    tmp_path,
):
    """Keeping the last one silently makes the table about a file nobody has."""
    _write_safetensors(tmp_path / "a.safetensors", {"w": torch.zeros(4)})
    _write_safetensors(tmp_path / "b.safetensors", {"w": torch.zeros(4)})
    table = wt.table_from_safetensors(tmp_path)
    assert table.tensors_total == 2
    assert "declared in both" in " ".join(table.notes)


def test_a_directory_with_no_safetensors_is_refused_by_name(tmp_path):
    (tmp_path / "pytorch_model.bin").write_bytes(b"not a safetensors file")
    with pytest.raises(wt.NotMeasured, match="no .safetensors file"):
        wt.table_from_safetensors(tmp_path)


def test_a_path_that_does_not_exist_is_refused_rather_than_returning_nothing(
    tmp_path,
):
    with pytest.raises(wt.NotMeasured, match="nothing at"):
        wt.table_from_safetensors(tmp_path / "absent.safetensors")


def test_a_truncated_header_is_refused_rather_than_read_as_an_empty_model(
    tmp_path,
):
    """An empty table is indistinguishable from a model with no weights."""
    broken = tmp_path / "broken.safetensors"
    broken.write_bytes(struct.pack("<Q", 4) + b"{")
    with pytest.raises(wt.NotMeasured):
        wt.table_from_safetensors(broken)


def test_a_header_missing_a_tensors_offsets_is_refused_not_listed_partly(
    tmp_path,
):
    """Listing it with the fields it did have would put a tensor of unknown
    size into a total that claims to be exact."""
    header = json.dumps({"w": {"dtype": "F32", "shape": [4]}}).encode("utf-8")
    path = tmp_path / "m.safetensors"
    path.write_bytes(struct.pack("<Q", len(header)) + header + b"\0" * 16)
    with pytest.raises(wt.NotMeasured, match="readable dtype, shape and offset"):
        wt.table_from_safetensors(path)
