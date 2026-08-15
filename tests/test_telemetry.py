"""Telemetry must not invent the numbers it cannot take.

Three traps. A memory column showing 0 is a claim that nothing was used, and
CPU has no allocator to ask. A context percentage computed from
`model_max_length` is arithmetic on a sentinel for a large share of tokenizers
— several report 1000000000000000019884624838656. And averaging prompt
processing into the decode rate makes a fast model look slow whenever the
prompt is long, which is exactly when somebody is looking at the number.

The fourth is the one the feature exists for: the introspection cost is real,
quadratic, and larger than the weights at long contexts, and it has to be
named before the run rather than explained after the allocation fails.
"""

from __future__ import annotations

import pytest

from modelmri import telemetry

# ----------------------------------------------- what watching the model costs


def test_the_readme_figure_reproduces():
    """12 layers x 12 heads x 4096^2 x 2 bytes = 4.8 GB — larger than the
    weights of most models this runs on."""
    got = telemetry.eager_attention_bytes(12, 12, 4096, 2)
    assert got / 1e9 == pytest.approx(4.83, abs=0.05)


def test_it_is_quadratic_in_sequence_length():
    a = telemetry.eager_attention_bytes(12, 12, 1024, 2)
    b = telemetry.eager_attention_bytes(12, 12, 2048, 2)
    assert b == a * 4


def test_the_warning_fires_before_a_run_that_will_not_fit():
    said = telemetry.warn_before(12, 12, 4096, 2, free_bytes=6_000_000_000)
    assert "4.8 GB" in said
    assert "6.0 GB free" in said
    assert "shorter prompt" in said


def test_the_warning_is_silent_when_there_is_room():
    assert telemetry.warn_before(12, 12, 128, 2, free_bytes=8_000_000_000) == ""


def test_the_warning_is_silent_when_free_memory_is_unknown():
    """A guard that fires because it could not measure blocks CPU users out of
    ignorance."""
    assert telemetry.warn_before(12, 12, 100_000, 2, free_bytes=None) == ""


def test_the_warning_shows_its_arithmetic():
    said = telemetry.warn_before(28, 16, 2048, 2, free_bytes=4_000_000_000)
    assert "28 x 16 x 2048^2 x 2 bytes" in said


# ------------------------------------------------------------ context limits


class Cfg:
    def __init__(self, **kw):
        for k, v in kw.items():
            setattr(self, k, v)


class Model:
    def __init__(self, cfg):
        self.config = cfg


class Tok:
    def __init__(self, model_max_length=None):
        if model_max_length is not None:
            self.model_max_length = model_max_length


def test_the_config_is_preferred_over_the_tokenizer():
    limit, source = telemetry.context_limit(
        Model(Cfg(max_position_embeddings=2048)), Tok(model_max_length=512)
    )
    assert limit == 2048 and "config" in source


def test_a_sentinel_context_is_rejected_by_size_not_by_magic_constant():
    """int(1e30) round-tripped is what several tokenizers actually report. A
    "0.0% of context used" derived from it is arithmetic on a placeholder."""
    limit, why = telemetry.context_limit(
        Model(Cfg()), Tok(model_max_length=1000000000000000019884624838656)
    )
    assert limit is None
    assert "sentinel" in why


def test_gpt2_style_n_positions_is_found():
    limit, source = telemetry.context_limit(Model(Cfg(n_positions=1024)), Tok())
    assert limit == 1024 and "n_positions" in source


def test_the_tokenizer_is_used_when_it_is_the_only_sane_source():
    limit, source = telemetry.context_limit(Model(Cfg()), Tok(model_max_length=8192))
    assert limit == 8192 and "tokenizer" in source


def test_no_usable_limit_returns_none_with_the_reason():
    limit, why = telemetry.context_limit(Model(Cfg()), Tok())
    assert limit is None and "no usable context length" in why


def test_a_zero_or_negative_limit_is_not_accepted():
    assert (
        telemetry.context_limit(Model(Cfg(max_position_embeddings=0)), Tok())[0] is None
    )


# ------------------------------------------------------------ the run report


def test_prompt_time_is_kept_apart_from_the_decode_rate():
    """Averaging them makes a fast model look slow whenever the prompt is
    long, which is exactly when somebody reads the number."""
    with telemetry.Run("cpu") as run:
        import time

        time.sleep(0.05)  # "prompt processing"
        run.token()
        for _ in range(4):
            time.sleep(0.005)
            run.token()

    t = run.finish(prompt_tokens=100, generated_tokens=5, context=(2048, "config"))
    assert t.prompt_ms is not None and t.prompt_ms >= 45
    assert t.decode_ms is not None
    assert t.generated_tokens == 5

    # The invariant is that the two spans are measured SEPARATELY and the rate
    # uses only the decode one — not that decode is the faster of the two.
    #
    # The first version asserted `decode_ms < prompt_ms`, reasoning from the
    # sleeps above (50 ms then 4x5 ms). That holds on an idle machine and not
    # on a loaded CI runner: macOS py3.11 measured prompt 50.6 ms against
    # decode 84.6 ms and failed every PR. A test whose truth depends on how
    # busy the host is tests the host.
    assert t.tokens_per_s == pytest.approx(4 / (t.decode_ms / 1000), rel=1e-6)
    # And that is strictly faster than the same tokens over the whole wall
    # clock, which is the thing separating the phases buys you.
    assert t.tokens_per_s > 4 / ((t.prompt_ms + t.decode_ms) / 1000)


def test_the_true_token_count_wins_over_the_stream_chunk_count():
    """Regression. A TextIteratorStreamer yields one chunk per token PLUS a
    final flush from TextStreamer.end(), so counting chunks is always one too
    many. Measured on real gpt2: 8 new tokens reported as 9, and 40 as 41."""
    with telemetry.Run("cpu") as run:
        for _ in range(9):  # 8 tokens + the flush chunk
            run.token()
    t = run.finish(prompt_tokens=5, generated_tokens=8)
    assert t.generated_tokens == 8
    assert t.context_used == 13  # not 14


def test_the_rate_divides_by_intervals_not_tokens():
    """`first` is stamped when the FIRST token arrives, so the decode window
    spans n-1 gaps. Dividing by n inflated the rate by n/(n-1), which is
    unbounded as n approaches 1 — a synthetic true 20 tok/s read 36,630 at
    n=1."""
    import time

    with telemetry.Run("cpu") as run:
        run.token()
        for _ in range(4):
            time.sleep(0.01)
            run.token()
    t = run.finish(prompt_tokens=1, generated_tokens=5)
    # 4 intervals over the decode window, not 5.
    assert t.tokens_per_s == pytest.approx(4 / (t.decode_ms / 1000), rel=1e-6)


def test_a_single_token_has_no_rate_rather_than_an_enormous_one():
    with telemetry.Run("cpu") as run:
        run.token()
    t = run.finish(prompt_tokens=5, generated_tokens=1)
    assert t.tokens_per_s is None
    assert any("no interval" in n for n in t.notes)


def test_a_fallback_count_is_labelled_approximate():
    """When the worker delivered no ids there is no true count, so the stream
    count is used AND said to be a stream count."""
    with telemetry.Run("cpu") as run:
        run.token()
        run.token()
    t = run.finish(prompt_tokens=1)
    assert any("approximate" in n for n in t.notes)


def test_a_true_count_is_not_labelled_approximate():
    with telemetry.Run("cpu") as run:
        run.token()
        run.token()
    t = run.finish(prompt_tokens=1, generated_tokens=2)
    assert not any("approximate" in n for n in t.notes)


def test_a_run_that_streamed_nothing_has_no_rate():
    with telemetry.Run("cpu") as run:
        pass
    t = run.finish(prompt_tokens=10)
    assert t.tokens_per_s is None
    assert any("no decode rate" in n for n in t.notes)


def test_cpu_reports_memory_as_unmeasured_never_zero():
    with telemetry.Run("cpu") as run:
        run.token()
    t = run.finish(prompt_tokens=1)
    assert t.peak_bytes is None
    assert t.notes  # and it says why
    assert "allocated by PyTorch" in t.memory_note


def test_context_fullness_is_the_whole_sequence():
    with telemetry.Run("cpu") as run:
        for _ in range(3):
            run.token()
    t = run.finish(prompt_tokens=97, context=(200, "config"))
    assert t.context_used == 100
    assert t.context_fraction == pytest.approx(0.5)


def test_no_context_limit_means_no_fraction_not_zero():
    with telemetry.Run("cpu") as run:
        run.token()
    t = run.finish(prompt_tokens=5, context=(None, "no usable context length"))
    assert t.context_fraction is None
    assert any("context length" in n for n in t.notes)


def test_the_introspection_line_is_computed_from_the_shape():
    with telemetry.Run("cpu") as run:
        for _ in range(4):
            run.token()
    t = run.finish(prompt_tokens=96, n_layers=12, n_heads=12, dtype_bytes=2)
    assert t.introspection_bytes == telemetry.eager_attention_bytes(12, 12, 100, 2)
    assert "never allocates" in t.introspection_note


def test_no_shape_means_no_introspection_figure():
    with telemetry.Run("cpu") as run:
        run.token()
    t = run.finish(prompt_tokens=1)
    assert t.introspection_bytes is None


def test_the_report_says_the_memory_is_the_allocators_view():
    with telemetry.Run("cpu") as run:
        run.token()
    d = run.finish(prompt_tokens=1).to_dict()
    assert "not what the card reports" in d["means"]
    assert "one sample" in d["means"]


def test_the_report_is_json_safe():
    import json

    with telemetry.Run("cpu") as run:
        run.token()
    json.dumps(run.finish(prompt_tokens=1, n_layers=2, n_heads=2).to_dict())


def test_first_token_is_idempotent():
    """The streaming loop calls it on every token; only the first counts."""
    with telemetry.Run("cpu") as run:
        run.first_token()
        first = run.first
        run.first_token()
        assert run.first == first
