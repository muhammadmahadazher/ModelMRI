# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Muhammad Mahad Azher

"""The lens has to report its own error, or it is a confident ranked list.

The plain logit lens fails silently on some model families: it returns
plausible tokens computed through a transform the unembedding never saw, and
nothing on screen says so. That is the documented failure the tuned-lens paper
exists to address, and until now this package shipped the failure mode with no
reliability number beside it.

Two things get pinned here. The last row is the model itself, so its KL is an
arithmetic floor and must never be quoted as lens accuracy. And a lens that is
too far from the model must be labelled unusable in words, not left for the
reader to infer from a number they have no scale for.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from modelmri import lens  # noqa: E402

# --------------------------------------------------------- the verdict


def test_a_close_lens_is_usable_and_says_what_the_number_means():
    out = lens._reliability([0.4, 0.2, 0.1], floor=0.0)
    assert out["measured"] is True
    assert out["usable"] is True
    assert out["best_kl"] == 0.1
    assert "KL from the model's real next-token distribution" in out["means"]


def test_a_lens_that_never_gets_close_is_called_unusable_in_words():
    """Not left as a number the reader has no scale for."""
    out = lens._reliability([9.0, 8.4, 7.7], floor=0.0)
    assert out["usable"] is False
    assert "does not describe what this model predicts" in out["means"]
    assert "7.7" in out["means"]


def test_the_threshold_is_stated_not_hidden():
    out = lens._reliability([0.5], floor=0.0)
    assert out["threshold"] == lens.LENS_UNUSABLE_NATS


def test_the_best_layer_decides_usability_not_the_median():
    """One layer that tracks the model makes the lens useful, even if most
    layers are far — the panel's job is to say which layer to read."""
    out = lens._reliability([9.0, 9.0, 0.05], floor=0.0)
    assert out["usable"] is True
    assert out["best_kl"] == 0.05


def test_a_single_layer_model_says_it_could_not_measure():
    out = lens._reliability([], floor=0.0)
    assert out["measured"] is False
    assert "no intermediate rows" in out["why"]
    assert "usable" not in out


# ------------------------------------------------------ against a real model


@pytest.fixture(scope="module")
def tiny():
    transformers = pytest.importorskip("transformers")
    cfg = transformers.GPT2Config(
        n_layer=3, n_head=2, n_embd=32, vocab_size=64, n_positions=32
    )
    torch.manual_seed(0)
    model = transformers.AutoModelForCausalLM.from_config(cfg).eval()
    tok = lambda ids: "".join(f"<{i}>" for i in ids)  # noqa: E731

    class Tok:
        def decode(self, ids):
            return tok(ids)

    return model, Tok()


def test_every_row_carries_its_own_error(tiny):
    model, tok = tiny
    out = lens.logit_lens(model, tok, torch.tensor([[1, 2, 3]]), top_k=3)
    for row in out["layers"]:
        assert row["kl_to_final"] is not None
        assert row["kl_to_final"] >= 0.0


def test_the_final_row_is_the_arithmetic_floor(tiny):
    """The last hidden state IS the model's output, so its KL is ~0 — and
    quoting that as the lens's accuracy would advertise a precision the lens
    does not have at any other layer."""
    model, tok = tiny
    out = lens.logit_lens(model, tok, torch.tensor([[1, 2, 3]]), top_k=3)
    assert out["layers"][-1]["kl_to_final"] == pytest.approx(0.0, abs=1e-4)
    assert out["reliability"]["floor_kl"] == out["layers"][-1]["kl_to_final"]
    # The floor is excluded from the lens's own score.
    assert out["reliability"]["best_kl"] >= 0.0
    middle = [r["kl_to_final"] for r in out["layers"][:-1]]
    assert out["reliability"]["best_kl"] == min(middle)


def test_the_intermediate_rows_are_not_free_of_error(tiny):
    """A lens whose every row matched the model exactly would mean the
    measurement is not reading intermediate states at all."""
    model, tok = tiny
    out = lens.logit_lens(model, tok, torch.tensor([[1, 2, 3]]), top_k=3)
    middle = [r["kl_to_final"] for r in out["layers"][:-1]]
    assert max(middle) > 0.0


def test_reliability_travels_with_the_rows(tiny):
    model, tok = tiny
    out = lens.logit_lens(model, tok, torch.tensor([[1, 2]]), top_k=2)
    assert "reliability" in out
    assert out["reliability"]["measured"] is True
    assert isinstance(out["reliability"]["usable"], bool)


def _dtype_survives_on_this_cpu(name: str) -> bool:
    """Whether a forward pass in this dtype survives here — MEASURED, in a
    child process, because it cannot be measured any other way.

    THE CRASH THIS EXISTS FOR. `tests (windows-latest, py3.10)` failed CI with

        Windows fatal exception: code 0xc000001d
        [gw1] node down: Not properly terminated
        FAILED ...agrees_with_the_model_in_every_dtype[dtype1]
        worker 'gw1' crashed

    `0xc000001d` is STATUS_ILLEGAL_INSTRUCTION: a torch CPU kernel reaching
    for an instruction set that runner's processor does not have. `dtype1` is
    bfloat16. It is a fact about the machine rather than about this repo — the
    same commit passed on macOS twice and on windows py3.13.

    IT CANNOT BE A `try/except`. An illegal instruction is not a Python
    exception; the interpreter is gone before any handler runs, which is why
    the whole xdist worker died and took the run down with it. The only way to
    ask "can this CPU do bfloat16" and survive the answer is to ask a process
    that is allowed to die.

    NOT a capability lookup either. `torch.backends.cpu.get_cpu_capability()`
    reports AVX2 on the machine this was written on, where bfloat16 works
    perfectly — so that string does not answer this question, and inferring
    from it would be guessing at which kernels a given torch build routes
    where.

    Cached for the session: the probe costs one interpreter start plus a
    transformers import.
    """
    import subprocess
    import sys

    probe = (
        "import torch, transformers;"
        f"d = torch.{name};"
        "cfg = transformers.GPT2Config("
        "n_layer=1, n_head=2, n_embd=32, vocab_size=64, n_positions=32);"
        "m = transformers.AutoModelForCausalLM.from_config(cfg).eval().to(d);"
        "m(torch.tensor([[1, 2]]))"
    )
    try:
        done = subprocess.run(
            [sys.executable, "-c", probe],
            capture_output=True,
            timeout=300,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return False
    return done.returncode == 0


_DTYPE_OK: dict[str, bool] = {}


def require_dtype(dtype) -> None:
    """Skip rather than crash the worker, and say which machine could not."""
    name = str(dtype).removeprefix("torch.")
    if name not in _DTYPE_OK:
        _DTYPE_OK[name] = _dtype_survives_on_this_cpu(name)
    if not _DTYPE_OK[name]:
        pytest.skip(
            f"this CPU cannot run a forward pass in {name} — a probe process "
            f"died on it. The invariant below is dtype-free and still true; "
            f"there is no machine here to check it on."
        )


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.float16])
def test_the_final_row_agrees_with_the_model_in_every_dtype(dtype):
    """Regression: the lens double-normed its final row on every bf16 load.

    The old check compared LOGITS with `allclose(atol=1e-3, rtol=1e-3)`.
    float32 logits differed far below that tolerance and passed; bfloat16
    differed far above it and failed — but at the magnitude logits reach, a
    difference that size IS agreement to the last representable digit in
    bfloat16. The check was reading the dtype, not the model.

    The consequence was not subtle: the final row read a plausible wrong token
    where the model actually said something else, and `final` and `settled_at`
    are both derived from that row. Every bf16 session shipped a wrong answer
    on the one row a reader can check by eye.

    The invariant is dtype-free: the last lens row is the model, so its top
    token must be the model's top token.
    """
    transformers = pytest.importorskip("transformers")
    # BEFORE the model is built, because the thing being guarded against is a
    # hard crash rather than an exception — see `require_dtype`.
    require_dtype(dtype)
    cfg = transformers.GPT2Config(
        n_layer=3, n_head=2, n_embd=32, vocab_size=64, n_positions=32
    )
    torch.manual_seed(0)
    model = transformers.AutoModelForCausalLM.from_config(cfg).eval().to(dtype)

    class Tok:
        def decode(self, ids):
            return f"<{ids[0]}>"

    ids = torch.tensor([[1, 2, 3, 4]])
    with torch.no_grad():
        want = int(model(ids).logits[0, -1].float().argmax())

    out = lens.logit_lens(model, Tok(), ids, top_k=1)
    assert out["final"] == f"<{want}>", f"{dtype}: lens disagrees with the model"
    # And the floor really is a floor.
    assert out["reliability"]["floor_kl"] == pytest.approx(0.0, abs=1e-3)


def test_the_normed_check_tolerance_sits_between_rounding_and_a_real_miss():
    """bf16 rounding lands near 1e-4 nats; a genuine double-norm costs whole
    nats. The threshold has to separate those two, with room."""
    assert 1e-3 < lens.NORMED_KL_TOLERANCE < 1.0


def test_the_internal_probs_tensor_does_not_leak_into_the_response(tiny):
    """`_probs` is scratch for computing the KL. Shipping a torch tensor in a
    JSON payload would fail at serialisation, in the route rather than here."""
    import json

    model, tok = tiny
    out = lens.logit_lens(model, tok, torch.tensor([[1, 2]]), top_k=2)
    for row in out["layers"]:
        assert "_probs" not in row
    json.dumps(out)  # must not raise
