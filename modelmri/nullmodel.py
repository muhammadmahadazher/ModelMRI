# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Muhammad Mahad Azher

"""The same architecture, untrained. Does your finding survive it?

Every ranking in this package answers "which head moved the answer most". None
of them answers the question underneath: **would this measurement have produced
a confident, ordered list anyway?**

It might. A 2025 result found automated interpretability metrics failing to
distinguish trained transformers from randomly initialised ones — the pipeline
runs, the numbers come out ordered, the top head looks meaningful, and none of
it is about anything the model learned. Ablation is not immune: removing a head
from an untrained network also perturbs the output, also by different amounts
per head, and the result is also a leaderboard.

So this builds the same architecture with random weights and runs the identical
measurement through it. If the two rankings agree, the panel says so, and what
it says is that the measurement is uninformative on this model — not that the
model is uninteresting.

Measured with bf16 on an RTX 4060, "The capital of France is", zero
baseline, layer 0, seed 0 — the trained model's top heads and the untrained
twin's are different heads, the rank correlation is negative, and the two top
fives share one entry. Two things in that: the order does not survive
(good — the ranking is about training), and the untrained twin's scores are
tens of times smaller and nearly uniform, which is what "this head did
not do anything in particular" looks like when nothing has learned anything.

That is the outcome you want and it is not the guaranteed one. The control is
worth shipping precisely because it can come back the other way.

**From the config alone.** `AutoModelForCausalLM.from_config` reads
`config.json` and initialises; it fetches no weights, so the control works
air-gapped and costs one model's worth of memory rather than a download. The
seed is reported, because a control whose seed is not stated is a control that
cannot be re-run.

**The same code paths, not a parallel implementation.** The twin is handed to
`ablate.rank_heads` and `attribute.attribute_tokens` exactly as the real model
is. A second implementation of the measurement would be free to differ from the
one being checked, which would make agreement meaningless in both directions.

**It costs a second model in memory.** Priced through `budget` before it is
built, and refused rather than attempted when it will not fit — an OOM halfway
through a control is worse than no control, because the panel has already told
the reader one is coming.
"""

from __future__ import annotations

from typing import Any

from .errors import Refusal


def build_twin(config: Any, *, seed: int, dtype: Any, device: str):
    """The same architecture, randomly initialised, on the same device.

    `config` is the loaded model's own config object, so the twin cannot drift
    from it — no re-reading a file that may have changed, no re-deriving a
    shape. Deterministic given `seed`, which is why the seed is returned to the
    caller and printed on screen.
    """
    import torch
    from transformers import AutoModelForCausalLM

    torch.manual_seed(seed)
    try:
        twin = AutoModelForCausalLM.from_config(config)
    except Exception as err:
        # A config transformers can load weights for but not instantiate from
        # scratch is rare and real (remote code, custom architectures). Naming
        # the class is enough — the exception's own text is machinery talking
        # to itself and may carry paths from this machine.
        raise Refusal(
            f"this model's architecture ({type(config).__name__}) cannot be "
            "built from its config alone, so there is no untrained twin to "
            "compare against. The ranking beside it is unaffected."
        ) from err

    twin.eval()
    # Same dtype and device as the real model, or the two runs differ by more
    # than training and the comparison stops being about training at all.
    return twin.to(dtype=dtype, device=device)


def teardown(twin) -> None:
    """Give the accelerator memory back, even though the caller still holds it.

    `del twin` here only unbinds this function's own parameter — the caller's
    variable is still a live reference, so `gc.collect()` collects nothing and
    `empty_cache()` has nothing to release. Measured: every byte a twin
    allocated was still allocated after teardown returned. The
    docstring claimed the memory came back immediately and it did not, which on
    an 8 GB card is the difference between the next analysis running and
    refusing.

    Moving the parameters to CPU frees the CUDA storage regardless of how many
    Python references survive, which is the only thing that works from inside a
    function the caller called.
    """
    import gc

    import torch

    try:
        twin.to("cpu")
    except Exception:  # noqa: S110 - the collector below is the fallback
        # A model that will not move is one we cannot free this way; the
        # collector below is then the only lever, and saying nothing would be
        # worse than trying and moving on.
        pass
    del twin
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def verdict(rho: float | None, *, top_k_shared: int, top_k: int) -> str:
    """One sentence about what the comparison means. No number invented.

    The thresholds are stated rather than tuned: this is a reading aid on top
    of the correlation, and the correlation is printed beside it so a reader
    who disagrees with the wording can see the number it came from.
    """
    if rho is None:
        return (
            "The untrained twin produced no ranking to compare against — its "
            "scores were all equal, so there is nothing to correlate."
        )
    if top_k <= 0:
        # Defensive: `compare_baselines` cannot produce top_k == 0 alongside a
        # real correlation, since a ranking short enough to give top_k 0 is
        # also too short for spearman to be defined. But the sentence below
        # would read "sharing 0 of the top 0" if it ever did, which is a
        # conclusion drawn from nothing.
        return (
            f"The untrained twin correlates at Spearman {rho:.2f}, but no top "
            "heads were compared, so there is nothing to say about which heads "
            "the two agree on."
        )
    if rho >= 0.7 or top_k_shared >= max(1, top_k - 1):
        return (
            f"An untrained model with the same shape ranks these heads almost "
            f"the same way (Spearman {rho:.2f}, sharing {top_k_shared} of the "
            f"top {top_k}). On this prompt the ranking is mostly reporting the "
            "architecture, not what this model learned."
        )
    if rho >= 0.3:
        return (
            f"An untrained model with the same shape partly reproduces this "
            f"ranking (Spearman {rho:.2f}, sharing {top_k_shared} of the top "
            f"{top_k}). Some of what you are seeing is the architecture."
        )
    return (
        f"An untrained model with the same shape ranks these heads differently "
        f"(Spearman {rho:.2f}, sharing {top_k_shared} of the top {top_k}), so "
        "this ranking is not just the shape of the network."
    )
