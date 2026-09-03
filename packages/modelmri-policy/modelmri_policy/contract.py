# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Muhammad Mahad Azher

"""The wire contract between ModelMRI and a policy sidecar.

Two processes, two environments, two pinned dependency sets. The whole reason
this package is separate is that lerobot's torch/numpy pins cannot share an
environment with ModelMRI's -- which means the two halves can be upgraded
independently, which means they can drift.

## The version is checked, not advertised

`CONTRACT` is declared HERE and declared again in `modelmri/policy.py`. That
duplication is deliberate and is the point: two independent statements of what
each side speaks, compared on every exchange. A single shared constant would
make drift undetectable, because both halves would move together by
construction even when only one of them was reinstalled.

ROADMAP #49 is explicit that this must be mandatory rather than advisory:
"contract drift silently serving actions from a stale policy is the worst
failure available here." An action chunk is a claim about what a robot would
DO. A stale one is not a degraded answer, it is a different policy's answer
wearing this one's name, and nothing downstream can tell.

So the version rides in EVERY response, not only the handshake. A sidecar can
be restarted underneath a live client -- upgraded, even -- and a check that
ran once at connect would not see it.

## What travels

    POST /act     {frames, state, instruction, seed, contract}
               -> {action_chunk, dtype, device, policy_repo, revision,
                   normalisation, contract}

    POST /hidden  {frames, state, instruction, layers, contract}
               -> safetensors bytes, with the contract in a header

`normalisation` is not decoration. A policy's action space is normalised
against ITS training statistics, and a dataset's recorded actions against its
own. Overlaying two curves in different units is the plausible-wrong output
ROADMAP #50 refuses to ship, and it cannot be refused without both sides
saying what units they are in.
"""

from __future__ import annotations

# Bumped when any payload above changes shape. Not a semver -- a handshake
# number. Adding an OPTIONAL response field does not move it; adding a
# required one, removing one, or changing what a field MEANS does.
CONTRACT = 1

# Loopback only, and not configurable. A process holding a policy and
# answering "what would the robot do" is not something to expose on a network
# interface by accident, and an argument that can be set to 0.0.0.0 is an
# argument somebody eventually sets to 0.0.0.0.
HOST = "127.0.0.1"

# 0 asks the OS for a free port. The chosen port is printed on stdout as
# `MODELMRI_POLICY_PORT=<n>` for the parent to read, rather than fixed here:
# a hard-coded port collides exactly when somebody runs two policies to
# compare them, which is a thing ROADMAP #45 asks for.
DEFAULT_PORT = 0

# The line the sidecar prints once it is serving. The parent waits for this
# rather than polling the port, so "started" means "ready to answer" instead
# of "the socket exists".
READY_PREFIX = "MODELMRI_POLICY_PORT="


class ContractError(RuntimeError):
    """The two halves do not speak the same contract, and neither guesses."""


def check(theirs: object, *, side: str) -> int:
    """Validate a contract number off the wire, or raise saying which side.

    `side` names whose number this is, because "contract mismatch" without it
    sends somebody to reinstall the wrong half.
    """
    if not isinstance(theirs, int) or isinstance(theirs, bool):
        raise ContractError(
            f"the {side} did not state a contract version (got "
            f"{type(theirs).__name__}). Every exchange carries one, so a "
            f"response without it is not from a sidecar this can trust."
        )
    if theirs != CONTRACT:
        raise ContractError(
            f"the {side} speaks contract {theirs}, this speaks {CONTRACT}. "
            f"These are different wire formats and an action chunk read "
            f"across them would be a different policy's answer wearing this "
            f"one's name. Reinstall the sidecar with "
            f"`modelmri policy install --force` so both halves match."
        )
    return theirs
