# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Muhammad Mahad Azher

"""Sentences to draw replacement activations from, and where they came from.

The resample baseline replaces a head with what it really computes on some
other sentence. Which other sentence is not a detail — a head that looks
irrelevant against eight sentences about weather may look load-bearing against
eight about geography, and the score would move without anything about the
model changing. So the corpus is part of the measurement, it is named in every
response, and the default is bundled rather than fetched.

**Bundled, not downloaded.** A corpus pulled from the network at analysis time
would make the same command produce different numbers on different days, and
would break the offline promise the rest of this package keeps. These sentences
ship with the wheel, so a resample score computed here reproduces on any
machine running the same version.

**Deliberately dull.** Weather, trains, coffee, a dog on a path. Nothing about
capitals, countries, landmarks or people, because the prompts this tool is
pointed at are overwhelmingly factual recall — "The capital of France is", "The
Eiffel Tower is located in the city of" — and a donor sentence that also talks
about geography would be replacing a head's France-related work with more
France-related work. That is not a neutral baseline, it is a differently biased
one, and it would understate every head that carries the fact.

Override with `MODELMRI_CORPUS` pointing at a text file, one sentence per line.
The label in the response changes with it, so the two are never confusable.
"""

from __future__ import annotations

import os
from pathlib import Path

from .errors import BadRequest, Refusal

# Long enough to reach the position most prompts attribute at, and ordinary
# enough to carry no topic the analysed prompt is likely to share.
BUILT_IN = (
    "The weather today is unusually warm for this time of year outside",
    "She opened the book and began to read slowly by the window",
    "Engineers replaced the bridge cables over three long weeks last winter",
    "He poured the coffee and stared out the window for a while",
    "Their train arrived late because of signal problems on the line again",
    "The committee agreed to postpone the vote until Friday next week",
    "Rain fell steadily against the roof throughout the night until morning",
    "A small dog followed them along the gravel path towards the gate",
    "The lamp flickered twice and then settled into a steady warm glow",
    "Workers painted the fence a pale green over the course of Saturday",
    "He folded the letter carefully and placed it back inside its envelope",
    "The kettle boiled while she searched the drawer for a clean spoon",
    "Someone had left a bicycle leaning against the wall near the door",
    "The meeting ran long and everyone left the room feeling rather tired",
    "Snow covered the garden path and the steps leading up to it",
    "She counted the boxes twice before signing the delivery note properly",
)

BUILT_IN_LABEL = f"built-in ({len(BUILT_IN)} plain sentences)"


def load() -> tuple[list[str], str]:
    """(sentences, a label naming where they came from).

    The label travels with every score computed from these sentences. It is not
    provenance trivia: two resample runs with different labels are two different
    measurements, and presenting them as comparable would be the error this
    whole module exists to prevent.
    """
    raw = (os.environ.get("MODELMRI_CORPUS") or "").strip()
    if not raw:
        return list(BUILT_IN), BUILT_IN_LABEL

    path = Path(raw).expanduser()
    if not path.is_file():
        raise Refusal(
            f"MODELMRI_CORPUS points at {path}, which is not a file. Unset it "
            f"to use the {BUILT_IN_LABEL}, or point it at a text file with one "
            "sentence per line."
        )
    try:
        lines = [ln.strip() for ln in path.read_text(encoding="utf-8").splitlines()]
    except (OSError, UnicodeDecodeError) as err:
        raise Refusal(
            f"MODELMRI_CORPUS at {path} could not be read as UTF-8 text. It "
            "should be one sentence per line."
        ) from err

    sentences = [ln for ln in lines if ln]
    if not sentences:
        raise Refusal(
            f"MODELMRI_CORPUS at {path} has no non-empty lines, so there is "
            "nothing to draw replacement activations from."
        )
    return sentences, f"{path.name} ({len(sentences)} lines)"


def donor_ids(tokenizer, sentences: list[str], *, at_least: int, want: int, device):
    """Tokenize, keep the ones long enough, and refuse clearly when too few.

    "Long enough" is not negotiable: a donor shorter than the analysed prompt
    has no activation to put at the later positions, and padding one out would
    score the padding. `ablate._donors_for` refuses on the same rule; this
    catches it earlier, where the message can name the corpus and the fix.
    """
    if want < 1:
        raise BadRequest("want must be at least 1")

    encoded = [tokenizer(s, return_tensors="pt").input_ids for s in sentences]
    long_enough = [ids for ids in encoded if int(ids.shape[1]) >= at_least]

    if len(long_enough) < want:
        longest = max((int(i.shape[1]) for i in encoded), default=0)
        raise Refusal(
            f"the resample baseline needs {want} corpus sentences at least "
            f"{at_least} tokens long, and this corpus has {len(long_enough)}. "
            f"Its longest is {longest} tokens. Point MODELMRI_CORPUS at a file "
            "with longer sentences, or attribute at an earlier position — "
            "padding a short donor out would score the padding rather than "
            "the head."
        )
    return [ids.to(device) for ids in long_enough[:want]]
