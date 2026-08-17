"""Two words for "no", so the server can tell them apart from "broken".

Every refusal in this project used to be a `RuntimeError`, and so is a CUDA
out-of-memory, a shape mismatch deep in transformers, and a socket that closed
mid-read. The API could not distinguish them, so it answered all of them the
same way: 409 Conflict, with the exception's own text pasted into the body. A
full GPU is not a conflict, and torch's message is not a sentence anybody wrote
for a reader — it is machinery talking to itself, and it can carry filesystem
paths from the machine the server is running on.

So there are three kinds of answer, and each gets its own status:

  `Refusal`      409 — ModelMRI decided not to answer, and says why in words
                       written for the reader. "This is a recording. Ranking
                       heads means running the model, and a `.mri` does not
                       carry one."
  `BadRequest`   422 — the request itself is malformed: a layer index outside
                       the model, an unknown baseline name, a repo id that is
                       not a causal LM.
  anything else  500 — something broke. The reader gets one generic sentence
                       pointing at the terminal they already have open, and
                       the traceback goes to the log rather than the browser.

WHY THESE SUBCLASS RuntimeError AND ValueError, which looks redundant:

so that the migration cannot half-land. Every `except RuntimeError` and
`except ValueError` already in this codebase keeps catching these, unchanged,
whether or not the module it guards has been converted yet. A raise site that
becomes a `Refusal` before its handler learns the word still answers 409
through the old arm; a handler that learns the word before its module does
still catches the plain `RuntimeError` through an explicit transitional arm.
Neither half of the change can silently turn a refusal into a 500 while the
other half is in flight. Do not "tidy" the base classes away — that safety net
is the reason they are here, and it costs one word each.
"""

from __future__ import annotations


class Refusal(RuntimeError):
    """A deliberate answer of "no", in words written for the reader.

    Raise this when ModelMRI understood the request perfectly and is declining
    it: the measurement is impossible here (a `.mri` carries no model), the
    backend cannot do it (Ollama has no forward pass to intervene in), or the
    order is wrong (nothing has been generated yet). The message is the whole
    point — it is what the user reads, so it says what happened and what to do
    next.

    Reaches the client as 409 with `str(err)` as the body. That means the
    message is published, and the rule is about where the words came from:
    never interpolate a caught exception's `str`, its args, or a Python repr.
    Those are machinery talking to itself, and they carry whatever the library
    underneath felt like carrying.

    A path this program CHOSE and the reader has to act on is a different
    thing, and it is allowed: "lerobot/pusht is not cached. Looked in: <the
    three directories that were checked>" is the answer, and a version with
    the directories removed would be a worse one — this tool runs on the
    reader's own machine, and the directory is the instruction. What is
    forbidden is a path that arrived *from an exception*, because nobody chose
    it and nobody checked what else came with it. vla.py holds a stricter line
    for itself (see the comment at its `model.safetensors` refusal, where the
    repo id is enough); that is a house style on top of this rule, not a
    contradiction of it.

    If you cannot write the sentence, this is probably not a refusal.

    Subclasses RuntimeError on purpose — see the module docstring.

    ## `sentence` is the published half, named

    Everything above is a rule about where the words came from, and it lived
    only in prose. `sentence` makes it a real attribute: it is set once, from
    the message this class was CONSTRUCTED with, and it is what the routes
    publish.

    Two things follow. A reader of a handler can see that what reaches the
    browser is an authored sentence rather than whatever `str()` on an
    exception happens to produce — the distinction the whole module is about.
    And a static analyser can see it too: `str(err)` on a caught exception is
    the literal signature of a stack-trace leak, and no analyser can know this
    project only ever puts its own prose there. Reading a field assigned from
    a literal is a different flow, and an honest one.

    It is exactly `str(self)` today. That is the point — this is naming an
    invariant that already held, not changing behaviour.
    """

    def __init__(self, *args):
        super().__init__(*args)
        # `args[0]` rather than `str(self)`: an exception built with no
        # message stringifies to `""`, and one built with several
        # stringifies to a tuple repr — neither is a sentence anybody wrote.
        self.sentence = str(args[0]) if args else ""


class BadRequest(ValueError):
    """The request itself is malformed — a bad layer index, an unknown
    baseline name.

    The distinction from `Refusal` is about what the caller has to change.
    A `Refusal` says "not here, not like this" about a state the caller can
    only fix by doing something else first; a `BadRequest` says "that
    parameter is wrong" about the call they just made. Range checks, enum
    validation and schema violations are all this.

    Reaches the client as 422 with `str(err)` as the body, so the same rule
    applies: the message is published, and it should name the acceptable
    values rather than merely rejecting the given one.

    Subclasses ValueError on purpose — see the module docstring.

    Carries `sentence` for the same reason `Refusal` does, and the two must
    keep carrying it identically: a handler that catches both and publishes
    one of them differently is the seam where one of the two stops being
    checked.
    """

    def __init__(self, *args):
        super().__init__(*args)
        self.sentence = str(args[0]) if args else ""
