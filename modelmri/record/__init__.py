"""The agent recorder — re-exported from the standalone `modelmri-record`.

This module used to be a second copy of the recorder, kept in step by hand.
It fell behind, and the way it fell behind mattered: the standalone package
grew credential redaction and this copy did not, while the README told people
to import from *here*. So the documented path had no scrubbing at all, and
SECURITY.md promised that credentials are removed before anything leaves your
process. For anyone following the README, that promise was not being kept.

There is now one implementation. `modelmri.record` and `modelmri_record` are
the same objects, so a redaction fix cannot land in one and miss the other.

    from modelmri.record import trace, step     # this module
    from modelmri_record import trace, step     # identical, no modelmri needed

The standalone package is the one to depend on inside an agent: it is stdlib
only, so it drags in no torch, no fastapi, nothing.
"""

from __future__ import annotations

try:
    from modelmri_record import (
        DEFAULT_ENDPOINT,
        __version__,
        instrument_anthropic,
        step,
        trace,
    )
    from modelmri_record import redact as redact
except ModuleNotFoundError as err:  # pragma: no cover - packaging accident
    raise ModuleNotFoundError(
        "modelmri.record needs the `modelmri-record` package, which modelmri "
        "depends on. Reinstall with `pip install --upgrade modelmri`, or "
        "install it directly: `pip install modelmri-record`."
    ) from err

__all__ = [
    "DEFAULT_ENDPOINT",
    "__version__",
    "instrument_anthropic",
    "redact",
    "step",
    "trace",
]
