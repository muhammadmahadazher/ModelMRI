"""`python -m modelmri_record doctor` — is the SDK on THIS machine traceable?

The recorder runs inside somebody else's process, where a failure to
instrument is invisible: no trace appears, and a trace that never existed
looks exactly like one that was never started. This is the command that tells
them which, before they go looking through their agent code for the bug.

It answers offline. The fingerprint reads the response models' declared
fields rather than making a call, so no API key and no money.
"""

from __future__ import annotations

import json
import sys

from . import verify

# Three outcomes, three codes, because a script has to tell them apart:
#
#   0 — instrumentable.
#   1 — the SDK is here and its shape MOVED. The actionable one: "my tracing
#       silently stopped working after a dependency bump" is exactly the
#       failure this feature exists to make loud, and it belongs in CI.
#   3 — anthropic is not installed. Nothing is broken and nothing is wrong;
#       folding this into 1 would make the check fail forever for anybody who
#       simply does not use Anthropic.
#
# 2 stays the usual "you called this wrong".
OK, MOVED, USAGE, ABSENT = 0, 1, 2, 3


def doctor(as_json: bool = False) -> int:
    """Print whether the SDK is instrumentable, and what moved if not."""
    report = verify.check()
    code = OK if report.ok else (ABSENT if not report.installed else MOVED)
    if as_json:
        print(json.dumps(report.to_dict(), indent=2))
        return code

    print(f"modelmri-record doctor — {report.package}")
    print(f"  installed : {'yes' if report.installed else 'no'}")
    if report.version:
        print(f"  version   : {report.version}")
    if report.installed:
        print(f"  capture   : {report.capture}")
        for name in report.missing:
            print(f"  MOVED     : {name}  (required — this is why it will not patch)")
        for name in report.missing_optional:
            print(
                f"  absent    : {name}  (optional — that column reads 'not reported')"
            )
        for note in report.notes:
            print(f"  note      : {note}")
    print()
    print(report.reason())
    return code


def main(argv=None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    as_json = "--json" in args
    args = [a for a in args if a != "--json"]
    if args and args[0] == "doctor":
        return doctor(as_json=as_json)
    print("usage: python -m modelmri_record doctor [--json]", file=sys.stderr)
    return USAGE


if __name__ == "__main__":
    raise SystemExit(main())
