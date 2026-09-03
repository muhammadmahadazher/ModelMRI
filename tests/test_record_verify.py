# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Muhammad Mahad Azher

"""Refuse to patch an SDK whose shape has moved.

The failure this prevents is specific and nasty. The wrapper reads token
counts with `getattr(usage, "input_tokens", None)`, and the trace store's
columns are nullable so `None` can honestly mean "the provider reported
nothing". If the attribute MOVED, every span carries that same None — and the
token ledger then reports "not reported by provider" about a provider that
reported perfectly well. The absence is real and the reason is wrong.

The opposite failure is tested just as hard: a fingerprint too strict refuses
a working SDK after a harmless minor release, and the user loses tracing
entirely over a field nobody reads. So an SDK that gained fields, or whose
models cannot be introspected at all, must still patch.
"""

from __future__ import annotations

import sys
import types

import pytest
from modelmri_record import verify


def _fake_anthropic(
    *,
    usage_fields=(
        "input_tokens",
        "output_tokens",
        "cache_read_input_tokens",
        "cache_creation_input_tokens",
    ),
    message_fields=("usage", "content", "id", "role"),
    version="0.40.0",
    introspectable=True,
    create_params=("self", "model", "messages"),
):
    """An `anthropic` package with exactly the shape asked for."""
    pkg = types.ModuleType("anthropic")
    pkg.__version__ = version

    def _model(name, fields):
        cls = type(name, (), {})
        if introspectable:
            cls.model_fields = {f: object() for f in fields}
        return cls

    types_mod = types.ModuleType("anthropic.types")
    types_mod.Usage = _model("Usage", usage_fields)
    types_mod.Message = _model("Message", message_fields)

    resources = types.ModuleType("anthropic.resources")
    messages_mod = types.ModuleType("anthropic.resources.messages")

    src = f"def create({', '.join(create_params)}):\n    return None\n"
    ns: dict = {}
    exec(src, ns)
    messages_mod.Messages = type("Messages", (), {"create": ns["create"]})

    return {
        "anthropic": pkg,
        "anthropic.types": types_mod,
        "anthropic.resources": resources,
        "anthropic.resources.messages": messages_mod,
    }


@pytest.fixture
def install(monkeypatch):
    def _install(mods):
        for name, mod in mods.items():
            monkeypatch.setitem(sys.modules, name, mod)

    return _install


# ------------------------------------------------------------ it patches


def test_a_matching_sdk_reports_full_capture(install, monkeypatch):
    monkeypatch.delenv(verify.FORCE_ENV, raising=False)
    install(_fake_anthropic())
    report = verify.check()
    assert report.installed and report.ok
    assert report.capture == "full"
    assert report.missing == [] and report.missing_optional == []
    assert "every field the recorder reads is present" in report.reason()


def test_an_sdk_that_gained_fields_still_patches(install, monkeypatch):
    """Too strict is worse than too loose here: refusing over a field nobody
    reads takes tracing away for no gain."""
    monkeypatch.delenv(verify.FORCE_ENV, raising=False)
    install(
        _fake_anthropic(
            usage_fields=(
                "input_tokens",
                "output_tokens",
                "cache_read_input_tokens",
                "cache_creation_input_tokens",
                "server_tool_use",
                "some_new_thing",
            ),
            message_fields=("usage", "content", "id", "role", "container"),
        )
    )
    report = verify.check()
    assert report.ok and report.capture == "full"


def test_extra_keyword_only_params_do_not_trip_the_signature_check(
    install, monkeypatch
):
    monkeypatch.delenv(verify.FORCE_ENV, raising=False)
    install(_fake_anthropic(create_params=("self", "model", "messages", "**kwargs")))
    assert verify.check().ok


# ------------------------------------------------------------ it refuses


def test_a_moved_required_field_refuses_to_patch(install, monkeypatch):
    """This is the whole feature. `input_tokens` gone means every span would
    carry tokens_in=None, which the ledger reads as provider silence."""
    monkeypatch.delenv(verify.FORCE_ENV, raising=False)
    install(_fake_anthropic(usage_fields=("prompt_tokens", "completion_tokens")))
    report = verify.check()
    assert not report.ok
    assert report.capture == "none"
    assert "Usage.input_tokens" in report.missing
    assert "Usage.output_tokens" in report.missing


def test_the_refusal_names_the_package_the_version_and_what_moved(install, monkeypatch):
    monkeypatch.delenv(verify.FORCE_ENV, raising=False)
    install(_fake_anthropic(usage_fields=("output_tokens",), version="1.2.3"))
    reason = verify.check().reason()
    assert "anthropic 1.2.3" in reason
    assert "Usage.input_tokens" in reason
    assert "indistinguishable from a provider" in reason


def test_the_refusal_says_how_to_force_it(install, monkeypatch):
    """A refusal with no way past it is a dead end for somebody whose SDK is
    fine and whose fingerprint is merely stale."""
    monkeypatch.delenv(verify.FORCE_ENV, raising=False)
    install(_fake_anthropic(usage_fields=("prompt_tokens", "completion_tokens")))
    assert verify.FORCE_ENV in verify.check().reason()


def test_forcing_patches_anyway_and_says_so(install, monkeypatch):
    install(_fake_anthropic(usage_fields=("prompt_tokens",)))
    monkeypatch.setenv(verify.FORCE_ENV, "1")
    report = verify.check()
    assert report.ok and report.forced
    assert report.capture == "partial"
    assert "instrumenting anyway" in report.reason()


def test_a_missing_model_keyword_is_a_break(install, monkeypatch):
    monkeypatch.delenv(verify.FORCE_ENV, raising=False)
    install(_fake_anthropic(create_params=("self", "messages")))
    report = verify.check()
    assert not report.ok
    assert any("model" in m for m in report.missing)


# ------------------------------------------- optional fields are not breaks


def test_a_version_without_cache_fields_still_patches(install, monkeypatch):
    """Anthropic added prompt caching partway through. An older SDK is not
    broken — it simply cannot supply those columns."""
    monkeypatch.delenv(verify.FORCE_ENV, raising=False)
    install(_fake_anthropic(usage_fields=("input_tokens", "output_tokens")))
    report = verify.check()
    assert report.ok
    assert report.capture == "partial"
    assert report.missing == []
    assert "Usage.cache_read_input_tokens" in report.missing_optional


def test_the_optional_note_blames_the_sdk_not_the_user(install, monkeypatch):
    """'not reported by provider' for every call is true of this SDK version,
    not of how somebody is calling it."""
    monkeypatch.delenv(verify.FORCE_ENV, raising=False)
    install(_fake_anthropic(usage_fields=("input_tokens", "output_tokens")))
    assert "true of this SDK, not of your usage" in verify.check().reason()


# --------------------------------------------- unknown is not the same as broken


def test_an_unintrospectable_model_patches_as_partial(install, monkeypatch):
    """Refusing on 'I could not tell' would take tracing from anybody whose
    SDK is merely unusual."""
    monkeypatch.delenv(verify.FORCE_ENV, raising=False)
    install(_fake_anthropic(introspectable=False))
    report = verify.check()
    assert report.ok
    assert report.capture == "partial"
    assert report.missing == []
    assert any("not 'broken'" in n for n in report.notes)


def test_no_anthropic_installed_is_not_an_error(monkeypatch):
    monkeypatch.setitem(sys.modules, "anthropic", None)
    report = verify.check()
    assert not report.installed
    assert "nothing to instrument" in report.reason()


def test_check_never_raises_whatever_the_sdk_looks_like(install, monkeypatch):
    """It runs at import inside somebody else's process."""
    monkeypatch.delenv(verify.FORCE_ENV, raising=False)
    broken = types.ModuleType("anthropic")

    class _Exploding:
        def __getattr__(self, name):
            raise RuntimeError("this SDK hates introspection")

    broken.__version__ = _Exploding()
    monkeypatch.setitem(sys.modules, "anthropic", broken)
    monkeypatch.setitem(sys.modules, "anthropic.types", None)
    monkeypatch.setitem(sys.modules, "anthropic.resources.messages", None)
    report = verify.check()  # must not raise
    assert isinstance(report.to_dict(), dict)


# ------------------------------------------------------------- the doctor


def test_the_doctor_exits_nonzero_when_it_will_not_patch(install, monkeypatch, capsys):
    """So a CI step can catch 'tracing silently stopped after a bump'."""
    monkeypatch.delenv(verify.FORCE_ENV, raising=False)
    install(_fake_anthropic(usage_fields=("prompt_tokens",)))
    from modelmri_record.__main__ import main

    assert main(["doctor"]) == 1
    out = capsys.readouterr().out
    assert "MOVED" in out and "Usage.input_tokens" in out


def test_the_doctor_exits_zero_on_a_healthy_sdk(install, monkeypatch, capsys):
    monkeypatch.delenv(verify.FORCE_ENV, raising=False)
    install(_fake_anthropic())
    from modelmri_record.__main__ import main

    assert main(["doctor"]) == 0
    assert "capture   : full" in capsys.readouterr().out


def test_the_doctor_speaks_json_for_a_script(install, monkeypatch, capsys):
    import json

    monkeypatch.delenv(verify.FORCE_ENV, raising=False)
    install(_fake_anthropic(usage_fields=("input_tokens", "output_tokens")))
    from modelmri_record.__main__ import main

    assert main(["doctor", "--json"]) == 0
    doc = json.loads(capsys.readouterr().out)
    assert doc["capture"] == "partial"
    assert doc["missing_optional"] == [
        "Usage.cache_read_input_tokens",
        "Usage.cache_creation_input_tokens",
    ]


# --------------------------------------------- the instrumentation itself


def test_instrument_refuses_and_does_not_patch(install, monkeypatch, capsys):
    monkeypatch.delenv(verify.FORCE_ENV, raising=False)
    mods = _fake_anthropic(usage_fields=("prompt_tokens",))
    install(mods)
    import modelmri_record

    before = mods["anthropic.resources.messages"].Messages.create
    assert modelmri_record.instrument_anthropic() is False
    after = mods["anthropic.resources.messages"].Messages.create
    assert after is before, "it patched an SDK it said it would not patch"
    assert "modelmri-record:" in capsys.readouterr().err


def test_instrument_patches_a_healthy_sdk(install, monkeypatch):
    monkeypatch.delenv(verify.FORCE_ENV, raising=False)
    mods = _fake_anthropic()
    install(mods)
    import modelmri_record

    assert modelmri_record.instrument_anthropic() is True
    assert getattr(
        mods["anthropic.resources.messages"].Messages.create, "_modelmri_wrapped", False
    )


def test_not_installed_is_a_different_exit_code_than_a_broken_shape(
    monkeypatch, capsys
):
    """Folding "you do not use Anthropic" into "your SDK broke" would make a
    CI check fail forever for anybody who simply does not use it."""
    from modelmri_record.__main__ import ABSENT, MOVED, main

    monkeypatch.delenv(verify.FORCE_ENV, raising=False)
    monkeypatch.setitem(sys.modules, "anthropic", None)
    assert main(["doctor"]) == ABSENT
    assert ABSENT != MOVED


def test_usage_error_is_its_own_code(capsys):
    from modelmri_record.__main__ import USAGE, main

    assert main(["nonsense"]) == USAGE


def test_the_fingerprint_covers_exactly_what_the_wrapper_reads():
    """The realistic drift: somebody adds a `getattr(usage, ...)` to the
    wrapper and forgets the fingerprint, so the recorder starts reading a
    field nothing checks — and a version that moved it goes back to producing
    silently empty spans.

    `anthropic` is deliberately not a declared dependency (the recorder
    discovers it via try/import), so this cannot run against the published
    package. It checks the two lists against each other instead, which is the
    failure that actually happens.
    """
    import inspect
    import re

    import modelmri_record

    src = inspect.getsource(modelmri_record.instrument_anthropic)
    read = set(re.findall(r'getattr\(\s*usage\s*,\s*["\'](\w+)["\']', src))
    # `getattr(usage, "x", None)` split across lines by the formatter.
    read |= set(re.findall(r'getattr\(\s*\n?\s*usage\s*,\s*\n?\s*["\'](\w+)["\']', src))

    fingerprinted = {attr for owner, attr, _ in verify.USAGE_FIELDS if owner == "Usage"}
    unchecked = read - fingerprinted
    assert not unchecked, (
        f"the wrapper reads {sorted(unchecked)} off usage, and the fingerprint "
        f"does not check for it — a version that moved it would produce empty "
        f"token fields with nothing to say why"
    )
