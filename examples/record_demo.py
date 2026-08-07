"""Generate a realistic demo agent trace — the "failed at step N" story.

Run while `modelmri serve` is up:  uv run python examples/record_demo.py
The trace appears in the AGENTS panel at http://localhost:5900.
No API keys needed; steps carry explicit timings shaped like a real run.
"""

from modelmri.record import step, trace

with trace("fix-failing-tests-run"):
    step(
        "user_turn",
        name="task",
        input="Fix the 3 failing tests in tests/test_auth.py and push.",
        started_ms=0,
        duration_ms=5,
    )
    step(
        "llm_call",
        name="claude-sonnet-5 · plan",
        input="Fix the 3 failing tests in tests/test_auth.py...",
        output="I'll run the suite first to see the failures, then patch.",
        started_ms=10,
        duration_ms=1840,
        tokens_in=912,
        tokens_out=138,
    )
    step(
        "tool_call",
        name="bash: pytest -q tests/test_auth.py",
        output="3 failed, 14 passed — assert token.expiry > now() fails",
        started_ms=1900,
        duration_ms=4210,
    )
    with step("subagent", name="auth-fixer", started_ms=6150, duration_ms=7000):
        step(
            "llm_call",
            name="claude-sonnet-5 · patch",
            input="Failing assertions + auth.py source",
            output="The expiry is set in UTC but compared naive. Patching...",
            started_ms=6200,
            duration_ms=2650,
            tokens_in=2114,
            tokens_out=402,
        )
        step(
            "tool_call",
            name="edit: src/auth.py",
            output="+ expiry = datetime.now(timezone.utc) + ttl",
            started_ms=8900,
            duration_ms=310,
        )
        step(
            "tool_call",
            name="bash: pytest -q tests/test_auth.py",
            output="17 passed",
            started_ms=9250,
            duration_ms=3905,
        )
    step(
        "llm_call",
        name="claude-sonnet-5 · commit msg",
        output="fix(auth): compare token expiry in UTC",
        started_ms=13250,
        duration_ms=980,
        tokens_in=640,
        tokens_out=52,
    )
    step(
        "tool_call",
        name="bash: git push",
        output="remote: Permission denied (publickey).",
        started_ms=14300,
        duration_ms=1530,
        error=True,
    )
    step(
        "llm_call",
        name="claude-sonnet-5 · recover",
        input="git push failed: Permission denied (publickey)",
        output="SSH agent has no key loaded. I can't fix credentials — reporting back.",
        started_ms=15900,
        duration_ms=1210,
        tokens_in=488,
        tokens_out=96,
    )

print("demo trace delivered (or saved to ./modelmri-traces/ if server is down)")
