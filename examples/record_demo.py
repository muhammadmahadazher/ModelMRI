"""Generate a realistic demo agent trace — the "failed at step N" story.

Run while `modelmri serve` is up:  uv run python examples/record_demo.py
The trace appears in the AGENTS panel at http://localhost:5900.
No API keys needed; steps carry explicit timings shaped like a real run.

It uses eight of the ten step kinds, retrieval leg included, because the
timeline's colours and shapes are the thing this sample exists to show and a
sample made of two kinds shows two of them.
"""

from modelmri.record import step, trace

# Tagged as a demo so the viewer labels it. Without this it sat in the list
# looking exactly like a run you had recorded — including its deliberately
# failed `git push`, which read as your agent failing rather than as sample
# data doing its job.
with trace("fix-failing-tests-run", meta={"demo": True}):
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
    # The retrieval leg. Three steps that would all have had to be `tool_call`
    # before these kinds existed, which is exactly the problem: "how much of
    # this run was retrieval" is unanswerable when retrieval is spelled the
    # same as everything else. They are here so the shipped sample shows what
    # a RAG-shaped run looks like on the timeline without anybody having to
    # write an agent first.
    # NO `tokens_in` HERE, and the omission is the interesting part. Embedding
    # providers do report an input token count, and this step carried one for
    # a while — but `ledger.TOKEN_KINDS` is `("llm_call",)` on purpose (the
    # whole rollup is named `n_llm_steps`, so folding embeddings in would
    # restate every stored run), and a step outside that tuple has its counts
    # neither summed nor counted among the ones that "reported nothing". The
    # visible result was this sample printing "14 tok in" in the inspector
    # header and, one line below it, "no LLM calls here, so there are no
    # tokens to count" — about the same step. A shipped sample must not
    # demonstrate a contradiction, and leaving the field off says nothing
    # false: absent is not zero. `tests/test_step_kinds.py` holds this.
    step(
        "embedding",
        name="text-embedding-3-small · failing assertion",
        input="assert token.expiry > now()",
        output="1536-d vector",
        started_ms=6150,
        duration_ms=180,
    )
    step(
        "retrieval",
        name="vector-store: src/**",
        input="how is token expiry computed",
        output="24 candidates — src/auth.py, src/session.py, src/clock.py, …",
        started_ms=6350,
        duration_ms=640,
    )
    step(
        "rerank",
        name="cross-encoder · 24 → 3",
        output="src/auth.py 0.91, src/clock.py 0.44, src/session.py 0.38",
        started_ms=7010,
        duration_ms=220,
    )
    with step("subagent", name="auth-fixer", started_ms=7280, duration_ms=7000):
        step(
            "llm_call",
            name="claude-sonnet-5 · patch",
            input="Failing assertions + auth.py source",
            output="The expiry is set in UTC but compared naive. Patching...",
            started_ms=7330,
            duration_ms=2650,
            tokens_in=2114,
            tokens_out=402,
        )
        step(
            "tool_call",
            name="edit: src/auth.py",
            output="+ expiry = datetime.now(timezone.utc) + ttl",
            started_ms=10030,
            duration_ms=310,
        )
        step(
            "tool_call",
            name="bash: pytest -q tests/test_auth.py",
            output="17 passed",
            started_ms=10380,
            duration_ms=3905,
        )
    step(
        "llm_call",
        name="claude-sonnet-5 · commit msg",
        output="fix(auth): compare token expiry in UTC",
        started_ms=14380,
        duration_ms=980,
        tokens_in=640,
        tokens_out=52,
    )
    # A guardrail that PASSES, and deliberately so. The one below it fails, and
    # a sample where the only safety check is the one that fired would teach
    # the wrong thing about the kind: `guardrail` is not `error`, and a
    # guardrail doing its job is the ordinary case.
    step(
        "guardrail",
        name="policy: no force-push to main",
        input="git push",
        output="allowed",
        started_ms=15400,
        duration_ms=60,
    )
    step(
        "tool_call",
        name="bash: git push",
        output="remote: Permission denied (publickey).",
        started_ms=15520,
        duration_ms=1530,
        error=True,
    )
    step(
        "llm_call",
        name="claude-sonnet-5 · recover",
        input="git push failed: Permission denied (publickey)",
        output="SSH agent has no key loaded. I can't fix credentials — reporting back.",
        started_ms=17120,
        duration_ms=1210,
        tokens_in=488,
        tokens_out=96,
    )

print("demo trace delivered (or saved to ./modelmri-traces/ if server is down)")
