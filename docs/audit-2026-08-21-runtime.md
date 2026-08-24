# Runtime audit — main @ f80fb34, exercised rather than read

## Method

Seven agents actually RAN the surface — no model loaded, no downloads, `TestClient(create_app())` for HTTP, real JSON-RPC for MCP, subprocess for the CLI, direct calls for library functions — instead of reading code for plausible-looking bugs. Every finding was then handed to an independent second agent told to reproduce it or refute it; only survivors are listed here.

112 raw findings → 46 independently reproduced → **44 distinct defects** after dedup (two pairs turned out to be one root cause on two surfaces).

Surface exercised: all 149 HTTP routes (sharded across 3 agents), all 18 CLI subcommands, the MCP server over stdio, library functions in `fit.py`/`image_fit.py`/`capacity.py`/`patterns.py`/`traces.py`/`sweep.py`/`vision_attr.py`/`image_cv.py`/`head_types.py`/`fmt.py`/`mri_diff.py`/`imaging.py`/`weights_scan.py` with empty/zero/negative/huge/NaN/unicode/bool-as-int inputs, every `frontend/src/api.ts` fetch compared against the real response, and every piece of process-wide mutable state hammered from a thread pool.

Shorthand:
```
PY = C:/venvs/modelmri/Scripts/python.exe    MM = C:/venvs/modelmri/Scripts/modelmri.exe
TC  = TestClient(create_app())
TCL = TestClient(create_app(), client=("127.0.0.1", 5000))   # loopback-guarded routes
```

## Status

**ALL 44 DEFECTS ARE FIXED.** Every item below carries the commit that closed it.

Fix now — 1.1 `627857f`, 1.2 `0f93582`, 1.3 `3493d88`, 1.4 `7e04e00`, 2.1 `d6b7dcf`, 2.2 `36ad22b`, 3.1 `1be2b5c`.

Next — 2.3 `9e08788`, 2.4 `04c126d`, 3.2 `e6fa391`, 3.3+3.4 `8143c98`, 3.5 `b0b53eb`, 3.6 `d7eadaf`.

Can wait — 1.5 `af7c4e0`, 1.6+3.7+3.8 `7839314`, 1.7a-d `93f4d71`, 2.5 `3c2ac49`, 2.6 `a48b054`, 2.7 `b577402`, 3.9 `ee33c81`, 4.1+4.2 `f305a52`, 4.3 `da5b8a1`, 4.4 `7bda8de`.

Four items were investigated and confirmed DELIBERATE — see "Not defects" at
the foot of this file. Fixing any of them would be a regression.

Three guards were added so these classes cannot recur silently:

  `tests/test_api_contract.py`                     diffs every reachable GET
                                                   payload against its `api.ts`
                                                   interface — 4.4, 2.2, 3.9
  `tests/demo_check.py::payload_shapes`            the same question for the
                                                   demo's own handlers — 3.9
  `tests/test_unknown_body_keys.py`                fails on a new raw-dict POST
                                                   route outside an allowlist
                                                   that must state a reason — 3.1

Ranked: crashes/hangs/data-loss → wrong numbers in front of a user → silent failures → messages and types only.

---

# TIER 1 — crashes, hangs, data loss

### 1.1 `POST /api/traces/import` 500s inside the write transaction and destroys the trace it was overwriting — FIXED (`627857f`)

**Root cause:** `traces.import_trace`'s pre-lock validation loop (`traces.py:509-535`) checks `kind`, `parent_id`, `meta` but not the five token fields the insert actually binds, and `_ms` (`:941-974`) catches `TypeError`/`ValueError` but not `OverflowError`. The raise happens *after* `_retract_from_index` and `DELETE FROM step` have already run.

**Repro:**
```bash
$PY -c "import tempfile;from pathlib import Path;from modelmri import traces as T
s=T.TraceStore(Path(tempfile.mkdtemp())/'t.db')
s.import_trace({'id':'x','name':'t','started_at':'q','steps':[{'kind':'llm_call','started_ms':0,'tokens_in':{'a':1}}]})"
# also: started_ms = Infinity -> 500 ; 1e400 -> 500 ; 2**63 -> 500
```

**Observed:** `sqlite3.InterfaceError: Error binding parameter 9` / `OverflowError: Python int too large to convert to SQLite INTEGER`, escaping to a 500. Consequence: import a good trace (200, 1 step, FTS hit) → re-import the same id with one bad token field → 500 → **the stored trace now has 0 steps and its FTS entry is gone, permanently**. A fresh id commits a phantom row that `GET /api/traces` lists as a healthy empty trace. A 2-step trace whose step 1 is bad commits step 0 and lists as complete.

**Fix applied:** added an `isinstance` guard (bool-first, since `isinstance(True, int)` is True) for `tokens_in/out/cache_read/cache_write/reasoning` in the pre-lock loop, and range/finiteness checking (`OverflowError` alongside the existing `TypeError`/`ValueError`, plus an explicit `-2**63 <= v <= 2**63-1` bound) inside `_ms`. Both now refuse at 422 before the transaction opens. Verified: all six malformed inputs refuse, the pre-existing trace stays intact and searchable afterward. Six new tests in `tests/test_trace_search.py`.

---

### 1.2 `/api/image/unload` and a second `/api/image/load` block forever, and starve the process executor

**Root cause:** `ImageHandle.load` (`image_runtime.py:420`) holds `self._lock` across identify + prefetch (a child process downloading gigabytes) + resolve + opcode scan + load. `unload()` (`:548`) waits on the same lock with **no timeout**. There is no image-side equivalent of `ModelRuntime._load_slot`, which already solves this on the text side.

**Repro:**
```python
h = app.state.image
threading.Thread(target=lambda: (h._lock.acquire(), held.set(), release.wait(25), h._lock.release()), daemon=True).start()
held.wait(5)
# each on its own daemon thread, 8s ceiling:
c.post("/api/image/unload"); c.post("/api/image/load", json={"repo": "stabilityai/sd-turbo"})
```

**Observed:** both TIMEOUT at 8s with no answer; `unload` only returns 200 once the lock releases (measured 8.74s for an 8.33s hold). The **text side under the identical setup returns 409 in 2.04s with a sentence.** Escalation: `/api/image/unload` is `async def` + `asyncio.to_thread` → the asyncio default executor (`min(32, cpu+4)` = 28 here). 28 blocked unload clicks starved `/api/model/unload` entirely — its own working 2s refusal never got a thread. On a 4-core box that's 8 clicks.

**Fix:** give `ImageHandle` an `acquire(timeout=LOAD_QUEUE_WAIT_S)` slot mirroring `runtime.py:795-830`, add `Refusal` arms to both routes naming the in-flight load and pointing at `/api/image/cancel` (already exists). `image_runtime.py:965-972` already documents the identical text-side incident being fixed with an etag timeout; the image side never got the analogous ceiling.

---

### 1.3 Repo-id validation is homemade and inconsistent — one path crashes, one waves traversal through

**Root cause:** three resolvers each roll their own id check. `huggingface_hub.utils.validate_repo_id` is already a hard dependency with **zero call sites** in the package.

| # | Surface | Input | Result |
|---|---|---|---|
| a | `vla_data.snapshot_path` `vla_data.py:115` | `"pusht"` | `ValueError: not enough values to unpack` → **HTTP 500** / CLI traceback rc=1 |
| b | same line, via `cli.py:319` | `modelmri audit` (no args) | `AttributeError: 'NoneType' has no attribute 'split'` |
| c | `vla._snapshot` `vla.py:227` | `"../../etc/passwd"`, `"/etc/passwd"`, `"a/b/c"` | 409 "is not cached. Download it first (`huggingface-cli download ../../etc/passwd`)" — a command that cannot run |
| d | `snapshot_path` path build | `"a/../../../etc/passwd"` | resolves outside the hub root (only `is_dir()`+glob today, so nothing leaks — but unguarded) |

**Why not a judgement call:** `vla.py:227-238` carries the exact fix comment for this crash class — *"it used to crash on the unpacking below with 'not enough values to unpack' — a message about this function's internals rather than about what was typed"* — and `vla_data.py:115` is the copy that never received it.

**Fix:** one shared `_validate_repo_id(raw) -> str` calling `huggingface_hub.utils.validate_repo_id`, raising the sentence `vla.py` already writes. Call it at the top of `snapshot_path` and in place of `vla.py:227`'s `"/" not in repo` test. Separately, `cli.py:319`: `discover(repo_id=repo_id or None)`'s `or None` defeats the `repo_id: str = DEFAULT_DATASET` default that exists for exactly the no-arg case — call `discover()` plainly when `repo_id` is empty.

---

### 1.4 `serve --port -1` spends 18s, prints a false "serving on" banner, then dumps a 55-line traceback

**Surface:** `cli.py:1439` (`--port`, plain `type=int`), banner at `:1787`, `uvicorn.run` at `:1795`.

**Repro:** `$MM serve --port -1`

**Observed:** rc=1 after 17.4s. stdout prints `ModelMRI 0.11.0 serving on http://127.0.0.1:-1` **before** anything binds. stderr: `OverflowError: bind(): port must be 0-65535` + a chained `CancelledError`, no clean shutdown. Wasted-time breakdown: `doctor.check()` 5.07s + `import modelmri.server` 15.34s + `create_app()` 0.45s, all before the bind attempt. `--port abc` refuses cleanly in 0.44s (rc=2) — inconsistent exit codes (rc=2 non-int, rc=1 out-of-range, rc=3 occupied port).

**Fix:** an explicit 0-65535 check on `--port` at parse time; move the banner after the socket actually binds. The `open` path already does this right at `cli.py:1005-1009`.

---

### 1.5 `modelmri uninstall` holds the terminal ~124s with zero output before its prompt

**Surface:** `cli.py:1023` `_tree_bytes`, called at `:1289` and `:1300`.

**Repro:** `$MM uninstall < /dev/null` — under a pipe, stdout is block-buffered so nothing appears at all for 2 minutes; even interactively the banner is delayed that long.

**Observed:** measured cost breakdown: data dir 56.1s (1.23 GB) + policy-venv 59.8s (1.03 GB) + hub 6.6s (126 GB) = 122.5s. **116 of those 124 seconds is the local dir, walked twice** — 38,635 of the data dir's 38,664 entries *are* the venv. Two independent causes: (i) `_tree_bytes` does a redundant second `stat` per entry (`is_symlink()` after `lstat`, when `lstat`'s own `st_mode` already answers it) — an `os.scandir` walk giving the byte-identical total takes 2.0s, 30x faster; (ii) the venv subtree is traversed twice. `--yes` pays the same two minutes because the block runs before the `if not yes:` gate.

**Fix:** print+flush the banner first; replace `rglob`+double-stat with `os.scandir`; compute the venv figure from the same traversal as `data`.

---

### 1.6 `MriStore.put()` is unlocked — concurrent completions with `mri:true` can 500 one that already succeeded

**Surface:** `openai_api.py:150-157`; store on `app.state.mri_store` (limit 8), used via `asyncio.to_thread` at `server.py:4875`/`:4931`.

**Repro:** hammer `store.put()` from 16 threads.

**Observed:** `next(iter(self._held))` races an insert; two threads reading the same `oldest` makes the loser's `del` raise `KeyError`. Both escape the route's `except (Refusal, BadRequest)` → 500, **after the completion was already generated and committed**. Measured ~4 failures per million puts — low rate, but the sibling store on the same `app.state` already documents this exact lesson (`traces.py:136`: *"RLock, not Lock… the 0.10 data race was an ABSENCE nobody noticed"*).

**Fix:** wrap `put`/`get`/`was_evicted` in a `threading.Lock`, or swap to `OrderedDict` + `popitem(last=False)` under it. Three lines.

---

### 1.7 Checkpoint readers raise raw Python exceptions instead of refusing

**Root cause:** every reader in `fit.py`/`image_fit.py` guards *some* malformed input and leaves the neighbor bare.

| # | Surface | Input | Escapes as | Reachable |
|---|---|---|---|---|
| a | `image_fit.py:884-886`, `count *= int(dim)` sits **outside** the try at `:861-877` | safetensors header `"shape": "abc"` or `[None,4]` | `ValueError`/`TypeError` — no except arm anywhere in `load` | **HTTP 500 on `POST /api/image/load`**, loopback-reachable |
| b | `fit.py:161-166`, `header.pop("__metadata__")` on whatever `json.loads` produced | header JSON is `[]`, `5`, `"abc"`, `true`, `null` | `TypeError`/`AttributeError` | library — escapes `weights_table`, `fit.weights_bytes`, `fit.plan` |
| c | `fit.py:443`/`:346`, `json.loads(config_path.read_text())` | half-written `config.json` (this project's own Drive cache produces exactly this) | `JSONDecodeError`/`AttributeError` | library |
| d | `fit.py:404`, `int(hidden) // n_heads` | `num_attention_heads: 0` | `ZeroDivisionError` | library |

**Worse than (d)'s crash:** with `head_dim` stated, `num_attention_heads: 0` does **not** crash — it silently returns `n_heads=0` and `kv_cache_bytes(...)` = **0 bytes at 4096 tokens**. Negatives sail through too: `num_hidden_layers: -5` → `kv_cache_bytes = -655360`, shrinking `plan()`'s total toward "it fits" — contradicting `fit.py:520`'s own stated promise of "never a negative or a fabricated minimum."

**Fix:** (a) move the `for dim in shape` loop inside the existing try. (b) guard non-dict headers before the `.pop()` — same pattern already used three other places in the codebase. (c) a guarded config-reading helper raising `BadRequest`. (d) make `need()` reject non-positive as it already rejects absent, and apply to `num_key_value_heads` too (its `or` currently swallows a stated 0). **(a) is the only one reachable today; b/c/d become Tier 1 the moment a route wires them, so fix all four together.**

---

# TIER 2 — wrong numbers in front of a user

### 2.1 MCP `Server.call` builds URLs by f-string and coerces with bare `int()`/`str()`/`or` — five distinct wrong-number/bad-refusal symptoms from ~30 lines

**Surface:** `mcp_server.py:252-282` (query building), `:386-388` (no type-check on `params`/`arguments`).

| symptom | input | observed |
|---|---|---|
| a. fragment silently drops a param | `rank_attention_heads {"baseline":"zero#"}`, no `layer` | attached server receives `?baseline=zero`; `&scope=all` is eaten as a URL fragment → route defaults `scope="layer"`, `target=0`. **A layer-0 ranking returned for a call whose schema says "omit for all."** `isError:false`. |
| b. parameter injection, injected value wins | `logit_lens {"top_k":5,"kind":"plain&top_k=999"}` | `?top_k=5&kind=plain&top_k=999`; last scalar wins → route records `top_k=999`. |
| c. validation made unreachable | `{"baseline":"zero#"}` | the server sees a *valid* `zero`, so `ablate.py`'s own "unknown baseline" refusal can never fire over `--attach`, and the payload reports `baseline:"zero"` as if that's what was asked. |
| d. bool/float silently coerced | `{"layer": true}` → `&layer=1`; `{"top_k": 4.9}` → `top_k=4` | `isError:false`. Defeats `runtime.py:1299`, which explicitly refuses a bool layer over HTTP — MCP launders it to `int` first. |
| e. explicit 0/"" collapse into defaults | `{"top_k": 0}` → `top_k=5`; `{"kind": ""}` → `plain` | `or` treats 0/""/False as absent — dead code for MCP that HTTP enforces correctly. `layer`/`position` in the *same function* correctly use `is None`. |
| f. wrong-typed arg → internal-error code | `{"top_k":"five"}` | `-32603` "ValueError inside ModelMRI… full error is in the terminal", which is **empty** — `serve()` never logs it. |
| g. non-object `arguments` → crash | `{"name":"inspect_mri","arguments":[1,2]}` | `AttributeError`, same -32603; tool-dependent (`status` with the same malformed frame succeeds, since it never touches `args`). |

**Fix, one pass:** `urllib.parse.urlencode` instead of f-strings (kills a, b); a shared `_arg_int` helper with `is None`→default, `isinstance(v, bool)`→refuse, non-integral float→refuse (kills d, e, half of f); `enum` on `baseline`/`kind` in the tool schemas (kills c); `isinstance(params, dict)` / `isinstance(arguments, dict)` checks in `handle()` (kills g); widen the outer `except` to catch `(ValueError, TypeError)` as backstop. **`tests/test_mcp_server.py:427-444` is itself blind to (a)** — it asserts on the raw path string where `"scope=all" in path` is True even though the URL fragment means it's never actually sent; needs to assert on `urlsplit(url).query` instead.

---

### 2.2 DirectPanel renders "434 of 40 components" — an impossible fraction, in the shipped **public demo**

**Surface:** `frontend/src/DirectPanel.tsx:139-140`; type `api.ts:4113-4130`; server `dla.py:406-418`.

**Observed:** the server counts `n_components`/`n_unreadable` over the whole decomposition *before* the deliberate `top_k` cut, but the panel divides by `data.components.length` — the post-cut slice. Demo prints *"Direct path only. 434 of 40 components fall below that residual."* `n_components` isn't declared in the TS interface at all, so the correct denominator isn't even typed-reachable — and the server's own explanatory sentence (`means`) is never rendered by the panel.

**Fix:** add `n_components: number` to the type, use it as the denominator, render `data.means`. Three lines.

---

### 2.3 Two concurrent Ollama pulls mix bytes under one name, and the short one marks the long one "ready"

**Surface:** `progress.py:367-391` (`publish`, guards only on `self._snap.active`), `:421-447` (`finish`, guards on nothing). Nothing serializes `POST /api/ollama/pull`.

**Observed (through the real route):** a 1 GB pull and a 200 MB pull started 0.15s apart. The 1 GB pull — having transferred **zero bytes** — gets reported to the client as `stage: 'ready', bytes_done: 200000000` (the small pull's numbers). In the other interleaving, the progress meter alternates between the two jobs under one name, and once the short pull's `finish()` runs, the long pull's remaining `publish()` calls are silently dropped (`if not self._snap.active: return`) — the bar freezes at "ready" with ~800 MB still to go. This is the exact cross-job bug the module's own docstring documents having fixed once already — recurring through `publish`/`finish` instead of the `_publish` method that already carries the generation guard.

**Fix:** have `start`/`start_external` return a per-job generation token; check it in `publish()` and `finish()` the same way `_publish` already does.

---

### 2.4 The weight-scan summary sentence contradicts the payload it summarizes — three copies of one bug

**Surface:** `server.py:888-938` `_scan_summary`; stale duplicate at `cli.py:1233-1237`.

**Observed:** `_scan_summary`'s `read == 0` branch (`:917-923`) `return`s **before** the `capped` clause is built at `:924-931`. So `POST /api/weights/scan {"path":"modelmri","limit":1}` (the default `limit` is 200; any tree over that hits this) returns `n_found: 86, reports: 1, truncated: true` with the sentence *"NONE of the **1** file(s) here could be looked inside"* — contradicting its own `n_found`/`truncated` fields in the same payload. A nonexistent path gets **200**, not the 422 the sibling `/api/custom/scan` gives the identical body. The CLI copy at `cli.py:1234` independently reproduces the self-contradiction the server-side comment says it already fixed once (`server.py:916`), and discards `ScanTree.n_total`/`.truncated` on `--limit`.

**Fix:** move the `capped` clause inside the `read == 0` branch; have that branch defer to `reports[0].reason` when present; delete the stale CLI copy and call the shared summary function. **Do not** touch rc=0-for-unscanned — that's a documented, deliberate choice; only the summary sentence is wrong.

---

### 2.5 `GET /api/hub/models` publishes `downloads: 0, likes: 0` when the Hub is unreachable — indistinguishable from a real zero

**Surface:** `server.py:1248-1261` → `hub.py:538-551`.

**Observed:** with the Hub down, `GET /api/hub/models` returns 200 with `downloads: 0, likes: 0, updated: ""` — byte-identical to a genuinely popular-but-zero-downloads repo. The sibling `?q=` search on the same route correctly refuses 409 for the identical outage. The same dict already distinguishes unknown from zero for `params`/`size_gb` (both `null` on failure) but not for these three fields. The rule is already written and tested in a sibling module (`image_catalog.py:620`).

**Fix:** `downloads`/`likes` → `int | None`, `updated` → `str | None` in the fallback path; widen the TS type to match.

---

### 2.6 `GET /api/vla/actions/cost?stride=-1` prices 161 forward passes instead of 54, and reports `stride: 1` as if that's what was asked

**Surface:** `vla_actions.py:583`; route `server.py:3625` (`stride: int = 0`, no bound).

**Observed:** default → `stride=3, passes=54`. `stride=-1` → `stride=1, passes=161` — **2.98x the priced cost**, with the response claiming `stride: 1` rather than saying the input was rejected. `-1` is truthy, so `max(1,int(stride)) if stride else …` takes the wrong branch. Three sibling routes correctly refuse the identical negative input with 422.

**Fix:** refuse `stride < 0` with `BadRequest` (the route already has the except arm wired), matching the pattern the neighboring preflight route already documents fixing for the identical bug shape.

---

### 2.7 `GET /api/image/search?limit=0` reports `limit_asked: 24` — the payload states a number the caller never sent

**Surface:** `image_catalog.py:179`/`:283` (`asked = int(limit or 24)`); notice at `server.py:2492`.

**Observed:** `limit=0` → `limit_asked: 24` (the *default*, not what was sent) because `or 24` rewrites 0 before it's recorded. For `limit=-1`, the values are honest (`asked=-1, used=1`) but the cap-notice condition `limit_asked > limit_used` is False for `-1 > 1`, so the clamp goes unmentioned there too.

**Fix:** `asked = int(limit)`, handle the default substitution separately; change the notice condition to `limit_asked != limit_used`.

---

# TIER 3 — silent failures

### 3.1 Routes annotated `body: dict` / raw `Request` bypass all request-model discipline — and the guard meant to catch this is structurally blind to them

**Root cause:** 13 POST routes are annotated `body: dict`, so `Body(extra="forbid")` never applies to them. `tests/test_unknown_body_keys.py::test_every_request_model_refuses_a_key_it_does_not_know` resolves endpoint annotations against the server module — `dict` resolves to nothing, so the test sees 33 clean pydantic models and is **blind to all 13** raw-dict routes. Affected (11 of 13 should have model discipline; `/v1/chat/completions` and `/v1/completions` legitimately need OpenAI-compat tolerance): `/api/custom/unload`, `/api/hub/signout`, `/api/image/cancel`, `/api/image/unload`, `/api/judge`, `/api/judge/plan`, `/api/model/cancel`, `/api/rubric`, `/api/rubric/score`, `/api/session/close`, `/api/traces/import`.

| symptom | repro | observed |
|---|---|---|
| a. unknown key silently dropped | `POST /api/rubric/score {"rulez": [...]}` (typo for `"rules"`) | 200, `counts: {}`, means: *"0 rule(s) against 111 recorded run(s)… **No run matched any rule.**"* — where `{"rules": …}` gives `counts: {'any_error': 64}`. **64 of 111 runs actually have errors; one typo produces a confident all-clear.** Sharper on the sibling: `/api/judge/plan {"n_paraphrases":5}` → 422 naming the cap; `{"n_paraphrase":5}` (typo) → 200 with a silently substituted default. Contrast: `POST /api/patch` with an unknown key → 422 `extra_forbidden`, correctly. |
| b. malformed JSON swallowed | `POST /api/custom/ablate` with non-JSON bytes | 422 *"no custom model is loaded"* — true statement, wrong reason; the body was never parsed. `POST /api/vla/sweep` with the same non-JSON bytes → **200 in 13.5s**, a full default sweep run off a body that was never read. |
| c. absent field collapses to `""`, then compares equal | `POST /api/diff/models {"prompts":[...]}`, no `a`/`b` | 422 *"both sides are the same model, so every difference would be zero by construction"* — for a request that named **no model at all**. And the asymmetric case (`{"prompts":[...], "a":"x"}`, no `b`) is **500**, not a refusal. |

**Fix:** give each non-`/v1` raw-dict route a proper `Body`-derived request model; delete the `body.get("rules", body)` fallback that lets the whole request parse as a rubric; replace `except Exception: body = {}` sites with the same 422 the other 8 similar sites already use. Then **extend the test** to fail on any POST route annotated `dict`/`Request` outside an explicit allowlist, or this recurs on the next raw-dict route.

---

### 3.2 A machine with `pyarrow` and no `av` gets a completed measurement instead of a refusal, and a UI full of controls that cannot work

**Surface:** the "no video decoder" state is refused correctly by 3 routes and invisible everywhere else, because per-frame `except Exception` swallows the underlying `ImportError`.

**Observed:** `POST /api/vla/sweep {"frame_stride": 10**12}` (or a plain `{}`) → **200**, `rows: []`, every entry's `why: "ModuleNotFoundError"` — silently degrading instead of refusing. The *same route* with `{"metric":"occlusion_peak"}` correctly 409s, because that one path happens to call the decoder outside the per-frame try. Downstream: `GET /api/vla/episodes` returns 200 with no readiness field at all, so the frontend swaps in a full panel — a 200+ entry episode picker, a frame scrubber, a "load vision tower" button — none of which can ever produce a picture, since every actual frame request 409s. The refusal *is* shown, but at the bottom of the panel, below four other sub-panels that shouldn't have rendered at all. The TS field shaped for exactly this (`VLAStatus.mode: "unavailable"|"data"|"perception"|"full"`) has `"data"` **never assigned anywhere** in the codebase.

**Fix:** let the `ImportError` propagate out of the per-frame handler (or re-raise once every sampled frame fails identically) so the existing 409 path fires; add `frames_readable: boolean` + `reason` to `/api/vla/episodes` and gate the frontend's panel swap on it, mirroring the contract the image panel already implements correctly.

---

### 3.3 An in-flight custom run writes its results onto whatever model is loaded when it finishes

**Surface:** `custom.py:939-985`. `CustomHandle.run()` takes the lock, copies `model`/`example`, **releases** the lock, runs the (possibly slow) forward pass, then re-takes the lock and writes results onto `self.status_` — not the object it captured before releasing.

**Repro:** load a slow-forward adapter, start a run, then unload (or load a different model) while the run is in flight.

**Observed (deterministic, 3/3):** after unload mid-run: the "loaded: false" status reports the departed model's `input_shape`/`input_origin` as if it were still live, and `handle.rows`/`handle.meta` — which tests assert must be empty after unload — get repopulated *after* the unload completes. After loading a different model mid-run: the new model's status carries the *previous* model's input shape, with "the shape you entered" attached to a shape nobody entered for it — and the next real run against it 422s on a shape mismatch that has nothing to do with what the user actually typed.

**Fix:** in the second critical section, only write when `self.status_ is status` (the object already captured at the top of the function, before the lock was released). Same guard for `self.rows`/`self.meta`.

---

### 3.4 One adapter import breaks a concurrent one, and the refusal blames the user's file

**Surface:** `custom.py:292-333`, `_import_adapter` mutates process-wide `sys.path` with an `added = parent not in sys.path` check that races.

**Repro:** two adapter files in the same folder, loaded concurrently, one importing a sibling module the other's top-level code needs.

**Observed:** the second load sees the path entry already present (from the first), sets `added=False`, and the first load's `finally` removes the directory out from under the second load's still-running import — `ModuleNotFoundError` for a module that genuinely exists right there. The error message tells the reader their file is broken; it cannot distinguish that from "ModelMRI removed your import path mid-import." Control test confirms it's the shared path entry, not concurrency per se: same two files in different folders, both succeed.

**Fix:** a module-level import lock around the insert/exec/remove sequence, or reference-count the path entry instead of a bare boolean.

---

### 3.5 An empty path field resolves to the server's own working directory — and one of them silently loads it

**Surface:** `server.py:307-315` (`GgufLoad.path`), `:318-323` (`QuantCompare`); resolver `custom.py:279-286`.

**Observed:** `POST /api/gguf/load {"path":""}` → 409 naming a directory the request never mentioned (the server's own cwd), with no next step — `Path("").resolve()` is the cwd, which passes `exists()` and fails `is_file()`. **Worse:** `POST /api/quantdiff/behaviour {"quantised":"<real .gguf>", "original":""}` is **not refused at all** — the empty `original` field resolves to the cwd, which *is* an allowed root, and the empty field is handed straight to the loader as "the full-precision checkpoint" — a route documented as expensive and destructive, proceeding on the server's own source tree.

**Fix:** `min_length=1` on the three affected fields — the same guard `hf_id` already carries elsewhere in this file for the identical bug class.

---

### 3.6 `GET /api/vla/audit` truncates the distinct-frames failure list to 4 with no true count — self-contradicting payload

**Surface:** `vla_audit.py:349-353`.

**Observed:** `episodes_sampled: 6`, `failed: [4 entries]`, `distinct_images: 0`. A reader counting the list sees 4/6 failed and concludes 2 succeeded; `distinct_images: 0` says none did. Every *other* capped list in this same response (`n_gaps`, `n_missing`, `n_drifted`, `n_pairs`+`pairs_complete`) correctly carries its true count; these two are the only ones that don't, and the frontend's generic cap-detection logic (`c.measured['n_'+key]`) silently degrades to "not capped" for exactly these two fields.

**Fix:** add `n_failed`/`n_collisions` alongside the truncated lists. One line each.

---

### 3.7 MCP `status` reports `loaded: false` while an image pipeline and a VLA are resident

**Surface:** `mcp_server.py:224-239`.

**Observed:** `/api/session` correctly shows a 3.3 GB SDXL pipeline and a VLA both loaded; the MCP `status` tool — whose description says "what model is loaded on this machine" — reports `loaded: false, hf_id: null`, because the adapter only lifts the text-model status and discards image/VLA. **Confirmed NOT a bug in the in-process/attach equality** (that's a tested, deliberate decision) — it's that the tool's description overpromises what it actually covers, and there are no image/VLA tools on the MCP surface for an agent to act on that info anyway.

**Fix:** narrow the tool description to the text model only, rather than widening the payload.

---

### 3.8 `python -m modelmri.cli <anything>` exits 0 having done nothing

**Surface:** no `if __name__ == "__main__"` guard, no `modelmri/__main__.py`.

**Observed:** `python -m modelmri.cli doctor --help` → rc=0, zero bytes of output. Silently does nothing rather than either running or erroring. Severity honestly low — this entry point is documented nowhere and the conventional guess (`python -m modelmri`) already fails loudly. **Bonus, found in passing:** `python -m modelmri_policy --help` **hangs** — it calls `server.main()` without parsing `--help` at all.

**Fix:** add `modelmri/__main__.py` and the guard; fix the sidecar's `--help` hang while there.

---

### 3.9 The demo bundle answers two routes with the wrong shape, and one of them gives a demo visitor wrong instructions

**Root cause:** `tests/demo_check.py` gates path coverage, never payload keys. `json<T>(r)` in `api.ts` is a bare type cast — nothing checks the shape at runtime either.

| route | real | demo |
|---|---|---|
| `/api/paths` | 15 keys | 9 — missing 6 keys the TS type declares required, plus `demo_note` (the sentence explaining the placeholder values) is typed nowhere and never rendered, so a visitor sees 7 unexplained placeholder rows |
| `/api/ollama` | `{host, installed, models, suggested, up}` | `{up, models, reason}` — `reason` is declared and read nowhere, so the handler's own explanatory sentence never renders and the frontend falls back to *"Ollama is not running. Install it from ollama.com…"* — a refusal naming a cause that is not the cause, on a static recording where neither suggested next step can change anything |

**Fix:** synthesize the missing keys in the demo baker (do not bake real machine values — the leak scanner exists precisely to catch that); add `reason?: string` to the Ollama type and render it when present; add a key-parity assertion to `demo_check.py`.

---

# TIER 4 — messages and types only

### 4.1 `_way_out()` invents a diagnosis from a blank snapshot and tells the reader to restart the server

**Surface:** `runtime.py:753-796`, called from `_load_slot` at `:826-827`.

**Observed:** hitting the load-slot contention window (structural, not rare: `unload()` holds the slot for its *entire* teardown — epoch bump, GC, cache clear — without ever starting the progress tracker, so `active` is False for that whole window) produces: *"…the weights have finished arriving and this load is now inside transformers, which offers no way to interrupt it… restarting the server is the way out."* **No weights had ever arrived; nothing had ever loaded.** `_way_out()` never checks `snap.active` at all — the one check `_in_flight()` (right next to it) correctly makes. Measured over 90s of alternating load/unload requests: 1 round in 4 got this false diagnosis.

**Fix:** guard `_way_out()` on `snap.active`; have `_in_flight()` and `_way_out()` share one snapshot so the two halves of a refusal describe the same instant. The existing test helper hardcodes `active=True`, so the false branch has zero coverage today.

---

### 4.2 "No model loaded" and "generate something first" are collapsed into one sentence on six routes

**Surface:** `runtime.py:1819` `_require_live_generation`, `:1672` `_ready_for_attention` — both test `if not self.loaded or self.last_ids is None`.

**Observed:** with literally nothing loaded, six routes (`/api/attention/ablate`, `/api/attention/baselines`, `/api/attention?layer=`, `/api/attention/attribute`, `/api/features/ablate`, `/api/session/export`) all say *"Generate something first…"* — telling the user to press a button that will itself just 409 with "no model loaded." Three sibling routes (`/api/attention/direct`, `/api/attention/meta`, `/api/lens`) correctly split the two states into different sentences.

**Fix:** apply the same split those three already use — the decision rationale is written directly above the working version (*"they want opposite things from the reader"*). Also fix a test that currently asserts the wrong state green by using a bare no-model client where a loaded-but-ungenerated client was needed.

---

### 4.3 Three sentences for one state, one a bare fragment; two state errors returned as 422 instead of 409

**Surface:** `server.py:1482`/`:5959`, `runtime.py:4290`, `custom.py:948`/`:1005`.

**Observed:** 4 different phrasings of "no model loaded" across the codebase, two of which return a raw `JSONResponse` bypassing `Refusal` entirely and putting the *machine-readable status reason* (deliberately lowercase, pinned by a test) directly into a human-facing refusal slot. The Generate button in the Playground is not gated on `model?.loaded`, so this is reachable by an ordinary click, not just direct API use.

**Fix:** standardize the two raw-`JSONResponse` sites on the sentence the other 10 sites already use. **The 422-vs-409 question on `/api/custom/run` is a documented, tested contract decision, not a bug** — leave it, or change it only together with the test that locks it and a written rationale.

---

### 4.4 Contract drift — four routes send fields the client's TypeScript does not declare

| route | undeclared fields | type |
|---|---|---|
| `GET /api/image/filmstrip/cost` | 9 of 16 fields (mostly always-`null` today — the route just doesn't compute them yet, so declare as `\| null` rather than "fixing" the route) |
| `POST /api/vla/sweep` | `dataset, policy, camera`, `strip.metric`, `strip.unit` |
| `GET /api/vla` | 5 fields; 5 geometry ints come back as `0` when nothing is loaded, while sibling fields `repo`/`warmup_ms` correctly use `null` in the identical situation |
| `GET /api/attention/direct` | `n_components` — this is the missing denominator from **2.2** |

Also found in passing: `VLAStatus.dataset` is declared and typed but **never assigned anywhere** in the codebase — always serializes `null`; `Sweep.policy` collapses `str | None` into `""` while the sibling status route correctly uses `null` for the same fact.

**Fix:** declare fields at their real nullability; add one test that diffs live response keys against each route's `api.ts` interface for every route that has one — this single test would have caught 4.4, 2.2's root cause, and 3.9's demo drift all at once.

---

# Root-cause groups (13 patterns behind the 44 defects)

1. Repo ids validated by hand, three different ways, none using the already-installed `validate_repo_id` → **1.3**
2. Validation that runs after the write has started → **1.1**
3. No load-slot contract on the image side; a lying one (`_way_out`) on the text side → **1.2, 4.1**
4. Unlocked shared state: `MriStore.put`, `progress.publish`/`finish` (no generation guard), `CustomHandle.run` (writes to the live pointer, not the captured object), `_import_adapter` (process-wide `sys.path`) → **1.6, 2.3, 3.3, 3.4**
5. Checkpoint readers raise where they should refuse — one guarded case, one bare, repeated across four functions → **1.7**
6. MCP arguments interpolated and coerced, never validated against the schema the tool advertises → **2.1**
7. Raw-dict/raw-`Request` routes have no request-model discipline, and the guard meant to catch it is structurally blind to them → **3.1**
8. Two copies of the scan summary, `read == 0` returns before the cap clause is built → **2.4**
9. Unknown collapsed into 0/""/a default without reporting the substitution → **2.5, 2.6, 2.7, 2.1e, 4.4**
10. A capped list published without its true length → **3.6, 2.2, 2.4c**
11. A missing optional dependency refused correctly on 3 routes, invisible everywhere else → **3.2**
12. Server payload and `api.ts` drift, nothing compares them (same failure mode as the demo bundle vs. real routes) → **4.4, 3.9**
13. Two different states sharing one sentence when they need opposite next steps → **4.2, 4.3**

---

# Suggested order

## Fix now (7) — real user impact, cheap or already scoped
- ~~**1.1** traces import — data loss.~~ **DONE** (`627857f`)
- ~~**1.2** image lock~~ **DONE** (`0f93582`) — 409 in 2.03s where it returned nothing
- ~~**1.3** repo-id validation~~ **DONE** (`3493d88`) — one shared `paths.validate_repo_id`
- ~~**2.1** MCP argument handling~~ **DONE** (`d6b7dcf`) — and the blind test now parses the query
- ~~**3.1** raw-dict routes~~ **DONE** (`1be2b5c`) — and the blind test now sees them
- ~~**2.2** DirectPanel~~ **DONE** (`36ad22b`) — reads "391 of 477", verified in the live DOM
- ~~**1.4** `serve --port`~~ **DONE** (`7e04e00`) — 17.4s+traceback -> 0.08s+sentence

## Next (9) — real, reproducible, narrower blast radius
~~2.3 progress gen guard~~ **DONE** (`9e08788`) · 2.4 scan summary · 3.2 missing-`av` readiness · ~~3.3 CustomStatus stamping~~ + ~~3.4 `sys.path` race~~ **DONE** (`8143c98`) · 3.5 empty path → cwd · 1.7a image_fit shape loop · 4.1 `_way_out` · 2.5 hub-offline zeros

## Can wait
1.5 uninstall timing (correct, just slow) · 1.6 MriStore (very low rate) · 1.7b/c/d (library-only until routed — but fix *before* routing, not after) · 2.6, 2.7 (single-parameter honesty, one line each) · 3.6, 3.7, 3.8, 3.9 · 4.2, 4.3, 4.4

## Explicitly NOT bugs — investigated and confirmed correct/deliberate
- `/api/vla/sweep`'s all-failed `means`/`n_failed`/`strip.low` reporting (correct, documented)
- rc=0 for unscanned files in `scan` (documented choice)
- MCP in-process/over-attach status equality (tested invariant — the fix is narrowing the tool description, not the equality)
- `/api/custom/run`'s 422-not-409 (locked by a test; a contract decision, change deliberately or not at all)

## Two tests to add alongside any of this, or it recurs
1. Extend `test_unknown_body_keys.py::_request_models()` to **fail** on any POST route annotated `dict`/`Request` outside an explicit `/v1` allowlist. Today it reports "33 clean" while blind to 13 routes.
2. A payload/type parity test: for every route with a declared `api.ts` interface, diff the live response keys against the declaration. One test, and it would have caught 4.4, 2.2, and 3.9 together.
