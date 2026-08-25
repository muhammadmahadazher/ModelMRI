"""Command-line interface: `modelmri serve`, `modelmri open`, `modelmri where`."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

from . import __version__


def compare_experiments(
    before,
    after,
    *,
    metric: str,
    higher_is_better: bool,
    dataset=None,
    floor: float | None = None,
    fail_on_worse: int = 0,
    as_json: bool = False,
) -> int:
    """Two runs of one dataset, case by case, as a CI gate.

    Pays for NO torch, like `diff` and unlike `verify`: both sides are already
    measured and the comparison is arithmetic over JSONL. That is the point —
    the roadmap wants this runnable on every pull request in milliseconds, on
    a machine with no accelerator.

    The exit code is the gate. Non-zero when more than `fail_on_worse` cases
    got worse, and the default is 0 because any regression should fail a gate
    — a threshold above zero is somebody deciding in advance how much breakage
    is acceptable, which is a decision to make out loud rather than by
    default.
    """
    from . import datasets
    from .errors import BadRequest, Refusal

    try:
        comparison = datasets.compare_experiments(
            datasets.read_experiment(before),
            datasets.read_experiment(after),
            metric=metric,
            higher_is_better=higher_is_better,
            dataset=datasets.read_dataset(dataset) if dataset else None,
            floor=floor,
        )
    except (BadRequest, Refusal) as err:
        # 2, not 1. A gate that cannot run is not a gate that passed, and it
        # must not be confused with one that ran and found regressions.
        print(f"modelmri experiments: {err}", file=sys.stderr)
        return 2

    if as_json:
        print(json.dumps(comparison.to_dict(), indent=2))
    else:
        print(datasets.render(comparison))

    worse = comparison.counts.get(datasets.WORSE, 0)
    if worse > fail_on_worse:
        print(
            f"\n{worse} case(s) got worse on {metric}, above the "
            f"{fail_on_worse} this gate allows.",
            file=sys.stderr,
        )
        return 1
    return 0


def diff_sessions(
    path_a, path_b, *, fail_over: float | None = None, as_json: bool = False
) -> int:
    """Compare two `.mri` and exit non-zero when something moved.

    Like `inspect` and unlike `verify`, this pays for NO torch: a `.mri` is
    gzipped JSON, both sides are already measured, and comparing them is
    arithmetic. That is what makes it usable as a CI step -- a job that has to
    install torch to check a regression is a job nobody adds.
    """

    from . import mri_diff
    from .errors import BadRequest, Refusal

    try:
        report = mri_diff.diff(path_a, path_b)
    except (BadRequest, Refusal) as err:
        print(f"modelmri: {err}", file=sys.stderr)
        return 2

    print(
        json.dumps(report.to_dict(), indent=2, allow_nan=False)
        if as_json
        else mri_diff.render(report, fail_over)
    )
    return report.exit_code(fail_over)


def resume_sweep(sweep_id: str, *, as_json: bool = False) -> int:
    """Finish a saved sweep, keeping every prompt already measured.

    `sweep.resume` was written, tested and PRICED — `resume_plan` answers what
    finishing would cost — and had no way to run from anywhere: no route, no
    flag, no button. A sweep that stopped could be listed as unfinished, priced
    cleanly, and never finished.

    The reachable case is not a crash, which leaves nothing saved because
    `save()` runs after `run()` returns. It is REFUSALS: `remaining()` counts
    every unmeasured row as still-to-run, so one prompt the model refused
    leaves a sweep permanently unfinished.

    Priced and checked before it runs, like everything else here — and it is
    `sweep.resume` that re-checks, so the price and the run cannot disagree.
    """
    from . import sweep as sweep_mod
    from .errors import BadRequest, Refusal
    from .runtime import ModelRuntime

    try:
        plan = sweep_mod.resume_plan(sweep_id)
    except (BadRequest, Refusal) as err:
        print(err.sentence, file=sys.stderr)
        return 2

    if plan["blocked"] is not None:
        # A sentence, never a warning to override. The three things it checks
        # make a resume WRONG rather than expensive.
        print(f"this sweep must not be resumed: {plan['blocked']}", file=sys.stderr)
        return 2
    if not plan["n_remaining"]:
        print(plan["means"])
        return 0

    print(plan["means"])
    print(f"  loading {plan['model']} to measure {plan['n_remaining']} prompt(s)…")

    runtime = ModelRuntime()
    try:
        runtime.load(plan["model"])
        job, rows = sweep_mod.resume(sweep_id, runtime)
    except (BadRequest, Refusal) as err:
        print(err.sentence, file=sys.stderr)
        return 2

    stats = sweep_mod.aggregate(rows, metric=job.metric)
    measured = sum(1 for r in rows if r.measured)
    if as_json:
        print(
            json.dumps(
                {
                    "sweep_id": sweep_id,
                    "model": job.model,
                    "metric": job.metric,
                    "n_prompts": len(rows),
                    "n_measured": measured,
                    "n_unmeasured": len(rows) - measured,
                    "stats": [st.to_dict() for st in stats],
                },
                indent=2,
            )
        )
    else:
        print(sweep_mod.render(job, rows, stats))
        if measured < len(rows):
            # A refusal stays a refusal, and it is why this sweep could sit
            # unfinished forever. Said rather than left to be inferred from a
            # count that did not reach the total.
            print(
                f"\n  {len(rows) - measured} prompt(s) still could not be "
                f"measured and are absent from the ranking rather than scored "
                f"zero. Resuming again will retry them."
            )
    return 0


def run_sweep(
    prompts_path,
    *,
    model: str,
    metric: str = "heads",
    baseline: str = "zero",
    layer: int | None = None,
    max_new_tokens: int = 8,
    out_dir=None,
    jsonl=None,
    yes: bool = False,
    as_json: bool = False,
) -> int:
    """Run one measurement over many prompts and report the distribution.

    Prints the projected pass count BEFORE anything runs. Cost is N prompts x
    per-prompt cost and the resample baseline multiplies through every row, so
    a sweep that is about to take an hour should say so while there is still
    time to press ctrl-c -- not after.
    """
    import datetime
    import uuid

    from . import sweep as sweep_mod
    from .errors import BadRequest, Refusal
    from .runtime import ModelRuntime

    try:
        prompts = sweep_mod.load_prompts(prompts_path)
        job = sweep_mod.Job(
            model=model,
            prompts=prompts,
            metric=metric,
            baseline=baseline,
            layer=layer,
            max_new_tokens=max_new_tokens,
            out_dir=Path(out_dir) if out_dir else None,
        ).validated()
    except (BadRequest, Refusal) as err:
        print(err, file=sys.stderr)
        return 2

    runtime = ModelRuntime()
    try:
        runtime.load(model, confirm=True)
        projection = sweep_mod.plan(job, runtime)
        print(
            f"{projection['prompts']} prompts x "
            f"{projection['passes_per_prompt']:,} passes = "
            f"{projection['passes_total']:,} forward passes",
            file=sys.stderr,
        )
        if not projection["aggregatable"]:
            print(
                f"  a {metric} sweep is per-prompt only: position 3 is a "
                f"different token in every prompt, so it is not aggregated",
                file=sys.stderr,
            )
        if projection["passes_total"] > 20_000 and not yes:
            # Refused rather than started. A sweep that cannot finish inside
            # anybody's patience is worse than one that never began, because
            # the half-finished one has already cost the time.
            print(
                "  that is a lot of passes. Re-run with --yes if you meant "
                "it, or narrow it with --layer.",
                file=sys.stderr,
            )
            return 2

        rows = sweep_mod.run(
            job,
            runtime,
            on_row=lambda row, total: print(
                f"  [{row.index + 1}/{total}] "
                + ("ok" if row.measured else f"refused: {row.could_not_measure}"),
                file=sys.stderr,
            ),
        )
    except (BadRequest, Refusal) as err:
        print(err, file=sys.stderr)
        return 2
    finally:
        try:
            runtime.unload()
        except Exception:  # noqa: S110 - the rows are already collected
            pass

    stats = (
        sweep_mod.aggregate(rows, metric=job.metric, top_k=job.top_k)
        if job.metric in sweep_mod.AGGREGATABLE
        else []
    )
    sweep_id = uuid.uuid4().hex[:12]
    sweep_mod.save(
        job,
        rows,
        started_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        sweep_id=sweep_id,
    )
    if jsonl:
        written = sweep_mod.write_jsonl(rows, jsonl)
        print(f"  rows written to {written}", file=sys.stderr)

    if as_json:
        print(
            json.dumps(
                {
                    "id": sweep_id,
                    "projection": projection,
                    "rows": [r.to_dict() for r in rows],
                    "stats": [s.to_dict() for s in stats],
                },
                indent=2,
                allow_nan=False,
            )
        )
    else:
        print(sweep_mod.render(job, rows, stats))
    # Every prompt refused is a failure of the run, not a finding.
    return 0 if any(r.measured for r in rows) else 1


def audit_dataset(repo_id: str = "", *, as_json: bool = False) -> int:
    """Prove a robot dataset is intact, or say exactly where it is not.

    Exit code is 1 when a check FAILED and 0 otherwise, so this drops into a
    pre-training script. A check that could not run here is NOT a failure --
    a missing PyAV is a fact about the machine, and exiting non-zero for it
    would make the gate about the environment rather than the data.
    """

    from . import vla_audit

    # Imported here rather than at module scope, like every other command in
    # this file: `cli.py` is what `modelmri --help` runs, and pulling errors
    # (and through it the rest of the package) at import time would put a
    # second of torch on the front of every invocation.
    from .errors import BadRequest, Refusal
    from .vla_data import LeRobotV3Reader

    try:
        # `discover()` with NOTHING when no dataset was named, rather than
        # `repo_id=repo_id or None`. That `or None` passed None EXPLICITLY,
        # which overrode the `repo_id: str = DEFAULT_DATASET` default that
        # exists for exactly the no-argument case — so `modelmri audit` on its
        # own reached `None.split("/")` and printed an AttributeError about
        # this program's internals. The default is the whole point of the
        # parameter; passing None defeats it.
        reader = (
            LeRobotV3Reader.discover(repo_id=repo_id)
            if repo_id
            else LeRobotV3Reader.discover()
        )
    except ImportError as err:
        # `err.name` and never `err` — the same rule `_missing_reader_dep`
        # already follows on the HTTP side. The module name is the useful half
        # and is bounded; the exception's own prose is free-form text about
        # this machine, and a leak test walks every sink in the package
        # looking for exactly that interpolation. Caught by it, not by review.
        what = f" ({err.name} is missing)" if getattr(err, "name", None) else ""
        print(f"Reading a LeRobot dataset needs pyarrow and av{what}.")
        print("  pip install 'modelmri[vla-lite]'")
        return 2
    except (Refusal, BadRequest) as err:
        print(err)
        return 2

    report = vla_audit.audit(reader)
    if as_json:
        print(json.dumps(report.to_dict(), indent=2, allow_nan=False))
        return 1 if report.broken else 0

    mark = {vla_audit.OK: "OK  ", vla_audit.BROKEN: "FAIL", vla_audit.UNCHECKED: "--  "}
    print(
        f"{report.repo_id} — {report.n_episodes} episodes, "
        f"{report.n_frames} frames, {report.seconds}s"
    )
    print()
    for check in report.checks:
        print(f"  {mark[check.verdict]} {check.name}")
        for line in _wrap(check.detail, 72):
            print(f"         {line}")
    print()
    for line in _wrap(report.means(), 76):
        print(f"  {line}")
    return 1 if report.broken else 0


def _port(raw: str) -> int:
    """A TCP port, refused at PARSE time rather than at bind time.

    `type=int` accepted anything an int can hold, so `--port -1` got all the
    way to the socket: `doctor.check()` 5.1s, `import modelmri.server` 15.3s,
    `create_app()` 0.4s — and only then `OverflowError: bind(): port must be
    0-65535`, as a traceback with a chained CancelledError and no clean
    shutdown. Seventeen seconds to reject a number argparse rejects in a
    millisecond.

    An `ArgumentTypeError` puts it on the same footing as `--port abc`, which
    already refused cleanly in 0.44s. The two answers were seventeen seconds
    and two exit codes apart for the same kind of mistake.
    """
    try:
        port = int(raw)
    except (TypeError, ValueError):
        raise argparse.ArgumentTypeError(
            f"{raw!r} is not a port number — a whole number from 0 to 65535."
        ) from None
    if not 0 <= port <= 65535:
        raise argparse.ArgumentTypeError(
            f"{port} is not a usable port. TCP ports run 0 to 65535, and "
            f"ModelMRI's default is 5900."
        )
    return port


def _wrap(text: str, width: int) -> list[str]:
    import textwrap

    return textwrap.wrap(text, width=width) or [""]


def check_trace(args) -> int:
    """`modelmri check` — structural assertions, for a CI step.

    Imports nothing heavy: no torch, no transformers, no network. The point is
    that this runs in a build container with no account, so anything that
    would need one is not allowed in this path.
    """
    from . import check as check_mod

    result, code, error = check_mod.check(
        args.target,
        no_errors=args.no_errors,
        max_steps=args.max_steps,
        no_retry_storms=args.no_retry_storms,
        no_loops=args.no_loops,
        max_repeat=args.max_repeat,
        max_ms=args.max_ms,
    )
    if result is None:
        # Unreadable is its own exit code: a missing file is a broken
        # pipeline, not a broken agent.
        if args.json:
            print(json.dumps({"error": error, "ok": False}, indent=2))
        else:
            print(error, file=sys.stderr)
        return code
    print(json.dumps(result.to_dict(), indent=2) if args.json else result.report())
    return code


def verify_session(path, *, as_json: bool = False) -> int:
    """Re-run a `.mri`'s measurements here and report what came back the same.

    Unlike `inspect` and `open`, this one DOES pay for torch and transformers,
    and there is no way around it: verifying a measurement means taking the
    measurement, through the same `ModelRuntime` methods the server calls. A
    second implementation would be verifying itself.

    Exit 1 only for a real disagreement. A file this machine cannot check is
    not a failure of the file, and exiting non-zero for it would make `verify`
    useless in CI the moment somebody ran it on a different accelerator.
    """

    from . import verify as verify_mod
    from .errors import BadRequest, Refusal
    from .runtime import ModelRuntime

    runtime = ModelRuntime()
    try:
        report = verify_mod.verify(path, runtime)
    except (BadRequest, Refusal) as err:
        print(err, file=sys.stderr)
        return 2
    finally:
        # The model is several gigabytes and this is a one-shot command, so it
        # goes back before the process does rather than at interpreter exit.
        try:
            runtime.unload()
        except Exception:  # noqa: S110 - the report is already printed
            # The process is about to exit and the OS reclaims the memory
            # regardless. Turning a tidy-up failure into a non-zero exit would
            # report a verification failure that did not happen.
            pass

    print(
        json.dumps(report.to_dict(), indent=2) if as_json else verify_mod.render(report)
    )
    return report.exit_code()


def inspect_session(path, *, as_json: bool = False) -> int:
    """Describe a `.mri` on the terminal. Returns the exit code.

    Same discipline as `open`: no torch, no transformers, no server. A `.mri`
    is gzipped JSON and everything below comes from the standard library plus
    session.py, so this stays instant on a cold cache — the reason `open` was
    rewritten in the first place was that 26 seconds of imports to read a
    54 KB file reads as a hang, and somebody pressed ctrl-c.
    """
    from pathlib import Path

    from . import session

    target = Path(path).expanduser()
    if not target.is_file():
        print(f"modelmri: no such file: {target}", file=sys.stderr)
        return 2
    try:
        parsed = session.parse(target.read_bytes())
    except session.SessionError as err:
        print(f"modelmri: {err}", file=sys.stderr)
        return 2

    meta = parsed.meta
    slices = sorted(
        (int(k.split(":")[0]), int(k.split(":")[1]))
        for k in parsed.attention
        if k.count(":") == 1 and all(part.isdigit() for part in k.split(":"))
    )
    summary = {
        "file": target.name,
        "bytes": target.stat().st_size,
        "model": meta.get("model") or "unknown",
        "device": meta.get("device"),
        "dtype": meta.get("dtype"),
        "n_params": meta.get("n_params"),
        "created_at": meta.get("created_at"),
        "modelmri": meta.get("modelmri"),
        "note": (meta.get("note") or "").strip(),
        "scope": meta.get("scope") or "",
        "precision": meta.get("precision") or "",
        "n_tokens": len(parsed.tokens),
        "n_prompt": parsed.n_prompt,
        "n_layers": parsed.n_layers,
        "n_heads": parsed.n_heads,
        "attention_maps": len(parsed.attention),
        "layers_present": sorted({li for li, _ in slices}),
        "heads_present": sorted({hi for _, hi in slices}),
        "lens_rows": len(parsed.lens),
        # A comparison of two models, which need NOT be this file's model.
        # `session._model_diff` requires both names for that reason, so both
        # are printed: a diff read as being about the model named at the top
        # of this output is the one confusion this section can cause.
        "model_diff": (
            {
                "present": True,
                "a": parsed.model_diff.get("model_a", ""),
                "b": parsed.model_diff.get("model_b", ""),
                "n_prompts": parsed.model_diff.get("n_prompts"),
                # The spread, never the median alone -- that is the whole
                # content of this section.
                "kl": parsed.model_diff.get("kl") or {},
                # `None` is a RESULT: the cosine never fell, so these two
                # checkpoints do not part company anywhere in particular.
                "consensus_layer": parsed.model_diff.get("consensus_layer"),
                "consensus_share": parsed.model_diff.get("consensus_share"),
                "n_heads_moved": len(parsed.model_diff.get("heads") or []),
            }
            if parsed.has_model_diff()
            else {"present": False}
        ),
        # Head labels. `/api/attention/types` already serves these from a
        # recording -- it is `inspect` that said nothing, so a file carrying a
        # day of labelling work triaged as though it carried none.
        "head_types": (
            {
                "present": True,
                "labelled": sum(
                    1 for r in parsed.head_types.get("labels") or [] if r.get("label")
                ),
                "rows": len(parsed.head_types.get("labels") or []),
                "counts": parsed.head_types.get("counts") or {},
            }
            if parsed.head_types.get("labels")
            else {"present": False}
        ),
        "patch": {
            "present": parsed.has_patch(),
            "components": sorted(parsed.patch.get("grids", {})),
            "clean": parsed.patch.get("clean", ""),
            "corrupt": parsed.patch.get("corrupt", ""),
        },
        "prompt": parsed.prompt,
        "generation": parsed.generation,
        # A graph is somebody else's measurement, so `inspect` -- which is
        # triage, run before opening anything -- has to say so. A file whose
        # only content is a graph would otherwise print as an empty session.
        "graph": (
            {
                "present": True,
                "n_nodes": parsed.graph.get("n_nodes"),
                "edges": len(parsed.graph.get("edges") or []),
                "producer": (parsed.graph.get("provenance") or {}).get("producer"),
                "model": (parsed.graph.get("provenance") or {}).get("model"),
            }
            if parsed.has_graph()
            else {"present": False}
        ),
        # A ROBOT finding, for exactly the reason the image block below exists
        # and found the same way: a file whose only content is an occlusion
        # map has no tokens, no layers and no heads, so `inspect` printed
        # "1 tokens, 0 attention maps" over a measured finding and the reader
        # deleted it.
        "vla": (
            {
                "present": True,
                "policy": (parsed.vla.get("provenance") or {}).get("policy", ""),
                "dataset": (parsed.vla.get("provenance") or {}).get("dataset", ""),
                "episode": (parsed.vla.get("provenance") or {}).get("episode"),
                "timestep": (parsed.vla.get("provenance") or {}).get("timestep"),
                "camera": (parsed.vla.get("provenance") or {}).get("camera", ""),
                "blocks": len((parsed.vla.get("occlusion") or {}).get("blocks") or []),
                # Controlled and CLEARED are different counts and neither is
                # the block total: a shift that never met a random occlusion
                # of the same size is a number, not a finding.
                "controlled": sum(
                    1
                    for b in (parsed.vla.get("occlusion") or {}).get("blocks") or []
                    if b.get("clears_control") is not None
                ),
                "cleared": sum(
                    1
                    for b in (parsed.vla.get("occlusion") or {}).get("blocks") or []
                    if b.get("clears_control") is True
                ),
                "baseline": (parsed.vla.get("occlusion") or {}).get("baseline", ""),
                # `None` survives: "not compared" is not "agrees at 0.0".
                "agreement": (parsed.vla.get("occlusion") or {}).get(
                    "attention_agreement"
                ),
                "layers": len(parsed.vla.get("attention") or []),
            }
            if parsed.has_vla()
            else {"present": False}
        ),
        # An IMAGE run, for exactly the reason the graph block above exists:
        # `inspect` is triage, run before opening anything, and a file whose
        # only content is a denoising strip has no tokens, no layers and no
        # heads -- so without this it prints as an empty session and the
        # reader deletes it.
        "image": (
            {
                "present": True,
                "kind": (parsed.image.get("provenance") or {}).get("kind", ""),
                "repo": (parsed.image.get("provenance") or {}).get("repo", ""),
                "prompt": parsed.image.get("prompt") or "",
                # `None` is not 0 here and never becomes it: an unseeded run
                # cannot be reproduced at all, and printing 0 would promise
                # that it can.
                "seed": parsed.image.get("seed"),
                "frames": len(parsed.image.get("frames") or []),
                "attention_steps": len(
                    (parsed.image.get("attention") or {}).get("steps") or []
                ),
                "readout_rows": len(
                    (parsed.image.get("readout") or {}).get("rows") or []
                ),
                "means": parsed.image.get("means") or "",
            }
            if parsed.has_image()
            else {"present": False}
        ),
    }
    if as_json:
        print(json.dumps(summary, indent=2))
        return 0

    def line(k: str, v) -> None:
        print(f"  {k:<14}{v}")

    print(f"{target.name} — {summary['bytes'] / 1024:.1f} KB")
    line("model", summary["model"])
    if summary["n_params"]:
        line("size", f"{summary['n_params'] / 1e6:,.0f}M parameters")
    if summary["device"] or summary["dtype"]:
        line("ran on", f"{summary['device'] or '?'} · {summary['dtype'] or '?'}")
    line("recorded", f"{summary['created_at']} by ModelMRI {summary['modelmri']}")
    if summary["note"]:
        line("note", summary["note"])
    print()
    line("tokens", f"{summary['n_tokens']} ({summary['n_prompt']} prompt)")
    line("shape", f"{summary['n_layers']} layers x {summary['n_heads']} heads")
    line("attention", f"{summary['attention_maps']} maps")
    if summary["scope"]:
        line("", summary["scope"])
    if summary["lens_rows"]:
        line("logit lens", f"{summary['lens_rows']} rows")
    if summary["head_types"]["present"]:
        ht = summary["head_types"]
        # LABELLED of ROWS, never the labelled count alone: "no type detected"
        # is the finding for most heads, and a bare count reads as coverage.
        kinds = ", ".join(f"{k} {v}" for k, v in sorted(ht["counts"].items()) if v)
        line(
            "head labels",
            f"{ht['labelled']:,} of {ht['rows']:,} heads labelled"
            + (f" — {kinds}" if kinds else ""),
        )
    if summary["model_diff"]["present"]:
        md = summary["model_diff"]
        line("model diff", f"{md['a'] or '?'} -> {md['b'] or '?'}")
        kl = md["kl"]
        if kl:
            # The middle half travels with the median. A median alone is the
            # single number this whole section exists to avoid printing.
            line(
                "  distance",
                f"median {kl.get('median', 0):.5f} nats, middle half "
                f"{kl.get('low', 0):.5f}-{kl.get('high', 0):.5f} over "
                f"{kl.get('n', 0)} prompt(s)",
            )
        line(
            "  diverges",
            (
                "nowhere in particular — the cosine never falls on a majority "
                "of prompts"
                if md["consensus_layer"] is None
                else f"at layer {md['consensus_layer']}"
                + (
                    f" on {md['consensus_share']:.0%} of prompts"
                    if isinstance(md["consensus_share"], (int, float))
                    else ""
                )
            ),
        )
        if md["n_heads_moved"]:
            line("  heads", f"{md['n_heads_moved']:,} recorded as moved")
    if summary["patch"]["present"]:
        line("patching", ", ".join(summary["patch"]["components"]))
        line("  clean", summary["patch"]["clean"])
        line("  corrupt", summary["patch"]["corrupt"])
    if summary["graph"]["present"]:
        g = summary["graph"]
        line("graph", f"{g['n_nodes']:,} nodes, {g['edges']:,} edges carried")
        line("  computed by", f"{g['producer']} on {g['model'] or 'an unnamed model'}")
        line("", "NOT measured by ModelMRI")
    if summary["vla"]["present"]:
        v = summary["vla"]
        line(
            "robot finding",
            f"{v['policy'] or 'an unnamed policy'} on "
            f"{v['dataset'] or 'an unnamed dataset'}",
        )
        line(
            "  frame",
            f"episode {v['episode']}, timestep {v['timestep']}"
            + (f", {v['camera']}" if v["camera"] else ""),
        )
        if v["blocks"]:
            # THREE COUNTS, NOT ONE. Occlusion is out of distribution, so
            # covering anything moves the action: a block that never met a
            # random control is not evidence, and folding the three together
            # would report a map of that as a finding.
            line(
                "  occlusion",
                f"{v['blocks']:,} block(s), {v['controlled']:,} controlled, "
                f"{v['cleared']:,} cleared their control"
                + (f" — filled with {v['baseline']}" if v["baseline"] else ""),
            )
        if v["layers"]:
            line("  attention", f"{v['layers']:,} layer(s)")
        # Spelled out rather than printed bare: `None` reads as a missing
        # field, and what it means -- nobody compared the two -- is a
        # different statement from "they agree at 0.0".
        line(
            "  agreement",
            (
                "attention was not compared with what moved the action"
                if v["agreement"] is None
                else f"{v['agreement']:+.3f} between attention and cause"
            ),
        )
    if summary["image"]["present"]:
        img = summary["image"]
        line("image run", f"{img['kind']} — {img['repo'] or 'an unnamed checkpoint'}")
        if img["frames"]:
            line("  frames", f"{img['frames']:,} decoded")
        if img["attention_steps"]:
            line("  attention", f"{img['attention_steps']:,} denoising step(s)")
        if img["readout_rows"]:
            line("  readout", f"{img['readout_rows']:,} row(s)")
        # Spelled out rather than printed as a bare value: `seed: None` reads
        # as a missing field, and the thing it actually means -- this run
        # cannot be reproduced -- is the most important line here.
        line(
            "  seed",
            (
                "none fixed, so this run does not repeat"
                if img["seed"] is None
                else img["seed"]
            ),
        )
        if img["prompt"]:
            line("  prompt", _clip(img["prompt"]))
    print()
    # Truncated on purpose: `inspect` is triage, and a 4,000-token prompt
    # scrolling past is the opposite of it. `--json` gives the whole thing.
    line("prompt", _clip(summary["prompt"]))
    line("answer", _clip(summary["generation"]))
    if summary["precision"]:
        print()
        line("precision", summary["precision"])
    return 0


def _clip(text: str, width: int = 62) -> str:
    flat = " ".join((text or "").split())
    return flat if len(flat) <= width else flat[: width - 1] + "…"


def list_models() -> int:
    """What can this machine run, without starting anything.

    `modelmri serve` and a browser tab is a lot of ceremony for "what have I
    got". This walks the same roots the picker does and prints what it finds,
    including the things that will NOT load and why -- discover.py's whole
    premise is that "why isn't my model here" is worse than "here it is, and
    here is why it won't open".
    """
    from . import discover, paths

    found = discover.discover()
    models = found.get("models", [])
    roots = found.get("roots", [])

    if not models:
        print("No models found. Looked in:")
        for r in roots:
            print(f"  {r}")
        print("\n  Point it somewhere else with MODELMRI_MODELS_DIR, or pull one:")
        print(
            "    modelmri serve   ->  the picker downloads from HuggingFace or Ollama"
        )
        return 0

    usable = [m for m in models if m.get("loadable")]
    print(f"{len(models)} found, {len(usable)} loadable\n")
    for m in sorted(models, key=lambda x: (not x.get("loadable"), x.get("name", ""))):
        mark = "  " if m.get("loadable") else "! "
        size = f"{m['size_gb']:.1f} GB" if m.get("size_gb") else ""
        print(f"{mark}{m.get('name', '?'):<38}{size:>10}  {m.get('kind', '')}")
        note = (m.get("note") or "").strip()
        if note and not m.get("loadable"):
            print(f"    {note}")
    print(f"\n  searched: {', '.join(str(r) for r in roots) or 'nothing'}")
    print(f"  downloads go to: {paths.hf_hub_cache()}")
    return 0


def open_graph(target, *, host="127.0.0.1", port=5900, browser=True) -> int:
    """Print what somebody else's attribution graph contains.

    The banner is the feature. A graph ModelMRI did not compute must never be
    mistakable for one it did, so the provenance -- file, producing tool,
    model, transcoder set -- prints before any number, and the disclaimer
    prints whether or not the file named a model.

    Nothing is loaded and no model is touched: this reads a tensor archive
    with a restricted unpickler and reduces on the tensor.
    """
    from pathlib import Path

    from . import circuit
    from .errors import BadRequest, Refusal

    try:
        graph = circuit.read(target)
    except (Refusal, BadRequest) as err:
        print(f"modelmri: {err}", file=sys.stderr)
        return 2

    p = graph.provenance
    print(f"ModelMRI {__version__} — reading {p['file']}")
    print()
    print("  PROVENANCE")
    print(f"    produced by   {p['producer']}")
    print(f"    model         {p['model'] or 'not named in the file'}")
    print(f"    transcoders   {p['scan'] or 'not named in the file'}")
    print(f"    {p['measured_by']}")
    print()

    s = graph.summary()
    print("  GRAPH")
    print(f"    nodes         {graph.n_nodes:,}")
    if graph.prompt:
        print(f"    prompt        {_clip(graph.prompt)}")
    print(
        f"    edges         {s['nonzero_edges']:,} non-zero of "
        f"{s['possible_edges']:,} possible"
        + (f"  (density {s['density']})" if s.get("density") is not None else "")
    )
    if s.get("max_abs_weight") is not None:
        print(f"    strongest     {s['max_abs_weight']:.6f}")
    strongest = graph.edges(limit=5)
    if strongest:
        print()
        print("  STRONGEST EDGES")
        for e in strongest:
            print(f"    {e['source']:>6} -> {e['target']:<6}  {e['weight']:+.6f}")
    for note in graph.notes:
        print()
        print(f"  note: {note}")
    if graph.foreign_classes:
        # Named because it is the evidence for `producer`, and because it is
        # the list of classes the reader refused to import.
        print()
        print(
            "  classes named by the file and NOT imported: "
            f"{', '.join(graph.foreign_classes)}"
        )

    # Then render it, in the same viewer as everything else. Written to a
    # temporary `.mri` rather than served from memory because that is how
    # every other finding travels -- and because it means the graph a person
    # is looking at is a file they can forward, with the provenance welded on.
    import tempfile

    from . import circuit as _circuit

    blob = _circuit.to_session(graph)
    # `mkdtemp`, not a guessable name in the shared temp root. The old path
    # was `gettempdir()/<stem>.mri` -- fixed, predictable, world-writable
    # parent, and `write_bytes` follows symlinks, so another user could
    # pre-create it as a link and have an arbitrary file overwritten with this
    # caller's privileges (CWE-377/CWE-59). The file also holds the stranger's
    # prompt, so a 0700 directory is the right home for it.
    tmp = (
        Path(tempfile.mkdtemp(prefix="modelmri-graph-"))
        / f"{Path(graph.path).stem}.mri"
    )
    tmp.write_bytes(blob)
    print()
    print(f"  written as {tmp}  ({len(blob) / 1024:.1f} KB) — forwardable")
    print("  no model will be loaded — this is somebody else's measurement")
    print()
    serve_viewer(tmp, host=host, port=port, browser=browser)
    return 0


def export_trace(
    trace_id: str | None,
    *,
    endpoint: str,
    headers: list[str],
    service_name: str = "modelmri",
    dry_run: bool = False,
) -> int:
    """Hand one recorded run to the collector the team already runs.

    Prints the semconv generation it targeted. That is not decoration: the
    `gen_ai.*` conventions were moved out of the main semantic-conventions
    repo on 2026-06-12 into one with no releases and no tags, so "which
    vocabulary do these spans speak" is a real question a consumer will have,
    and the answer has to travel with the spans and be visible at the moment
    of sending.
    """

    from . import otel, paths
    from .errors import BadRequest, Refusal
    from .traces import TraceStore

    db = paths.trace_db_path()
    if not db.exists():
        print(f"No trace database yet ({db}).")
        print("  Record one:  uv run python examples/record_demo.py")
        return 1

    store = TraceStore(db)
    if not trace_id:
        rows = store.list_traces()
        if not rows:
            print(f"No traces recorded yet ({db})")
            return 1
        trace_id = rows[0]["id"]
        print(f"No id given, so: the most recent, {rows[0]['name']!r} ({trace_id})")

    doc = store.get_trace(trace_id)
    if doc is None:
        print(f"No trace with id {trace_id!r}. `modelmri traces` lists them.")
        return 1

    try:
        extra = _parse_headers(headers)
        if dry_run:
            body = otel.to_otlp(doc, service_name=service_name)
            print(json.dumps(body, indent=2))
            spans = body["resourceSpans"][0]["scopeSpans"][0]["spans"]
            print(
                f"\n{len(spans)} spans, semconv {otel.SEMCONV_GENERATION}, "
                "nothing sent (--dry-run)"
            )
            return 0
        result = otel.send(doc, endpoint, headers=extra, service_name=service_name)
    except (Refusal, BadRequest) as err:
        print(f"{err}")
        return 1

    print(f"{result.spans} spans -> {result.endpoint}  (HTTP {result.status})")
    print(f"  semconv generation: {result.semconv}")
    if result.rejected_spans:
        # A 200 does not mean accepted. OTLP returns partialSuccess in the
        # body, and a collector over quota or running a filter uses it to say
        # it dropped spans while still answering 200 -- so "N spans -> ok"
        # without reading it is a claim nobody checked.
        print(
            f"  the collector REJECTED {result.rejected_spans} of "
            f"{result.spans}; {result.accepted} accepted"
            + (f" — {result.reject_message}" if result.reject_message else "")
        )
    if result.epoch_fallback:
        print(
            "  this trace's start time could not be parsed, so every span "
            "sits at the epoch (1970) in your collector. The spans are "
            "correct relative to each other and wrong absolutely."
        )
    if result.undated_spans:
        # Said out loud rather than left in an attribute nobody reads. OTLP
        # has no way to express "the end time is unknown", so these went as
        # zero-length spans, and a zero-length span on a waterfall reads as an
        # instantaneous operation -- a claim about something nobody measured.
        print(
            f"  {result.undated_spans} of {result.spans} had no recorded "
            f"duration and were sent as zero-length, marked "
            f"`modelmri.duration.recorded=false`. OTLP cannot say "
            f'"unknown" for an end time.'
        )
    return 0


def _parse_headers(pairs: list[str]) -> dict[str, str]:
    """`K=V` strings into a dict, refusing the ones that are not.

    A silently dropped auth header is a 401 several minutes later against a
    hosted collector, so a malformed one stops here with the offending text.
    """
    from .errors import BadRequest

    out: dict[str, str] = {}
    for raw in pairs or []:
        if "=" not in raw:
            raise BadRequest(f"--header wants K=V, got {raw!r}")
        key, _, value = raw.partition("=")
        key, value = key.strip(), value.strip()
        if not key:
            raise BadRequest(f"--header has an empty name in {raw!r}")
        # `.strip()` only removes the ENDS. An interior newline survived it,
        # and http.client then either raises ValueError (which escaped as a
        # traceback rather than a refusal) or -- for newline-space, which its
        # regex deliberately permits as obs-fold -- puts a folded header on
        # the wire. Header smuggling through a proxy sitting in front of the
        # collector is the thing that buys.
        for text, what in ((key, "name"), (value, "value")):
            bad = next((c for c in text if ord(c) < 0x20 or ord(c) == 0x7F), None)
            if bad is not None:
                raise BadRequest(
                    f"--header {what} contains a control character "
                    f"({bad!r}) in {raw!r}. A newline in a header value is "
                    "how a second header gets smuggled onto the wire."
                )
        if not value:
            # The 401-several-minutes-later this function exists to prevent.
            raise BadRequest(
                f"--header {key!r} has an empty value. An empty auth header "
                "is refused by a collector minutes later and reads as a "
                "network problem."
            )
        if key.lower() == "content-type":
            # Would override the JSON body's own type and make the collector
            # parse it as something it is not.
            raise BadRequest(
                "--header cannot set Content-Type: this sends OTLP/HTTP with "
                "a JSON body and the type has to match it."
            )
        out[key] = value
    return out


def list_traces() -> int:
    """Agent runs recorded on this machine, newest first.

    The panel that shows these is behind a server and a scroll. `record_demo`
    finishing in another terminal is the moment you want this answer, and the
    terminal is where you already are.
    """
    from . import paths
    from .traces import TraceStore

    db = paths.trace_db_path()
    if not db.exists():
        print("No trace database yet.")
        print(f"  It would live at {db}")
        print("\n  Record one:")
        print("    uv run python examples/record_demo.py")
        print("  or instrument your own agent with `from modelmri.record import trace`")
        return 0

    rows = TraceStore(db).list_traces()
    if not rows:
        print(f"No traces recorded yet ({db})")
        print("\n  Record one:  uv run python examples/record_demo.py")
        return 0

    print(f"{len(rows)} recorded  ({db})\n")
    for r in rows[:40]:
        flag = "demo" if r.get("demo") else ""
        errs = f"  {r['n_errors']} failed" if r.get("n_errors") else ""
        # `n_timed`, which the store ships beside `total_ms` for exactly this.
        # A step's `duration_ms` is optional — `otel.py` leaves it None when the
        # span carried no end time, and `/api/traces/import` documents it as
        # optional — so a run where nothing was timed has `total_ms` 0, and
        # this printed "0.0s" as a measurement of a run that took some real
        # amount of time nobody recorded. `AgentsPanel` gets this right; the
        # CLI was the one consumer ignoring the field.
        n_timed = int(r.get("n_timed") or 0)
        n_steps = int(r["n_steps"])
        if not n_timed:
            took = "  not timed"
        else:
            # ">=" when only some steps were timed: the total is a floor, not
            # the run's duration.
            mark = ">=" if n_timed < n_steps else " "
            took = f"{mark}{r['total_ms'] / 1000:>6.1f}s"
        print(f"  {r['name'][:30]:<30} {n_steps:>4} steps  {took:>9}{errs}  {flag}")
    if len(rows) > 40:
        print(f"  ... and {len(rows) - 40} more")
    print("\n  Open them in the browser:  modelmri serve")
    return 0


def serve_viewer(target, *, host: str, port: int, browser: bool) -> None:
    """Serve the bundled `.mri` viewer, using only the standard library.

    `modelmri open` used to start the full application, which imports torch
    and transformers — measured at 26 seconds — to display a 54 KB recording
    that needs neither. The first person to run it pressed ctrl-c partway
    through, reasonably concluding it had hung.

    Nothing here imports anything heavy. It serves the same viewer bundle
    that is published to GitHub Pages, plus the one file, and the page opens
    it from the `?f=` link rather than making you find and drop a file you
    just named on the command line.

    Two things this deliberately does NOT do: bind anything but the loopback
    interface, and serve any directory other than the viewer's own. A local
    file reader has no business being reachable from the network or exposing
    the tree it happens to be started in.
    """
    import functools
    import http.server
    import socketserver
    import threading
    import webbrowser
    from importlib.resources import files
    from pathlib import Path

    bundle = Path(str(files("modelmri") / "static" / "viewer"))
    if not (bundle / "index.html").is_file():
        print(
            "modelmri: this build has no bundled viewer.\n"
            "  Use `modelmri serve` and open the file from the page, or read\n"
            "  it at https://muhammadmahadazher.github.io/ModelMRI/viewer/",
            file=sys.stderr,
        )
        raise SystemExit(1)

    payload = Path(target).read_bytes()
    # Derived from the file, not fixed. `session.mri` for everything meant the
    # URL said nothing about what was open, and a graph served as "session"
    # is the one thing this release is trying not to do.
    #
    # Sanitised hard, because `name` becomes both a served path and a `?f=`
    # value: anything outside this alphabet could walk the path or smuggle a
    # query, so it is rebuilt character by character rather than escaped.
    stem = "".join(c if c.isalnum() or c in "-_" else "-" for c in Path(target).stem)[
        :60
    ]
    name = f"{stem or 'session'}.mri"

    # Said once, plainly, rather than assumed. The docstring above promises
    # loopback; --host takes anything, and a recording is somebody's prompts.
    if host not in ("127.0.0.1", "localhost", "::1"):
        print(
            f"  NOTE: serving on {host}, not loopback — anyone who can reach "
            f"this\n  machine on port {port} can read this recording.",
            file=sys.stderr,
        )

    class Handler(http.server.SimpleHTTPRequestHandler):
        def __init__(self, *a, **kw):
            super().__init__(*a, directory=str(bundle), **kw)

        def _is_ours(self) -> bool:
            """Reject a request that reached us under someone else's name.

            A page on any website can point a name it controls at 127.0.0.1
            and then read whatever answers — DNS rebinding. The recording is
            somebody's prompts and generations, so checking the Host is the
            difference between "served to me" and "served to a tab I happened
            to have open".
            """
            sent = (self.headers.get("Host") or "").rsplit(":", 1)[0].strip("[]")
            if sent in ("127.0.0.1", "localhost", "::1", "", host):
                return True
            self.send_error(
                421,
                "Misdirected Request",
                f"This viewer only answers to localhost, not {sent!r}.",
            )
            return False

        def _payload_headers(self) -> bool:
            self.send_response(200)
            self.send_header("Content-Type", "application/octet-stream")
            self.send_header("Content-Length", str(len(payload)))
            # It is one local file for one local page; nothing should cache
            # it, and nothing else should be allowed to frame or embed it.
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.end_headers()
            return True

        def do_GET(self):  # stdlib's spelling
            if not self._is_ours():
                return
            if self.path.split("?")[0] == f"/{name}":
                self._payload_headers()
                self.wfile.write(payload)
                return
            super().do_GET()

        def do_HEAD(self):
            # Overriding do_GET alone made HEAD /session.mri answer 404 while
            # GET answered 200 — the two disagreeing about whether a file
            # exists is the kind of thing that breaks a client for no visible
            # reason.
            if not self._is_ours():
                return
            if self.path.split("?")[0] == f"/{name}":
                self._payload_headers()
                return
            super().do_HEAD()

        def log_message(self, *a):  # a request log is noise here
            pass

        def handle_one_request(self):
            # A browser that closes a keep-alive socket, or a scanner sending
            # a malformed path, otherwise prints a full traceback into the
            # terminal of someone who only wanted to look at a file.
            try:
                super().handle_one_request()
            except (ConnectionError, TimeoutError):
                self.close_connection = True

    class Server(socketserver.ThreadingTCPServer):
        # ThreadingTCPServer, not TCPServer. `daemon_threads` on a
        # single-threaded server does nothing at all: one browser holding a
        # keep-alive socket open stalled every later request, and the page
        # simply never finished loading.
        allow_reuse_address = True
        daemon_threads = True

    try:
        httpd = Server((host, port), functools.partial(Handler))
    except OSError as err:
        print(f"modelmri: cannot listen on {host}:{port} — {err}", file=sys.stderr)
        print("  another ModelMRI may be running; try --port 5901", file=sys.stderr)
        raise SystemExit(1) from err

    # The port it ACTUALLY got, not the one that was asked for. `--port 0`
    # means "any free one", and the two lines below are the only way a reader
    # learns which — printing the request would send them to
    # `http://127.0.0.1:0/`, and opening that in a browser is the whole
    # feature failing silently.
    port = httpd.server_address[1]
    url = f"http://{host}:{port}/?f={name}"
    if browser:
        threading.Timer(0.4, lambda: webbrowser.open(url)).start()
    print(f"  reading it at {url}\n  (ctrl-c to stop)")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped.")
    finally:
        httpd.server_close()


def _walk_bytes(root, mark=None) -> tuple[int, int]:
    """Bytes this tree occupies on disk, and how many of them are under `mark`.

    `lstat` semantics, and symlinks are not followed. The HuggingFace cache is
    built out of `snapshots/` symlinks pointing at `blobs/`, so following them
    counts every weight file two or three times: an 8 GB cache was reported at
    20 GB, in the confirmation prompt for deleting it.

    `os.scandir`, not `rglob` + `lstat`, and the difference is not academic.
    MEASURED on this project's own data directory (1.23 GB, 38,664 entries, on
    a network-backed drive): the `rglob` version took 68.4 s. Two costs, both
    avoidable — `rglob` materialises a `Path` per entry, and `f.is_symlink()`
    after `f.lstat()` is a SECOND stat syscall asking what `st_mode` from the
    first one already answered. `S_ISREG` is false for a symlink under
    `lstat`, so that call was never deciding anything.

    `mark` exists because the second figure used to cost a second walk.
    `modelmri uninstall` reports the action expert's venv separately, and that
    venv is INSIDE the data directory — 38,635 of those 38,664 entries are it
    — so the same subtree was traversed twice, once for each line. Now the
    caller asks for both totals and the tree is read once.

    A directory that cannot be opened is skipped rather than ending the walk.
    Either way the number can only be an undercount, and this is the smaller
    one: the previous outer handler returned whatever had accumulated, so one
    unreadable folder near the start reported gigabytes as almost empty.
    """
    total = marked = 0
    target = os.path.normcase(os.path.abspath(str(mark))) if mark is not None else None
    here = str(root)
    stack = [
        (here, target is not None and os.path.normcase(os.path.abspath(here)) == target)
    ]
    while stack:
        folder, tagged = stack.pop()
        try:
            with os.scandir(folder) as entries:
                for entry in entries:
                    try:
                        if entry.is_dir(follow_symlinks=False):
                            stack.append(
                                (
                                    entry.path,
                                    tagged
                                    or (
                                        target is not None
                                        and os.path.normcase(
                                            os.path.abspath(entry.path)
                                        )
                                        == target
                                    ),
                                )
                            )
                            continue
                        st = entry.stat(follow_symlinks=False)
                    except OSError:
                        # An entry that vanished between the scan and the stat,
                        # or one this account cannot stat. Skipping just this
                        # entry is what keeps the walk going.
                        continue
                    # S_ISREG on the entry itself: a symlink contributes only
                    # its own tiny inode, and the blob it points at is counted
                    # once, where it actually lives.
                    if st.st_mode & 0o170000 == 0o100000:
                        total += st.st_size
                        if tagged:
                            marked += st.st_size
        except OSError:
            continue
    return total, marked


def _tree_bytes(root) -> int:
    """Just the total, for the callers that do not need a second figure."""
    return _walk_bytes(root)[0]


def policy_command(args) -> int:
    """`modelmri policy …` — the action expert's own environment and process.

    Separate from every other command in this file because it is the only one
    that builds a SECOND Python environment. That is not an implementation
    detail to be hidden: it costs about 6 GB, it exists because lerobot's pins
    cannot share an environment with ModelMRI's, and a user who does not know
    that will be surprised by the disk usage and by the fact that upgrading
    ModelMRI does not upgrade this.
    """
    from . import policy as _policy

    what = getattr(args, "policy_command", None)

    if what == "install":
        # Ask before spending gigabytes. `--yes` for scripts; the prompt is
        # for the person who typed the command without reading the help.
        if not args.yes:
            local = _policy.source_dir()
            print(f"ModelMRI {__version__} — installing the action expert\n")
            print(f"  into    {_policy.venv_dir()}")
            print(f"  package {_policy.requirement()}")
            print(
                f"  source  {'this checkout' if local else 'PyPI'}\n"
                f"  size    about {_policy.VENV_DISK_BYTES / 1e9:,.0f} GB — it "
                f"downloads its own torch\n"
                f"  why     lerobot pins torch and numpy hard enough that "
                f"installing it\n"
                f"          beside ModelMRI breaks ModelMRI."
            )
            try:
                reply = input("\nBuild it? [y/N] ").strip().lower()
            except (EOFError, KeyboardInterrupt):
                reply = ""
            if reply not in ("y", "yes"):
                print("nothing installed.")
                return 1

        try:
            result = _policy.install(
                force=args.force, echo=lambda line: print(f"  {line}", flush=True)
            )
        except Exception as err:
            print(f"\nmodelmri: {err}", file=sys.stderr)
            return 2
        print(f"\n{result['means']}")
        return 0

    if what == "start":
        # The residency refusal only fires if somebody TELLS it what this
        # machine has, and this is the only production caller. Without these
        # three arguments `check_capacity` saw `vram_gb=None`, took its
        # unknown-VRAM early return, and the whole "two processes cannot
        # offload into each other's memory" rule never ran from the command
        # line that starts the second process. A guard nobody passes evidence
        # to is a guard that is off.
        found = None
        try:
            from . import devices as _devices

            found = _devices.detect()
        except Exception:
            # Measuring the machine is best-effort; failing to measure it must
            # not stop a sidecar from starting. `check_capacity` already
            # treats unknown VRAM as "do not refuse on no evidence".
            found = None

        try:
            status = _policy.start(
                policy_repo=args.repo,
                device=args.device,
                echo=lambda line: print(f"  {line}", flush=True),
                vram_gb=getattr(found, "vram_gb", None) if found else None,
                accel_name=getattr(found, "name", "") if found else "",
                confirm=args.yes,
            )
        except Exception as err:
            print(f"modelmri: {err}", file=sys.stderr)
            return 2
        print(f"\n  port      {status.port}")
        print(f"  {status.means()}")
        if not args.wait:
            return 0
        # Foreground by default: the sidecar is a child of THIS process, and a
        # command that returned here would take its own child down with it.
        # `--no-wait` is for the case where the caller has its own supervisor.
        print("\nServing. Ctrl-C stops it.")
        try:
            while True:
                time.sleep(3600)
        except KeyboardInterrupt:
            print("\nstopping.")
        finally:
            _policy.stop()
        return 0

    if what == "status":
        status = _policy.status()
        if args.json:
            print(json.dumps(status.to_dict(), indent=2))
            return 0
        print(f"ModelMRI {__version__} — the action expert on this machine\n")
        print(f"  installed {'yes' if _policy.installed() else 'no'}")
        print(f"  venv      {_policy.venv_dir()}")
        print(f"  running   {'yes' if status.running else 'no'}")
        if status.port:
            print(f"  port      {status.port}")
        if status.policy_repo:
            print(f"  policy    {status.policy_repo}")
            print(f"  revision  {status.revision or 'not recorded'}")
            print(f"  device    {status.device} · {status.dtype}")
            # An empty dict is a fact worth printing. It means an overlay
            # against a dataset's recorded actions must be refused rather
            # than drawn, and a blank line here would read as "fine".
            units = status.normalisation
            named = (
                ", ".join(sorted(units))
                if units
                else "not published by this policy — do not overlay"
            )
            print(f"  units     {named}")
        print(f"\n  {status.means()}")
        return 0 if status.running else 1

    print("modelmri policy: say install, start or status", file=sys.stderr)
    return 2


def scan_weights(target, *, as_json: bool = False, limit: int = 200) -> int:
    """`modelmri scan` — look inside weights before anything loads them.

    Exit 1 when something dangerous is found, so this drops into CI. An
    UNSCANNED file is exit 0 and is still printed: refusing every format the
    scanner cannot read would make the gate a function of its own coverage,
    and most of what it cannot read is harmless. What it must never do is
    print "safe" for a file nobody looked inside.
    """
    from pathlib import Path

    from . import weights_scan

    where = Path(target).expanduser()
    reports = (
        weights_scan.scan_dir(where, limit=limit)
        if where.is_dir()
        else [weights_scan.scan(where)]
    )
    if as_json:
        print(json.dumps([r.to_dict() for r in reports], indent=2))
        return 1 if any(r.dangerous for r in reports) else 0

    print(f"ModelMRI {__version__} — what is inside {where}")
    print()
    if not reports:
        # An unopenable directory walks to an empty tree, so "nothing
        # weight-shaped here" was printed for a folder whose contents nobody
        # ever saw — the unearned all-clear the server side already refuses
        # to give. `readable` is the difference between the two.
        for line in _wrap(
            weights_scan.summary(
                reports, [], [], readable=getattr(reports, "readable", True)
            ),
            76,
        ):
            print(f"  {line}")
        return 0

    mark = {
        weights_scan.DANGEROUS: "DANGER",
        weights_scan.UNSCANNED: "  --  ",
        weights_scan.SAFE: "  ok  ",
    }
    for r in reports:
        print(f"  {mark[r.verdict]} {Path(r.path).name}")
        for f in r.findings:
            for line in _wrap(f"{f.kind}: {f.detail}", 66):
                print(f"           {line}")
        if r.verdict == weights_scan.UNSCANNED and r.reason:
            for line in _wrap(r.reason, 66):
                print(f"           {line}")

    bad = [r for r in reports if r.dangerous]
    unknown = [r for r in reports if r.verdict == weights_scan.UNSCANNED]
    print()
    # `weights_scan.summary`, not a sentence written here. The copy that used
    # to live at this spot said "N file(s) read, nothing executable found. M
    # could not be read" with N counting the unread files too — the exact
    # contradiction the shared version's docstring says was settled once
    # already — and it never mentioned the `--limit` cap at all, so a tree
    # over the limit printed a verdict on a subset as if it were the whole
    # tree. `ScanTree` carries both facts; this now reads them.
    for line in _wrap(
        weights_scan.summary(
            reports,
            bad,
            unknown,
            n_total=getattr(reports, "n_total", len(reports)),
            readable=getattr(reports, "readable", True),
        ),
        76,
    ):
        print(f"  {line}")
    return 1 if bad else 0


def uninstall(*, yes: bool = False, models: bool = False) -> int:
    """Remove everything ModelMRI has written, after showing what that is.

    Leaving a trail nobody can find is the same discourtesy as a bad install.
    Every location is resolved by `paths`, per-platform and per-account, so
    this deletes *your* directories on *your* OS — there is no list of
    guessed paths here to go stale.

    Two things it deliberately will not do without being asked:

    * The HuggingFace cache is shared. `transformers`, `datasets` and every
      other tool on the machine read the same directory, so deleting it as
      part of removing one app would take other people's downloads with it.
      `--models` opts in, and the size is shown either way.
    * It cannot remove the installed package while running from inside it, so
      it prints the `pip uninstall` line rather than pretending.
    """
    from pathlib import Path

    from . import paths

    # `flush`, and it is not cosmetic. Everything below measures directories,
    # and under a pipe stdout is block-buffered — MEASURED, `modelmri
    # uninstall < /dev/null` printed NOTHING for two minutes and then the
    # whole page at once, which reads as a hung command. Each line below
    # flushes for the same reason.
    print(f"ModelMRI {__version__} — what is on this machine\n", flush=True)

    targets: list[tuple[str, Path]] = []
    kept: list[Path] = []
    for label, path in (
        ("data", paths.data_dir()),
        ("config", paths.config_dir()),
        ("cache", paths.cache_dir()),
        ("legacy", paths.legacy_root()),
    ):
        if path is None:
            continue
        resolved = Path(path)
        if not resolved.exists():
            continue
        # On Windows cache_dir() is data_dir()/Cache — nested, not equal — so
        # an equality check saw two distinct paths, listed both, deleted the
        # parent, and then reported the child as a failure it had itself
        # caused. Containment is the test that matches the comment.
        if any(resolved == k or k in resolved.parents for k in kept):
            continue
        kept.append(resolved)
        targets.append((label, resolved))

    if not targets:
        print("  nothing to remove — ModelMRI has not written anything here.")

    # Named separately even though it is INSIDE `data` and already counted
    # there. A 6 GB figure on a line labelled "data" reads as recordings; this
    # is a whole second Python with its own torch, and somebody deciding
    # whether to delete deserves to know which of the two they are looking at.
    #
    # Counted DURING the `data` walk rather than by a second one. The venv is
    # a subtree of the data directory — MEASURED, 38,635 of that directory's
    # 38,664 entries are it — so the two lines used to read the same files
    # twice: 68.4 s and 78.9 s of the ~124 s this command spent before showing
    # anything at all.
    from . import policy as _policy

    venv = _policy.venv_dir()
    venv_bytes = 0
    for label, path in targets:
        # The path FIRST, then the size, so a slow directory shows what is
        # being measured rather than an idle cursor.
        print(f"  {label:<8} {path}", end="", flush=True)
        total, within = _walk_bytes(path, mark=venv if label == "data" else None)
        if label == "data":
            venv_bytes = within
        print(f"  ({total / 1e6:.1f} MB)", flush=True)

    if venv.exists():
        # Walked on its own ONLY when it was not already counted — a
        # `MODELMRI_HOME` split across two disks puts the venv outside the
        # data directory, and then `within` is legitimately 0.
        if not venv_bytes:
            venv_bytes = _tree_bytes(venv)
        print(
            f"\n  of which {venv_bytes / 1e9:.2f} GB is the action "
            f"expert's own\n           Python environment at {venv}.",
            flush=True,
        )

    # Existence, not size, decides whether this is disclosed and whether
    # `--models` acts on it. Gating on bytes meant an empty-but-present cache
    # silently did nothing under `--models`, and the SHARED warning — the
    # whole reason this is opt-in — disappeared with it.
    hub = paths.hf_hub_cache()
    if hub.exists():
        print(f"\n  models   {hub}", end="", flush=True)
        hub_bytes = _tree_bytes(hub)
        print(f" ({hub_bytes / 1e9:.2f} GB)", flush=True)
        print(
            "           SHARED with transformers, datasets and anything else "
            "using\n           the HuggingFace cache."
            + (" Deleting it, as asked." if models else " Left alone.")
        )
        if models:
            targets.append(("models", hub))

    if not targets:
        return 0

    if not yes:
        print()
        try:
            reply = input("Delete the above? [y/N] ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            reply = ""
        if reply not in ("y", "yes"):
            print("nothing deleted.")
            return 1

    import shutil

    freed = 0
    failures = 0
    for label, path in targets:
        before = _tree_bytes(path)
        errors: list[str] = []
        # rmtree stops at the FIRST entry it cannot remove, having already
        # deleted everything it walked before that. Reporting "could not
        # remove <the whole directory>" then tells the user it was left alone
        # when most of it is gone. Collect every failure and re-measure.
        # `onexc` is 3.12+; `onerror` is what 3.10 and 3.11 have, and it is
        # only deprecated, not removed. This package supports >=3.10, so it
        # has to speak both.
        if sys.version_info >= (3, 12):
            shutil.rmtree(path, onexc=lambda _f, p, e: errors.append(f"{p}: {e}"))
        else:
            shutil.rmtree(
                path, onerror=lambda _f, p, info: errors.append(f"{p}: {info[1]}")
            )
        after = _tree_bytes(path) if path.exists() else 0
        freed += before - after
        if errors:
            failures += 1
            print(
                f"  PARTLY removed {label:<8} {path}\n"
                f"    {len(errors)} item(s) could not be deleted; "
                f"{after / 1e6:.1f} MB remains:",
                file=sys.stderr,
            )
            for line in errors[:5]:
                print(f"      {line}", file=sys.stderr)
        else:
            print(f"  removed {label:<8} {path}")

    print(f"\nfreed {freed / 1e6:.1f} MB")
    print("\nThe package itself is still installed. To remove it:")
    print("  pip uninstall modelmri modelmri-record")
    # A partial delete is not success. Exit non-zero so a script that chains
    # off this does not assume the machine is clean.
    return 2 if failures else 0


def main() -> None:
    # BEFORE anything else. huggingface_hub computes its cache constants at
    # import time, so this has to win that race -- every reader inside
    # ModelMRI re-reads the environment at call time and will follow.
    from . import paths as _paths

    _paths.adopt_models_home()

    # Windows consoles hand Python a cp1252 stdout, which cannot encode a path
    # containing (say) a Cyrillic or CJK username. Printing where things live
    # would then die with a UnicodeEncodeError -- the command that exists to
    # answer "where is my stuff?" failing precisely for the users whose stuff
    # is hardest to find. backslashreplace degrades instead of raising.
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            try:
                reconfigure(encoding="utf-8", errors="backslashreplace")
            except (ValueError, OSError):
                # This pair is exact, not defensive, and was checked by
                # provoking each case on CPython 3.13.12: a closed underlying
                # buffer gives ValueError("I/O operation on closed file."), a
                # detached one ValueError("underlying buffer has been
                # detached"), and a stream already read from gives
                # io.UnsupportedOperation — which subclasses BOTH OSError and
                # ValueError, so the tuple already had it. A bad codec would
                # be LookupError, but the encoding here is the literal
                # "utf-8", so there is no way to reach it. Streams that are
                # not TextIOWrapper (io.StringIO under pytest's capture) have
                # no `.reconfigure` at all and never get here — the getattr
                # above stops them.
                #
                # Carrying on is right because this is the fallback, not the
                # feature: a stream we cannot reconfigure is one nobody is
                # reading, and refusing to start `modelmri` over the encoding
                # of a closed stdout would be the actual failure.
                pass

    parser = argparse.ArgumentParser(
        prog="modelmri",
        description="ModelMRI — Chrome DevTools for AI models and agents.",
    )
    parser.add_argument(
        "--version", action="version", version=f"modelmri {__version__}"
    )
    sub = parser.add_subparsers(dest="command")

    audit = sub.add_parser(
        "audit",
        help="Prove a robot dataset is intact, or say exactly where it is not",
    )
    audit.add_argument(
        "dataset",
        nargs="?",
        default="",
        help="a cached LeRobot repo id, e.g. lerobot/pusht",
    )
    audit.add_argument(
        "--json", action="store_true", help="machine-readable, for a CI gate"
    )

    serve = sub.add_parser("serve", help="Start the ModelMRI server")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=_port, default=5900)

    opener = sub.add_parser(
        "open",
        help="Open a shared analysis (.mri) or an attribution graph (.pt) — "
        "no model needed",
    )
    opener.add_argument("file", help="the .mri or circuit-tracer .pt someone sent you")
    opener.add_argument("--host", default="127.0.0.1")
    opener.add_argument("--port", type=_port, default=5900)
    opener.add_argument(
        "--no-browser", action="store_true", help="just serve it, don't open a tab"
    )

    # `open` starts a viewer; this one prints and exits. Someone triaging an
    # issue with six attached `.mri` files wants to know which is which
    # without opening six browser tabs, and a `.mri` is JSON under a gzip
    # header, so answering that needs no browser, no model and no torch.
    reader = sub.add_parser(
        "inspect", help="Print what a .mri contains, without opening anything"
    )
    reader.add_argument("file", help="the .mri to describe")
    reader.add_argument(
        "--json",
        action="store_true",
        help="emit the summary as JSON instead of text",
    )

    differ = sub.add_parser(
        "diff",
        help="Compare two .mri of the same prompt and exit non-zero when "
        "something moved",
    )
    differ.add_argument("a", help="the baseline .mri")
    differ.add_argument("b", help="the .mri to check against it")
    differ.add_argument(
        "--fail-over",
        type=float,
        default=None,
        metavar="X",
        help="exit 1 only when a metric moved by more than X, in that "
        "metric's own units. Omit to fail on anything past the files' own "
        "noise floor.",
    )
    differ.add_argument("--json", action="store_true", help="emit JSON")

    experiments = sub.add_parser(
        "experiments",
        help="Compare two runs of one dataset, case by case, and exit "
        "non-zero when cases got worse",
    )
    experiments.add_argument("before", help="the baseline experiment .jsonl")
    experiments.add_argument("after", help="the experiment to check against it")
    experiments.add_argument(
        "--metric",
        required=True,
        metavar="NAME",
        help="which recorded metric to compare. Required: a run can carry "
        "several, and picking one for you would pick the conclusion.",
    )
    # No default and no name map. KL divergence is better lower and
    # faithfulness better higher; guessing inverts every conclusion in half of
    # all comparisons and the output looks entirely right either way.
    direction = experiments.add_mutually_exclusive_group(required=True)
    direction.add_argument(
        "--higher-is-better",
        dest="higher_is_better",
        action="store_true",
        help="a larger number is an improvement (faithfulness, accuracy)",
    )
    direction.add_argument(
        "--lower-is-better",
        dest="higher_is_better",
        action="store_false",
        help="a smaller number is an improvement (KL divergence, loss)",
    )
    experiments.add_argument(
        "--dataset",
        default=None,
        metavar="PATH",
        help="the dataset both runs cover, so the report can quote what each "
        "case asked. Omitted means nothing looked, which is reported as "
        "unknown rather than as none.",
    )
    experiments.add_argument(
        "--floor",
        type=float,
        default=None,
        metavar="X",
        help="smallest difference worth calling a change, in the metric's own "
        "units. Omit to use the coarser floor the two files themselves state, "
        "or exact arithmetic when neither states one.",
    )
    experiments.add_argument(
        "--fail-on-worse",
        type=int,
        default=0,
        metavar="N",
        help="exit 1 when more than N cases got worse. Default 0 — any "
        "regression fails, which is what a gate is for.",
    )
    experiments.add_argument("--json", action="store_true", help="emit JSON")

    sweeper = sub.add_parser(
        "sweep",
        help="Run one measurement over many prompts and report the "
        "distribution, not one number",
    )
    sweeper.add_argument(
        "prompts",
        nargs="?",
        default=None,
        help="a .jsonl (objects with `prompt`) or .txt. Omitted with --resume.",
    )
    sweeper.add_argument(
        "--resume",
        default=None,
        metavar="SWEEP_ID",
        help="finish a saved sweep, keeping every prompt already measured. "
        "`modelmri sweeps` lists the ids.",
    )
    sweeper.add_argument("--model", help="which model to load")
    sweeper.add_argument(
        "--metric",
        default="heads",
        choices=("heads", "tokens", "features"),
        help="heads: the ablation ranking (default). features: SAE features. "
        "tokens: per-prompt only, never aggregated",
    )
    sweeper.add_argument(
        "--baseline", default="zero", help="heads only: zero, mean or resample"
    )
    sweeper.add_argument(
        "--layer", type=int, default=None, help="one layer; omit to sweep all"
    )
    sweeper.add_argument("--max-new-tokens", type=int, default=8)
    sweeper.add_argument("--out-dir", default=None, help="write one .mri per prompt")
    sweeper.add_argument("--jsonl", default=None, help="write one row per prompt")
    sweeper.add_argument(
        "--yes", action="store_true", help="run even when the projection is large"
    )
    sweeper.add_argument("--json", action="store_true", help="emit JSON")

    mcp = sub.add_parser(
        "mcp",
        help="Speak MCP over stdio, so an agent can call the measurements "
        "directly (read-only)",
    )
    mcp.add_argument(
        "--attach",
        default="",
        metavar="URL",
        help="drive an already-running `modelmri serve` instead of loading a "
        "second copy of the model, e.g. http://127.0.0.1:5900",
    )

    gate = sub.add_parser(
        "check",
        help="Gate a merge on structural facts about a recorded run "
        "(exits non-zero when an assertion fails)",
    )
    gate.add_argument("target", help="a trace id in this machine's store, or a .json")
    gate.add_argument(
        "--no-errors", action="store_true", help="fail if any step recorded an error"
    )
    gate.add_argument(
        "--max-steps", type=int, default=None, help="fail above this many steps"
    )
    gate.add_argument(
        "--no-retry-storms",
        action="store_true",
        help="fail if one name failed twice in a row inside the retry window",
    )
    gate.add_argument(
        "--no-loops",
        action="store_true",
        help="fail if any sequence of steps repeated back to back",
    )
    gate.add_argument(
        "--max-repeat",
        type=int,
        default=None,
        help="fail if a step ran more than N times with the same input",
    )
    gate.add_argument(
        "--max-ms",
        type=int,
        default=None,
        help="OPT-IN AND FLAKY: fail above this total wall-clock. A shared CI "
        "runner is slow for reasons unrelated to your diff, and a gate that "
        "goes red on a noisy neighbour teaches people to ignore the check",
    )
    gate.add_argument(
        "--json", action="store_true", help="emit the report as JSON instead of text"
    )

    checker = sub.add_parser(
        "verify",
        help="Re-run the measurements in a .mri on this machine and report "
        "what reproduced",
    )
    checker.add_argument("file", help="the .mri to check")
    checker.add_argument(
        "--json",
        action="store_true",
        help="emit the report as JSON instead of text",
    )

    sub.add_parser(
        "models", help="List the models on this machine, and what will not load"
    )
    sub.add_parser("traces", help="List agent runs recorded on this machine")

    export = sub.add_parser(
        "export",
        help="Send a recorded run to an OTLP collector (Langfuse, Phoenix, "
        "Grafana, Honeycomb)",
    )
    export.add_argument(
        "trace_id",
        nargs="?",
        help="which run; omit for the most recent",
    )
    export.add_argument(
        "--otlp",
        required=True,
        metavar="ENDPOINT",
        help="OTLP/HTTP endpoint, e.g. http://localhost:4318 (port 4317 is "
        "gRPC and is not spoken)",
    )
    export.add_argument(
        "--header",
        action="append",
        default=[],
        metavar="K=V",
        help="extra request header, repeatable — for a hosted collector's auth token",
    )
    export.add_argument(
        "--service-name",
        default="modelmri",
        help="service.name on the exported resource (default: modelmri)",
    )
    export.add_argument(
        "--dry-run",
        action="store_true",
        help="print the OTLP body and send nothing",
    )

    sub.add_parser("where", help="Print every directory ModelMRI reads or writes")

    # `pip install` cannot run this for you: a wheel is an archive and pip does
    # not execute code from it, which is the whole difference between a wheel
    # and an sdist. So the capability check lives here and on the serve banner,
    # where it can also give a different answer to "can I open a recording"
    # than to "can I load a 7B model".
    sub.add_parser(
        "doctor",
        help="Report what this machine can and cannot run, and why",
    )

    # Nested subcommands, and the only place in this CLI that has them. The
    # action expert is a second environment with its own lifecycle — build it,
    # run it, ask about it — and flattening that into `modelmri policy-install`
    # would hide that these three act on one thing.
    policy_parser = sub.add_parser(
        "policy",
        help="The robot action expert: its own environment and its own process",
    )
    policy_sub = policy_parser.add_subparsers(dest="policy_command")

    policy_install = policy_sub.add_parser(
        "install",
        help="Build the action expert's separate environment (about 6 GB)",
    )
    policy_install.add_argument(
        "--force", action="store_true", help="rebuild even if one already exists"
    )
    policy_install.add_argument(
        "--yes", action="store_true", help="skip the confirmation prompt"
    )

    policy_start = policy_sub.add_parser(
        "start", help="Start the policy sidecar and wait until it can answer"
    )
    policy_start.add_argument(
        "--repo", default="", help="a policy to load once it is up, e.g. lerobot/…"
    )
    policy_start.add_argument(
        "--device", default="", help="cuda, cuda:1 or cpu (default: whatever it finds)"
    )
    policy_start.add_argument(
        "--no-wait",
        dest="wait",
        action="store_false",
        help="return once it is serving instead of holding the terminal",
    )
    policy_start.add_argument(
        "--yes",
        action="store_true",
        help="start it even when this machine's VRAM says two copies will not fit",
    )

    policy_status = policy_sub.add_parser(
        "status", help="Say whether an action expert is installed and running"
    )
    policy_status.add_argument("--json", action="store_true", help="machine-readable")

    scanner = sub.add_parser(
        "scan",
        help="Look inside weights for anything that executes on load",
    )
    scanner.add_argument("path", help="a checkpoint, or a directory of them")
    scanner.add_argument("--json", action="store_true", help="machine-readable")
    scanner.add_argument(
        "--limit",
        type=int,
        default=200,
        help="how many files a directory walk may read (default 200)",
    )

    remove = sub.add_parser(
        "uninstall", help="Remove everything ModelMRI has written to this machine"
    )
    remove.add_argument(
        "--yes", action="store_true", help="skip the confirmation prompt"
    )
    remove.add_argument(
        "--models",
        action="store_true",
        help="also delete the HuggingFace model cache (shared with other tools)",
    )

    args = parser.parse_args()
    if args.command == "doctor":
        from . import doctor as _doctor

        # SystemExit, not a bare return of an int: `main` is annotated -> None
        # and every other exit code in this file is raised, not returned. A
        # function that returns None on most paths and 1 on one is a function
        # whose caller has to know which.
        raise SystemExit(_doctor.write_to())

    if args.command == "serve":
        import uvicorn

        # Measured on THIS machine at startup, every time. A user who lands on
        # a page saying "no model loaded" should already know whether that is
        # a choice or a limit.
        from . import doctor as _doctor

        report = _doctor.check()
        # INTENT, not fact. This said "serving on http://…" before anything
        # had bound, so a port that could never work printed a success line
        # and then a traceback — and an occupied port did the same. uvicorn
        # prints its own "Uvicorn running on …" once the socket is actually
        # listening, and that is the line that is true.
        print(f"ModelMRI {__version__} starting on http://{args.host}:{args.port}")
        print(f"  {_doctor.one_line(report)}")
        for blocker in report.blockers:
            print(f"  PROBLEM  {blocker}", file=sys.stderr)
        try:
            uvicorn.run(
                "modelmri.server:create_app",
                factory=True,
                host=args.host,
                port=args.port,
            )
        except KeyboardInterrupt:
            print("\nstopped.")
            return
        except OSError as err:
            # The answer `open` already gives for the same failure, suggestion
            # included. `serve` let this escape as a traceback, so the two
            # commands answered one situation two different ways.
            # Bound to locals so this fits the one-line shape `open` uses at
            # the top of this file. The leak scanner exempts a line that
            # carries `file=sys.stderr`, and splitting the call across lines
            # left the `{err}` on a line of its own — flagged, correctly, since
            # from the scanner's side it could have been going anywhere.
            host, port = args.host, args.port
            print(f"modelmri: cannot listen on {host}:{port} — {err}", file=sys.stderr)
            print("  another ModelMRI may be running; try --port 5901", file=sys.stderr)
            raise SystemExit(3) from None
    elif args.command == "open":
        from pathlib import Path

        from . import session

        target = Path(args.file).expanduser()
        if not target.is_file():
            print(f"modelmri: no such file: {target}", file=sys.stderr)
            raise SystemExit(2)

        # A circuit-tracer attribution graph is a different file entirely, so
        # it takes its own reader and its own banner. Routed by extension
        # rather than by sniffing: a `.pt` that turns out not to be a graph is
        # refused by `circuit.read` with a sentence about what it actually
        # holds, which is more useful than a gzip error from `session.parse`.
        if target.suffix.lower() in (".pt", ".pth"):
            raise SystemExit(
                open_graph(
                    target,
                    host=args.host,
                    port=args.port,
                    browser=not args.no_browser,
                )
            )

        # Parse before starting anything. Someone who was sent the wrong file
        # should get one sentence, not a server they then have to shut down.
        try:
            parsed = session.parse(target.read_bytes())
        except session.SessionError as err:
            print(f"modelmri: {err}", file=sys.stderr)
            raise SystemExit(2) from err

        note = (parsed.meta.get("note") or "").strip()
        print(f"ModelMRI {__version__} — opening {target.name}")
        print(f"  model     {parsed.meta.get('model') or 'unknown'}")
        if note:
            print(f"  note      {note}")
        print(
            f"  contains  {len(parsed.tokens)} tokens, "
            f"{len(parsed.attention)} attention maps"
        )
        print("  no model will be loaded — this is a recording\n")

        serve_viewer(
            target, host=args.host, port=args.port, browser=not args.no_browser
        )
        return
    elif args.command == "inspect":
        raise SystemExit(inspect_session(args.file, as_json=args.json))
    elif args.command == "diff":
        raise SystemExit(
            diff_sessions(args.a, args.b, fail_over=args.fail_over, as_json=args.json)
        )
    elif args.command == "experiments":
        raise SystemExit(
            compare_experiments(
                args.before,
                args.after,
                metric=args.metric,
                higher_is_better=args.higher_is_better,
                dataset=args.dataset,
                floor=args.floor,
                fail_on_worse=args.fail_on_worse,
                as_json=args.json,
            )
        )
    elif args.command == "sweep" and args.resume:
        raise SystemExit(resume_sweep(args.resume, as_json=args.json))
    elif args.command == "sweep":
        if not args.prompts:
            print(
                "sweep needs a prompt file, or --resume with a saved sweep id "
                "(`modelmri sweeps` lists them).",
                file=sys.stderr,
            )
            raise SystemExit(2)
        if not args.model:
            print("sweep needs --model.", file=sys.stderr)
            raise SystemExit(2)
        raise SystemExit(
            run_sweep(
                args.prompts,
                model=args.model,
                metric=args.metric,
                baseline=args.baseline,
                layer=args.layer,
                max_new_tokens=args.max_new_tokens,
                out_dir=args.out_dir,
                jsonl=args.jsonl,
                yes=args.yes,
                as_json=args.json,
            )
        )
    elif args.command == "mcp":
        from . import mcp_server

        raise SystemExit(mcp_server.serve(attach=args.attach))
    elif args.command == "check":
        raise SystemExit(check_trace(args))
    elif args.command == "verify":
        raise SystemExit(verify_session(args.file, as_json=args.json))
    elif args.command == "audit":
        raise SystemExit(audit_dataset(args.dataset, as_json=args.json))
    elif args.command == "policy":
        raise SystemExit(policy_command(args))
    elif args.command == "scan":
        raise SystemExit(scan_weights(args.path, as_json=args.json, limit=args.limit))
    elif args.command == "models":
        raise SystemExit(list_models())
    elif args.command == "traces":
        raise SystemExit(list_traces())
    elif args.command == "export":
        raise SystemExit(
            export_trace(
                args.trace_id,
                endpoint=args.otlp,
                headers=args.header,
                service_name=args.service_name,
                dry_run=args.dry_run,
            )
        )
    elif args.command == "uninstall":
        raise SystemExit(uninstall(yes=args.yes, models=args.models))
    elif args.command == "where":
        from . import paths

        info = paths.describe()
        platform = info.pop("platform")
        width = max(len(k) for k in info)
        print(f"ModelMRI {__version__} on {platform}")
        print()
        for key, value in info.items():
            if value is None or value == []:
                continue
            if isinstance(value, list):
                value = os.pathsep.join(value)
            print(f"  {key:<{width}}  {value}")
        print()
        print("  Override any of it:")
        print("    MODELMRI_HOME        all of the above under one directory")
        print("    MODELMRI_MODELS_HOME where downloaded models go")
        print("    MODELMRI_MODELS_DIR  extra places to look for your models")
        print("    MODELMRI_TRACE_DIR   where undelivered traces are written")
        print("    HF_HOME/HF_HUB_CACHE where models download (HuggingFace's)")
    else:
        parser.print_help()


if __name__ == "__main__":
    # `python -m modelmri.cli` used to import this module, define `main`, and
    # exit 0 without calling it. MEASURED: `doctor --help`,
    # `totally-bogus --nonsense` and `check <missing>.json --no-errors` each
    # returned rc=0 with zero bytes on stdout AND stderr — and the third of
    # those is a CI gate, where exiting 0 without running is indistinguishable
    # from running and passing.
    #
    # `modelmri/__main__.py` routes the conventional spelling to the same
    # `main`; this is the one directory down, kept because the wrong guess
    # should be wrong loudly rather than silently.
    raise SystemExit(main())
