"""Does tests/test_custom.py actually test anything?

    uv run python tests/mutation_check.py

A test that stays green against broken code is decoration. This breaks one
behaviour of modelmri/custom.py at a time and asserts that the named test
notices. Run it after changing either file.

Not named test_*.py on purpose: it rewrites a source file, so pytest must not
collect it into a normal run. The source is restored in a `finally` and the
restoration is asserted.

Two of these mutations were survivors when first written, and both were the
test's fault, not the harness's:

  * "hooks are not removed" originally asserted the NEXT run reported three
    layers. It always did — a leaked hook closes over the previous run's list,
    so it appends out of sight. The test now asserts the invariant directly.
  * "discovery imports the files it finds" was mutated with a line that
    imported importlib rather than the candidate, so nothing broke and the
    test looked complicit. The mutation was wrong, not the test.

An anchor that no longer matches counts as a FAILURE, not a skip: a mutation
harness that silently stops mutating is the exact thing it exists to prevent.
"""

import pathlib
import subprocess
import sys

SRC = pathlib.Path("modelmri/custom.py")
ORIGINAL = SRC.read_text(encoding="utf-8")

MUTATIONS = [
    (
        "non-finite values are not filtered out",
        "    good = f[finite]",
        "    good = f",
        "test_nonfinite_values_are_counted_and_do_not_poison_the_row",
    ),
    (
        "saturation reported for every activation, not just bounded ones",
        "    if kind in _BOUNDED:",
        "    if True:",
        "test_saturation_is_only_reported_for_bounded_activations",
    ),
    (
        "hooks are not removed when the forward pass raises",
        "    finally:\n        for h in handles:\n            h.remove()\n"
        "        if was_training:\n            model.train()",
        "    finally:\n        pass",
        "test_hooks_are_removed_even_when_the_forward_pass_fails",
    ),
    (
        "training mode is not restored",
        "        for h in handles:\n            h.remove()\n"
        "        if was_training:\n            model.train()",
        "        for h in handles:\n            h.remove()",
        "test_training_mode_is_restored_after_inspection",
    ),
    (
        "dead units counted as non-zero instead of zero",
        '    out["pct_zero"] = round(float((good == 0).float().mean()) * 100, 2)',
        '    out["pct_zero"] = round(float((good != 0).float().mean()) * 100, 2)',
        "test_a_dead_relu_is_counted",
    ),
    (
        "an oversized input shape is allocated instead of refused",
        "            if n > 64_000_000:",
        "            if n > 10**18:",
        "test_absurd_shapes_are_refused_before_allocating",
    ),
    (
        "adapter discovery imports the files it finds",
        "            if not _MODULE_LEVEL_LOAD.search(head):",
        "            _import_adapter(path)\n"
        "            if not _MODULE_LEVEL_LOAD.search(head):",
        "test_discovery_finds_adapters_without_importing_them",
    ),
    (
        "discovery matches `def load(self, ...)` methods again",
        '_MODULE_LEVEL_LOAD = re.compile(r"^def\\s+load\\s*\\(\\s*(?!self\\b)", re.MULTILINE)',
        '_MODULE_LEVEL_LOAD = re.compile(r"def\\s+load\\s*\\(")',
        "test_a_load_method_is_not_an_adapter",
    ),
    (
        "test suites are scanned for adapters again",
        '        "tests",\n        "test",\n        ".pytest_cache",',
        "",
        "test_a_test_suite_is_not_scanned_for_adapters",
    ),
    (
        "an Embedding model is fed floats",
        "            if _wants_integer_input(model):",
        "            if False:",
        "test_an_embedding_model_gets_integer_input",
    ),
    (
        "paths outside the allowed roots are accepted",
        '        raise AdapterError(\n            f"{p} is outside the directories',
        "        pass\n    if False:\n        raise AdapterError("
        '\n            f"{p} is outside the directories',
        "test_refuses_a_path_outside_the_allowed_roots",
    ),
    (
        "a state_dict is accepted instead of explained",
        "    if isinstance(obj, dict):",
        "    if False:",
        "test_a_state_dict_checkpoint_is_refused_with_the_reason",
    ),
]

failures = []
try:
    for label, old, new, test in MUTATIONS:
        if old not in ORIGINAL:
            print(f"  [ANCHOR GONE] {label}")
            failures.append(label)
            continue
        SRC.write_text(ORIGINAL.replace(old, new, 1), encoding="utf-8")
        r = subprocess.run(
            [
                sys.executable,
                "-m",
                "pytest",
                f"tests/test_custom.py::{test}",
                "-q",
                "--no-header",
                "-x",
            ],
            capture_output=True,
            text=True,
        )
        caught = r.returncode != 0
        print(f"  [{'caught' if caught else 'MISSED'}] {label}")
        if not caught:
            failures.append(label)
finally:
    SRC.write_text(ORIGINAL, encoding="utf-8")

assert SRC.read_text(encoding="utf-8") == ORIGINAL, "source not restored!"
print(f"\n{len(MUTATIONS) - len(failures)}/{len(MUTATIONS)} mutations caught")
sys.exit(1 if failures else 0)
