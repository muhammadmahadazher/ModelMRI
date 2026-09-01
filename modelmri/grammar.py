"""What constrained decoding cost you, per step.

Ollama, llama.cpp (GBNF), vLLM and LM Studio all ship constrained decoding as a
black box: you get valid JSON and no idea what it cost. Nobody shows the mask's
effect on the distribution, and "structured output mode makes the model dumber"
is a widespread complaint with no instrument pointed at it.

This is that instrument. Per step it records the model's UNCONSTRAINED top-k
beside the grammar's allowed set, how much probability mass the mask deleted,
and whether the token the model most wanted was forbidden. A step where the
mask removed most of the distribution is where your structured output stopped
being the model's answer and started being the schema's.

MEASURED with a two-field object schema: a single-figure number of tokens is
legal before anything is written (whitespace and the brace variants), and
about as few at the key position — over 99.9% of the vocabulary masked — while
inside a free string value almost the whole vocabulary is legal. The contrast
IS the finding, and neither number is visible from the completion alone.

(An earlier draft of this docstring quoted 2 and 50,174. Those came from a run
with a broken token list — see `tokenizer_data` — and are exactly the kind of
number this project refuses to leave standing: measured once, from a bug, and
written down as a property.)

## The grammar is not ours, on purpose

`lm-format-enforcer` compiles the schema. This module does not write a
token-level grammar compiler and never will: a wrong mask is an invisible
failure in a tool whose whole premise is not shipping invisible wrong answers.
You would get valid-looking JSON built from a mask nobody checked.

What IS written here is the ~40 lines of glue that turn its enforcer into a
`LogitsProcessor`. Their own `integrations.transformers` module cannot be used:
it imports `transformers.tokenization_utils.PreTrainedTokenizerBase`, which
does not exist in transformers 5.x, and raises "transformers is not installed"
on a machine where transformers is installed and working. That is the same
defect class `modelmri_record.verify` exists to catch — an integration layer
reading an attribute that moved.

## The enforcer must see every step

`TokenEnforcer` memoises per prefix and expects to be asked about each position
in order. Handing it an arbitrary prefix it never walked returns the ROOT
state's answer, silently — measured, asking about `{"name": "` cold gave the
same two tokens as asking about the empty string. A `LogitsProcessor` is called
once per step, which is exactly the access pattern it wants, and
`_STEP_ORDER_NOTE` below is why nothing here offers a "just check this prefix"
helper.

## The deleted mass is measured under the sampler that was running

Probability is read from the pre-mask distribution at that step, so
`deleted_mass` is what the mask actually removed from what the model was going
to do — not from a temperature-1 distribution it never had. Temperature and
top-k travel with the record for that reason.

That claim used to be FALSE, and the wiring is what makes it true.
`generate`'s sampling warpers are appended AFTER any custom processor
(transformers 5.13 `generation/utils.py:1244-1290`, above a literal
`# TODO (joao): find a strategy to specify the order of the processors`), so a
recorder handed in as `logits_processor=[...]` sees pre-temperature,
pre-top-k logits and would report the mass removed from a distribution nobody
sampled from. So `MaskRecorder` OWNS the temperature — it divides before it
softmaxes and before it masks, exactly as `TemperatureLogitsWarper` does — and
`runtime.generate_stream` neutralises HF's own warpers for the constrained
run. The recorder is then genuinely the last thing to touch the logits, and
the sentence above describes what happened.

Two consequences worth stating out loud. A caller that passes a temperature
here and ALSO leaves HF's own `temperature` in place applies it twice; nothing
but `runtime.generate_stream` should be constructing these. And `top_k`
silently defaults to 50 in transformers, so an un-neutralised constrained run
would additionally be truncating the very distribution it was reporting on.

## A schema is compiled BEFORE anything is generated

`JsonSchemaParser(schema)` only compiles the ROOT. A property's value schema
is compiled lazily, at the moment the model emits the `:` after that key — and
if it cannot be compiled the enforcer swallows the exception and sets the
allowed set to {EOS}. MEASURED against `{"a": {"type": "widget"}}`: no error at
construction, and then a 200 whose body is `{"a"` — truncated JSON produced by
a mask nobody checked. The other half of the same trap raises mid-generation
on the `generate` worker thread, where the exception dies and the consumer
blocks on the streamer's timeout instead.

`validate_schema` therefore walks the whole tree and compiles every sub-schema
up front. It is not an optimisation; it is the only place either failure can
be turned into an answer.

A third way into the same trap needs a separate walk, because compiling cannot
find it: JSON Schema allows a bare `true` where a schema goes, the enforcer's
object model accepts one, and its compiler reads `.anyOf` off it. MEASURED,
`{"type": "object", "additionalProperties": true}` — a 200 whose body is ` { "`
and whose mask permitted nothing but end-of-sequence from step 2. See
`_non_schema_children`.

## What is enforced and what is merely written down

The compiler reads sixteen schema attributes and no others, so `minimum`,
`maximum`, `multipleOf`, `uniqueItems`, `format` and `patternProperties` are
accepted, ignored, and never mentioned again — MEASURED, an `"age"` bounded to
`maximum: 5` came back as `30`, parsing cleanly beside a receipt that said the
mask was enforced. Refusing those schemas would turn every schema written for
validation into a 400, so they are DISCLOSED instead, in `Trace.notes`,
alongside the array cap. `NOT_COMPILED_KEYWORDS` is that list.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

from .errors import BadRequest, Refusal

# How many unconstrained candidates are kept per step. Enough to see what the
# model wanted and what it settled for; not so many that a 200-token
# completion becomes megabytes.
TOP_K = 8

# Above this the recorder stops rather than growing without bound. Reported,
# never silent -- a truncated trace read as a complete one is the failure this
# project keeps refusing.
MAX_STEPS = 2_000

_STEP_ORDER_NOTE = (
    "the enforcer is memoised per prefix and must be asked about every step in "
    "order; asking about an arbitrary prefix returns the root state's answer "
    "with nothing saying so"
)

#: What `"json_object"` means, spelled out rather than left to a default.
#:
#: MEASURED against the installed enforcer: `None` and `{}` both mean "any
#: JSON VALUE" to it (`jsonschemaparser.py:14,44` expands them to a union of
#: every type), so a bare string or a bare `null` satisfies them. A client
#: asking for a JSON object and receiving `"hello"` has been answered with
#: something else. `{"type": "object"}` is the one that keeps keys free and
#: values free while still requiring an object.
ANY_JSON_OBJECT = {"type": "object"}

#: The `type` values the enforcer's schema compiler can build a mask for, read
#: off its own `get_parser`. Named in refusals so a rejected schema comes back
#: with the list rather than with a bare "no".
COMPILABLE_TYPES = (
    "string",
    "integer",
    "number",
    "boolean",
    "null",
    "object",
    "array",
)

#: The keywords it accepts INSTEAD of a `type`. A sub-schema carrying neither
#: has nothing to build a mask from, which is the most common way a schema
#: written for validation fails to be a schema you can decode under.
COMPILABLE_KEYWORDS = ("enum", "const", "$ref", "anyOf", "allOf", "oneOf")

#: Real JSON Schema constraints the grammar compiler NEVER READS.
#:
#: Read off the compiler the same way `COMPILABLE_TYPES` is, rather than
#: remembered: the complete set of schema attributes `jsonschemaparser` touches
#: is `additionalProperties, allOf, anyOf, enum, extras (which is where `const`
#: lands), items, maxItems, maxLength, minItems, minLength, oneOf, pattern,
#: properties, ref, required, type`. Everything below is a keyword a caller can
#: legitimately write that is simply not in that list.
#:
#: MEASURED against the installed 0.11.3 with
#: `{"age": {"type": "integer", "minimum": 0, "maximum": 5}}`: after the model
#: has emitted `{"age":3` the mask still permits every digit, so `30` is legal
#: and the bound was never compiled. The completion parses, the receipt says
#: what the mask cost, and nothing anywhere says the bound was dropped -- which
#: is the more misleading direction than the array cap below, because the
#: caller believes what they wrote was applied.
#:
#: DISCLOSED rather than refused. The schema IS enforceable, just not entirely,
#: and a caller told which half can validate the other half themselves. A
#: refusal would turn every schema written for validation into a 400.
NOT_COMPILED_KEYWORDS = (
    "contains",
    "dependentRequired",
    "dependentSchemas",
    "else",
    "exclusiveMaximum",
    "exclusiveMinimum",
    "format",
    "if",
    "maxProperties",
    "maximum",
    "minProperties",
    "minimum",
    "multipleOf",
    "not",
    "patternProperties",
    "prefixItems",
    "propertyNames",
    "then",
    "uniqueItems",
)

# Where a schema tree branches. Walked depth-first by `validate_schema`,
# because every one of these is a place the enforcer defers compilation to
# decode time -- which is the one moment a failure cannot be reported.
#
# `patternProperties` is deliberately NOT here, and used to be. Compiling its
# children asserted a support the compiler does not have: the keyword appears
# nowhere in `jsonschemaparser.py`, so a sub-schema under it is never built and
# a bad one was being turned into a 400 for a schema that would have decoded
# fine. It is in NOT_COMPILED_KEYWORDS instead, where it is true.
_CHILD_MAPS = ("properties", "$defs", "definitions")
_CHILD_NODES = ("items", "additionalProperties")
_CHILD_LISTS = ("anyOf", "allOf", "oneOf")

# How much of the schema the receipt carries. Bounded because a generated model
# schema can be very large and the receipt already carries a row per step; see
# `Trace.schema_truncated` for why the cut is published rather than silent.
MAX_SCHEMA_CHARS = 2_000


class GrammarError(BadRequest):
    """This schema cannot be enforced honestly, and the message says why."""


def _enforcer_module():
    """`lm-format-enforcer`, or a refusal naming the extra that provides it."""
    try:
        import lmformatenforcer  # noqa: F401
        from lmformatenforcer import (
            JsonSchemaParser,
            TokenEnforcer,
            TokenEnforcerTokenizerData,
        )
    except ImportError as err:
        raise Refusal(
            "Constrained decoding needs `lm-format-enforcer`, which is an "
            "optional extra: `pip install modelmri[grammar]`. It is optional "
            "because it is the only way to get a correct token-level mask, and "
            "ModelMRI will not hand-write one -- a wrong mask is an invisible "
            "failure, and you would get valid-looking output built from it."
        ) from err
    return JsonSchemaParser, TokenEnforcer, TokenEnforcerTokenizerData


def _installed_version() -> str:
    """The enforcer's version, for a refusal that has to name it."""
    import importlib.metadata

    try:
        return importlib.metadata.version("lm-format-enforcer")
    except Exception:  # pragma: no cover - a source checkout has no metadata
        return "an unknown version"


# The pre-flight compiler, or the reason there is not one. Memoised because
# the answer cannot change inside a process and the self-test below costs two
# schema compilations. The IMPORT is deliberately outside the memo: the
# missing-extra refusal has to fire on every call, or a process that once had
# the enforcer would keep answering as though it still did.
_PREFLIGHT: tuple | None = None


def _preflight():
    """The enforcer's own schema compiler, PROVEN to still work here.

    `get_parser` and `JsonSchemaObject` are deep imports of symbols the
    library does not export. That is a real coupling and it is the lesser of
    two evils: the alternative is hand-writing a table of which constructs
    compile, which duplicates the function that actually builds the mask and
    drifts away from it silently. Borrowing the real one cannot drift; it can
    only stop importing, which is loud.

    So it is checked rather than assumed, once, against two schemas whose
    answers are known: one that must compile and one that must not. A checker
    that says yes to everything would pass a positive-only self-test and then
    wave through exactly the schemas this exists to catch, so both directions
    are asserted. A version bump that moves the symbol, changes its signature
    or changes its verdict lands here as a refusal naming the version --
    never as a schema being blamed for the checker.
    """
    JsonSchemaParser, _, _ = _enforcer_module()
    global _PREFLIGHT
    if _PREFLIGHT is not None:
        return _PREFLIGHT

    try:
        from lmformatenforcer.external.jsonschemaobject import JsonSchemaObject
        from lmformatenforcer.jsonschemaparser import get_parser
    except ImportError as err:
        raise Refusal(_no_preflight()) from err

    good = {"type": "object", "properties": {"a": {"type": "string"}}}
    bad = {"type": "object", "properties": {"a": {"type": "modelmri-not-a-type"}}}
    try:
        for schema, must_compile in ((good, True), (bad, False)):
            root = JsonSchemaParser(schema)
            node = schema["properties"]["a"]
            try:
                get_parser(root, JsonSchemaObject(**node))
            except Exception:
                compiled = False
            else:
                compiled = True
            if compiled is not must_compile:
                raise RuntimeError("the pre-flight self-test did not agree")
    except Refusal:
        raise
    except Exception as err:
        raise Refusal(_no_preflight()) from err

    _PREFLIGHT = (get_parser, JsonSchemaObject)
    return _PREFLIGHT


def _no_preflight() -> str:
    return (
        f"this server cannot check a JSON schema before enforcing it: the "
        f"installed `lm-format-enforcer` ({_installed_version()}) no longer "
        f"answers the schema compiler this check is written against. Without "
        f"it a schema could only be validated by generating under it, and an "
        f"unenforceable one collapses the mask to end-of-sequence mid-object "
        f"and hands back a fragment. Refusing rather than finding that out "
        f"during the completion. `pip install 'lm-format-enforcer>=0.10'` "
        f"reinstalls a version this was written against."
    )


def _array_cap() -> int | None:
    """How many items an array with no `maxItems` will actually be allowed.

    ASKED, not remembered. The enforcer's own default is 20, but
    `CharacterLevelParserConfig` reads `LMFE_MAX_JSON_ARRAY_LENGTH` from the
    environment at construction, so the number that will apply on THIS machine
    is the one that machine reports. `CharacterLevelParserConfig` is a public
    export; None when it cannot be read, which the note below says out loud
    rather than printing a default nobody checked.
    """
    try:
        from lmformatenforcer import CharacterLevelParserConfig

        return int(CharacterLevelParserConfig().max_json_array_length)
    except Exception:  # pragma: no cover - a moved or renamed config field
        return None


def _unbounded_arrays(schema: dict) -> list:
    """JSON pointers of every array this schema does not bound itself.

    Each one is a place the compiler will apply a limit the caller never
    wrote. `maxItems` is the only thing that overrides it from inside the
    schema (`jsonschemaparser.py:647` falls back to the config only when the
    schema states nothing), so its absence is exactly the condition.
    """
    found = []
    if isinstance(schema, dict) and schema.get("type") == "array":
        if "maxItems" not in schema:
            found.append("the schema's own root")
    for node, pointer in _subschemas(schema):
        if node.get("type") == "array" and "maxItems" not in node:
            found.append(pointer)
    return found


def _subschemas(schema: dict, pointer: str = "", seen: set | None = None):
    """Every sub-schema in `schema`, depth-first, with its JSON pointer.

    `seen` is keyed on object identity rather than content: a schema is free
    to reuse the same dict in two places, and a `$defs` entry that references
    itself would otherwise walk forever. Identity stops the cycle without
    refusing the perfectly ordinary case of one shared definition.
    """
    if seen is None:
        seen = set()
    if id(schema) in seen:
        return
    seen.add(id(schema))

    for key in _CHILD_MAPS:
        block = schema.get(key)
        if isinstance(block, dict):
            for name, node in block.items():
                if isinstance(node, dict):
                    here = f"{pointer}/{key}/{name}"
                    yield node, here
                    yield from _subschemas(node, here, seen)
    for key in _CHILD_NODES:
        node = schema.get(key)
        if isinstance(node, dict):
            here = f"{pointer}/{key}"
            yield node, here
            yield from _subschemas(node, here, seen)
    for key in _CHILD_LISTS:
        block = schema.get(key)
        if isinstance(block, list):
            for i, node in enumerate(block):
                if isinstance(node, dict):
                    here = f"{pointer}/{key}/{i}"
                    yield node, here
                    yield from _subschemas(node, here, seen)


def _non_schema_children(schema: dict, pointer: str = "", seen: set | None = None):
    """Every child position holding something the compiler cannot compile.

    JSON Schema draft-6 and later allow a bare `true` or `false` where a schema
    goes, and `lm-format-enforcer`'s own object model accepts one: `items` and
    `additionalProperties` are typed `Union[..., bool, None]` and every
    `properties` value is `Union[JsonSchemaObject, bool]`. Its COMPILER does
    not. `get_parser` reads `.anyOf` off whatever it is handed, so a boolean
    arrives as `'bool' object has no attribute 'anyOf'` inside
    `TokenEnforcer._compute_allowed_tokens`'s blanket `except Exception`, and
    the allowed set becomes {EOS}.

    MEASURED against the installed 0.11.3, all four collapsing mid-value behind
    a 200: `{"type": "object", "additionalProperties": true}` at the `:` after
    the first key, `{"type": "array", "items": true}` at the opening `[`,
    `{"type": "object", "properties": {"a": true}}` at the `:` after `"a"`, and
    a `$defs` entry of `true` at the `:` after the key that `$ref`s it. That is
    exactly the failure `validate_schema`'s docstring says it makes
    unreachable, and `_subschemas` walked straight past all four because it
    yields a child only `if isinstance(node, dict)`.

    Two shapes are deliberately NOT yielded, because for them the compiler's
    behaviour and the schema's meaning coincide well enough that a 400 would be
    wrong:

    * `additionalProperties: false`. The compiler tests the keyword for TRUTH,
      so `false` never reaches `get_parser` -- it falls through to the
      any-JSON default. What that does instead (permit arbitrary keys under a
      schema that asked for none) is a `Trace.notes` disclosure; see
      `_free_dictionaries`.
    * `additionalProperties` of any kind on a node that declares `properties`.
      `ObjectParsingState.is_dictionary` is `properties is None`, so the
      keyword is never read there at all -- and `{"properties": {...},
      "additionalProperties": false}` is what OpenAI strict mode emits.

    A non-array node carrying `items` gets no such exemption, and the
    difference is about how common the mistake is rather than about the
    library. `{"properties": ..., "additionalProperties": false}` is what half
    the schemas in the world look like, so refusing it would be refusing
    ordinary correct input; `{"type": "object", "items": true}` is a schema
    somebody got wrong either way, and a 400 naming the position is a better
    answer than a mask that quietly ignores it.
    """
    if seen is None:
        seen = set()
    if id(schema) in seen:
        return
    seen.add(id(schema))

    for key in _CHILD_MAPS:
        block = schema.get(key)
        if isinstance(block, dict):
            for name, node in block.items():
                here = f"{pointer}/{key}/{name}"
                if isinstance(node, dict):
                    yield from _non_schema_children(node, here, seen)
                else:
                    yield here, node

    items = schema.get("items")
    if isinstance(items, dict):
        yield from _non_schema_children(items, f"{pointer}/items", seen)
    elif items is not None:
        # A list here is JSON Schema's tuple form, which `ListParsingState`
        # hands to `get_parser` whole and which has no `.anyOf` either.
        yield f"{pointer}/items", items

    extra = schema.get("additionalProperties")
    if isinstance(extra, dict):
        yield from _non_schema_children(extra, f"{pointer}/additionalProperties", seen)
    elif extra and "properties" not in schema:
        yield f"{pointer}/additionalProperties", extra

    for key in _CHILD_LISTS:
        block = schema.get(key)
        if isinstance(block, list):
            for i, node in enumerate(block):
                here = f"{pointer}/{key}/{i}"
                if isinstance(node, dict):
                    yield from _non_schema_children(node, here, seen)
                else:
                    yield here, node


def _dropped_constraints(schema: dict) -> list:
    """Every place this schema states a constraint the compiler never reads.

    The mirror of `_unbounded_arrays`, and the more misleading half of the
    pair. That one discloses a limit the COMPILER added; this one discloses a
    limit the CALLER wrote and the compiler discarded -- which is worse,
    because the caller believes it was applied and the completion parses.
    """
    found: list = []

    def _at(node: dict, pointer: str) -> None:
        names = [k for k in NOT_COMPILED_KEYWORDS if k in node]
        if names:
            found.append((pointer or "the schema's own root", names))

    if not isinstance(schema, dict):
        return found
    _at(schema, "")
    for node, pointer in _subschemas(schema):
        _at(node, pointer)
    return found


def _free_dictionaries(schema: dict) -> list:
    """Pointers where `additionalProperties: false` permits anything at all.

    `ObjectParsingState` reads `additionalProperties` only when the node
    declares no `properties`, and reads it for TRUTH -- so `false` falls
    through to `ANY_JSON_OBJECT_SCHEMA`. MEASURED: under
    `{"type": "object", "additionalProperties": false}` the mask permits a free
    key and then the whole JSON-value set after the `:`. The caller wrote "no
    properties at all" and got "any key, any value", which is the exact
    inversion of what they asked for.
    """
    found = []
    nodes = [(schema, "the schema's own root")] if isinstance(schema, dict) else []
    if isinstance(schema, dict):
        nodes.extend((node, pointer) for node, pointer in _subschemas(schema))
    for node, pointer in nodes:
        if node.get("additionalProperties") is False and "properties" not in node:
            found.append(pointer)
    return found


def _defined_names(schema: dict) -> set:
    names = set()
    for key in ("$defs", "definitions"):
        block = schema.get(key)
        if isinstance(block, dict):
            names.update(block)
    return names


def _fault(node, root: dict) -> str:
    """Why this sub-schema cannot be compiled, in this project's own words.

    Read off the CALLER'S OWN schema, never off the exception. `errors.py` is
    explicit that a published sentence may not interpolate a caught
    exception's `str`, args or repr -- those are machinery talking to itself
    and they carry whatever the library underneath felt like carrying -- and
    `MaskRecorder.__init__` has held that line for this module since it was
    written. Every value quoted below came out of the request body, so
    quoting it is the same thing as telling somebody what they sent.

    The order matters: the first condition that matches is reported, so the
    most specific diagnosis wins. When none matches, the caller still gets a
    true sentence naming where the compiler stopped -- a vague-but-honest
    answer, rather than a confident wrong one.
    """
    if isinstance(node, bool):
        return (
            f"it is `{str(node).lower()}`. JSON Schema draft-6 and later allow "
            f"a bare boolean where a schema goes, and the grammar compiler "
            f"does not read one -- it asks the value for its 'anyOf', a "
            f"boolean has none, and the allowed set collapses to "
            f"end-of-sequence at that position rather than raising. Write the "
            f'sub-schema out: `true` is `{{"type": ...}}` for whatever you '
            f"will accept there."
        )
    if not isinstance(node, dict):
        return (
            f"it is {type(node).__name__}, and a JSON schema is an object. "
            f"The grammar compiler builds a parser out of a schema object and "
            f"has nothing to build one from here."
        )

    # `type` is a string or a list of them; anything else is a shape mistake
    # rather than an unsupported value, and says so differently.
    kinds = node.get("type")
    named = [kinds] if isinstance(kinds, str) else kinds
    if isinstance(named, list):
        unknown = [k for k in named if k not in COMPILABLE_TYPES]
        if unknown:
            return (
                f"its 'type' names {', '.join(repr(k) for k in unknown)}, and a "
                f"token-level mask can be built for "
                f"{', '.join(COMPILABLE_TYPES)}."
            )
    elif kinds is not None:
        return f"its 'type' is {type(kinds).__name__}, and a 'type' is a string."

    if "$ref" in node:
        target = node["$ref"]
        defined = _defined_names(root)
        if str(target).rsplit("/", 1)[-1] not in defined:
            return (
                f"its '$ref' points at {target!r}, and this schema defines "
                f"{', '.join(sorted(defined)) or 'nothing'} under '$defs' or "
                f"'definitions'."
            )

    if "pattern" in node and ("minLength" in node or "maxLength" in node):
        return (
            "it carries both a 'pattern' and a length bound, and the compiler "
            "builds a state machine for one or the other, not for both at "
            "once. Express the length inside the pattern."
        )
    if "pattern" in node:
        return (
            f"its 'pattern' {node['pattern']!r} uses a regular-expression "
            f"construct that cannot be turned into a finite state machine -- "
            f"back-references and look-around are the usual ones."
        )

    if "enum" in node:
        values = node["enum"]
        if isinstance(values, list):
            types = sorted({type(v).__name__ for v in values})
            if len(types) > 1:
                return (
                    f"its 'enum' mixes {', '.join(types)} in one list, and the "
                    f"compiler builds a mask for an enum of one type."
                )

    if kinds is None and not any(k in node for k in COMPILABLE_KEYWORDS):
        return (
            f"it names no 'type' and none of "
            f"{', '.join(repr(k) for k in COMPILABLE_KEYWORDS)}, so there is "
            f"nothing to build a mask from. A sub-schema that only describes "
            f"itself validates anything, and 'anything' is not a grammar."
        )

    return (
        "the grammar compiler would not accept it. Every 'type' must be one "
        f"of {', '.join(COMPILABLE_TYPES)}, every '$ref' must resolve, and "
        "every value must be the type its keyword expects."
    )


def _unenforceable(pointer: str, node, root: dict) -> str:
    where = f"at {pointer!r}" if pointer else "at its root"
    return (
        f"that JSON schema cannot be compiled into a token-level grammar. The "
        f"compiler refused the sub-schema {where}: {_fault(node, root)} "
        f"Nothing was generated -- a schema this server cannot enforce would "
        f"otherwise stop the completion mid-object and hand back a fragment "
        f"that still looks like an answer."
    )


def validate_schema(schema: dict) -> None:
    """Compile every sub-schema now, so an unenforceable one is a refusal.

    Returns nothing and raises `GrammarError` (a `BadRequest`, so a 400) or
    `Refusal`. See the module docstring for what happens without this: the
    enforcer compiles only the root eagerly, and a nested construct it cannot
    build either collapses the allowed set to {EOS} mid-object behind a 200,
    or raises on the `generate` worker thread where nothing is listening.
    """
    if not isinstance(schema, dict):
        raise GrammarError(
            f"a JSON schema is an object, and this request sent "
            f"{type(schema).__name__}."
        )
    if not schema:
        raise GrammarError(
            "an empty JSON schema constrains nothing -- to the grammar "
            "compiler `{}` means 'any JSON value at all', so a bare string or "
            "a bare null would satisfy it. Send a schema with a 'type', or "
            'ask for `{"type": "json_object"}` if any object will do.'
        )

    JsonSchemaParser, _, _ = _enforcer_module()
    get_parser, JsonSchemaObject = _preflight()

    # A CHILD THAT IS NOT A SCHEMA OBJECT, before anything is compiled.
    #
    # This has to come first because the compile loop below cannot see it:
    # `_subschemas` yields a child only when it is a dict, so a boolean
    # sub-schema was never handed to `get_parser` and this function returned
    # clean on a schema that then collapsed the mask to {EOS} mid-object. See
    # `_non_schema_children` for the measurement and for the two shapes that
    # must keep passing.
    for pointer, node in _non_schema_children(schema):
        raise GrammarError(_unenforceable(pointer, node, schema))

    root_failed = False
    try:
        root = JsonSchemaParser(schema)
    except Exception:
        root_failed = True
        # A root that will not build cannot resolve `$ref`s either, so the
        # locator gets a minimal object schema carrying only this schema's own
        # definitions. It exists to find WHERE, not to decide whether.
        seed: dict = dict(ANY_JSON_OBJECT)
        for key in ("$defs", "definitions"):
            if isinstance(schema.get(key), dict):
                seed[key] = schema[key]
        try:
            root = JsonSchemaParser(seed)
        except Exception as err:
            raise GrammarError(_unenforceable("", schema, schema)) from err

    for node, pointer in _subschemas(schema):
        try:
            get_parser(root, JsonSchemaObject(**node))
        except Exception as err:
            raise GrammarError(_unenforceable(pointer, node, schema)) from err

    if root_failed:
        # Every child compiles, so the fault is the root's own.
        raise GrammarError(_unenforceable("", schema, schema))


class ChoiceTap:
    """A stopping criterion that stops nothing and records what was chosen.

    `MaskRecorder.__call__` runs BEFORE the token is sampled, so it cannot
    know what came out of the step it just described -- and `record_choice`
    only ever writes `steps[-1]`, so it cannot be replayed in a loop
    afterwards either. The one hook inside `generate` that fires after the
    token is appended and before the next processor call is a stopping
    criterion: `_sample` calls it once per iteration with `input_ids` already
    extended, which is exactly the moment `record_choice` is documented to
    want.

    Returning all-False makes it inert -- `StoppingCriteriaList` ORs the
    results, so a criterion that never says stop cannot end a generation.

    Deliberately not a `transformers.StoppingCriteria` subclass: this module
    imports nothing from transformers (see the module docstring on why its
    vendor integration cannot be used), and the list dispatches by call, not
    by type. `runtime.generate_stream` wraps it in the real
    `StoppingCriteriaList`, which is where that dependency belongs.
    """

    def __init__(self, recorder):
        self.recorder = recorder

    def __call__(self, input_ids, scores=None, **kwargs):
        import torch

        self.recorder.record_choice(int(input_ids[0, -1]))
        return torch.zeros(
            input_ids.shape[0], dtype=torch.bool, device=input_ids.device
        )


@dataclass
class Step:
    """One decoding step, with and without the mask."""

    step: int
    # What the model wanted, before the grammar touched anything.
    wanted: str = ""
    wanted_id: int = -1
    wanted_p: float = 0.0
    # Whether the grammar permitted it.
    wanted_was_allowed: bool = True
    # What was actually emitted.
    chosen: str = ""
    chosen_id: int = -1
    # The emitted token's pre-mask probability, or None when it was not among
    # the `top` rows recorded here. It defaulted to 0.0, which reads as "the
    # model gave this token no probability" -- the opposite of the truth for a
    # token the grammar forced through from outside the top few. `top` holds
    # the strongest handful; anything past that is unrecorded, not zero.
    chosen_p: float | None = None
    allowed_count: int = 0
    vocab_size: int = 0
    # Probability the mask removed, from the PRE-MASK distribution at this
    # step under the sampler that was running.
    deleted_mass: float = 0.0
    # The one allowed token was end-of-sequence, so the grammar had nowhere
    # left to go. `validate_schema` exists to make this unreachable -- a
    # sub-schema the enforcer cannot compile collapses the allowed set to
    # exactly this, silently, and the completion ends mid-object behind a 200.
    # Recording it is the tripwire that says the pre-flight is still working,
    # and `allowed_count == 1` alone cannot say it: a forced closing brace is
    # also one token and is exactly what should happen.
    eos_only: bool = False
    top: list = field(default_factory=list)

    @property
    def masked_fraction(self) -> float:
        return 1.0 - (self.allowed_count / self.vocab_size) if self.vocab_size else 0.0

    def to_dict(self) -> dict:
        out = asdict(self)
        out["masked_fraction"] = round(self.masked_fraction, 6)
        return out


@dataclass
class Trace:
    """A whole constrained generation, and what the schema cost it."""

    steps: list = field(default_factory=list)
    schema: str = ""
    # Whether `schema` above is the whole schema or the first
    # `MAX_SCHEMA_CHARS` characters of it, and how long the whole one was.
    #
    # The cut is old; publishing it is not. Nothing put `to_dict()` on the wire
    # until `/v1` started returning this receipt, and a field named `schema`
    # holding a string that is not the schema and does not parse is an unknown
    # rendered as a value -- a reader comparing the receipt against what they
    # sent finds a mismatch with nothing explaining it. MEASURED with a 3,824
    # character schema: the published field ended mid-string at 2,000 and
    # `json.loads` on it raised "Unterminated string".
    schema_truncated: bool = False
    schema_bytes: int = 0
    temperature: float = 0.0
    truncated: int = 0
    vocab_size: int = 0
    # Places where what the grammar enforces and what the schema says are not
    # the same thing. The class started as "a limit the caller never wrote" --
    # the cap the compiler puts on an array with no `maxItems` -- and a list
    # rather than a field because that class was always going to grow. It has:
    # a constraint the caller DID write and the compiler discards
    # (`_dropped_constraints`), and an `additionalProperties: false` the
    # compiler reads as its opposite (`_free_dictionaries`). Carried into
    # `to_dict` and into `means`, because a note only a machine reads is a note
    # nobody reads.
    notes: list = field(default_factory=list)

    @property
    def overridden(self) -> list:
        """Steps where the token the model most wanted was forbidden."""
        return [s for s in self.steps if not s.wanted_was_allowed]

    @property
    def collapsed(self) -> list:
        """Steps where the grammar had nothing left to permit.

        `eos_only` is the parser-collapse signature `Step` documents, and it
        was WRITTEN and never read: nothing turned it into a sentence, a status
        or a verdict, so a completion that ended because the mask ran out came
        back as an ordinary answer that happened not to parse. This is the
        reader.

        An empty allowed set belongs to the same class and is strictly worse.
        It cannot come from the enforcer -- which always appends
        end-of-sequence -- but it CAN come from this module: `__call__` clamps
        allowed ids to the width of the logits row, and a row narrower than
        every allowed id leaves nothing. The step is then all `-inf`, whose
        softmax is NaN, and `eos_only` is False because the set is not the EOS
        set. Flagged here rather than raised, because raising happens on
        `generate`'s worker thread where nothing is listening.
        """
        return [s for s in self.steps if s.eos_only or s.allowed_count == 0]

    def to_dict(self) -> dict:
        return {
            "steps": [s.to_dict() for s in self.steps],
            "schema": self.schema,
            "schema_truncated": self.schema_truncated,
            "schema_bytes": self.schema_bytes,
            "temperature": self.temperature,
            "truncated": self.truncated,
            "vocab_size": self.vocab_size,
            "n_overridden": len(self.overridden),
            "n_collapsed": len(self.collapsed),
            "notes": list(self.notes),
            "means": self.means(),
        }

    def _tail(self) -> list:
        """Sentences that describe the SCHEMA and this record of it.

        Separate from the per-step sentences because they apply just as much to
        a run that generated nothing: the notes describe what the compiler did
        to the schema, and the truncation describes this receipt.
        """
        tail = [*self.notes]
        if self.schema_truncated:
            tail.append(
                f"THE 'schema' FIELD IN THIS RECEIPT IS CUT: the schema that "
                f"was enforced is {self.schema_bytes} characters and this "
                f"carries the first {len(self.schema)}, so the field will not "
                f"parse as JSON. What was enforced is the whole schema that "
                f"was sent, not this abbreviation of it."
            )
        return tail

    def means(self) -> str:
        tail = self._tail()
        if not self.steps:
            # The notes still apply: they describe the SCHEMA, and a request
            # that refused or generated nothing has one just the same.
            return " ".join(["Nothing was generated under this schema.", *tail])
        overridden = self.overridden
        collapsed = self.collapsed
        worst = max(self.steps, key=lambda s: s.deleted_mass)
        parts = [
            f"{len(self.steps)} step(s) under this schema. The mask deleted a "
            f"median of {self._median_deleted():.1%} of the probability at each "
            f"step, measured on the distribution the model actually had at that "
            f"step rather than on a temperature-1 one it never had."
        ]
        if overridden:
            names = ", ".join(repr(s.wanted) for s in overridden[:4])
            parts.append(
                f"AT {len(overridden)} STEP(S) THE TOKEN THE MODEL MOST WANTED "
                f"WAS FORBIDDEN ({names}). That is where the output stopped "
                f"being the model's answer and started being the schema's."
            )
        else:
            parts.append(
                "At no step was the model's first choice forbidden, so the "
                "schema shaped the output without overriding it."
            )
        if collapsed:
            first = collapsed[0]
            what = (
                "no token at all"
                if first.allowed_count == 0
                else "only end-of-sequence"
            )
            also = (
                f", and at {len(collapsed)} step(s) in total"
                if len(collapsed) > 1
                else ""
            )
            parts.append(
                f"AT STEP {first.step} THE GRAMMAR PERMITTED {what}{also}. That "
                f"is not a schema being satisfied; it is the parser having "
                f"nowhere left to go, so anything after it is a fragment "
                f"rather than an answer. `validate_schema` exists to make this "
                f"unreachable, and reaching it means a construct got past the "
                f"pre-flight."
            )
        parts.append(
            f"The heaviest single step was {worst.step}, where "
            f"{worst.deleted_mass:.1%} was removed and "
            f"{worst.allowed_count} of {worst.vocab_size} tokens remained."
        )
        if self.truncated:
            parts.append(
                f"{self.truncated} further step(s) were NOT recorded: the "
                f"recorder holds {MAX_STEPS}."
            )
        parts.extend(tail)
        return " ".join(parts)

    def _median_deleted(self) -> float:
        import statistics

        return statistics.median([s.deleted_mass for s in self.steps])


def tokenizer_data(tokenizer):
    """The vocabulary, in the shape `TokenEnforcer` wants.

    Built once per tokenizer and reused: it walks the whole vocabulary, which
    is 50k+ `convert_ids_to_tokens` calls, and doing that per generation would
    dominate the cost of the thing being measured.
    """
    _, _, TokenEnforcerTokenizerData = _enforcer_module()

    eos = tokenizer.eos_token_id
    if eos is None:
        raise GrammarError(
            "this tokenizer states no end-of-sequence token, so the grammar has "
            "no way to know a value is finished."
        )

    vocab_size = len(tokenizer)
    special = set(getattr(tokenizer, "all_special_ids", ()) or ())

    # This mirrors `lmformatenforcer.integrations.transformers`, which cannot
    # be imported here (see the module docstring). Getting it wrong is not
    # subtle in its effects but IS subtle to spot:
    #
    # A first version used `convert_ids_to_tokens`, which on a byte-level BPE
    # vocabulary returns the ENCODED form — " Doe" comes back as "ĠDoe",
    # where Ġ stands in for the space. The grammar then saw Ġ as a
    # literal character inside a JSON string, its parser dead-ended, and the
    # allowed set collapsed to {EOS} mid-value. Measured: `{"name":"John` had
    # 50,174 tokens allowed and `{"name":"John Doe` had exactly 1.
    #
    # It survived an earlier check because that check used `{"name":"bob"}` —
    # no spaces inside any value, so no word-start token was ever exercised.
    token_zero = tokenizer.encode("0")[-1]
    regular = []
    for token_id in range(vocab_size):
        if token_id in special:
            continue
        # Prepend a known token and drop its one character, so a word-start
        # token yields the leading space it actually contributes.
        after_zero = tokenizer.decode([token_zero, token_id])[1:]
        plain = tokenizer.decode([token_id])
        regular.append((token_id, after_zero, len(after_zero) > len(plain)))

    def decode(ids) -> str:
        # A byte-BPE token can decode to a PARTIAL UTF-8 sequence, which comes
        # back as U+FFFD. Left in place it would be a character the grammar has
        # to account for and cannot.
        return tokenizer.decode(ids).rstrip("�")

    return TokenEnforcerTokenizerData(regular, decode, eos, False, vocab_size)


class MaskRecorder:
    """A `LogitsProcessor` that enforces a schema AND records what it cost.

    One object does both because they must see the same distribution: a
    recorder that ran beside the enforcer would read logits from a different
    step and report a mask that was never applied.
    """

    def __init__(self, tokenizer, schema: dict, data=None, temperature: float = 0.0):
        JsonSchemaParser, TokenEnforcer, _ = _enforcer_module()
        # The WHOLE tree, before anything is built. Constructing the parser
        # alone only compiles the root, and a nested construct it cannot build
        # is not discovered until the model has emitted the key it hangs off
        # -- by which time the only reports left are a collapsed mask behind a
        # 200 and an exception on a worker thread nobody is listening to.
        validate_schema(schema)
        try:
            # A FRESH parser, not the one `validate_schema` walked: that walk
            # compiles every sub-schema against its root and leaves parser
            # state behind it. The enforcer gets one nobody has touched.
            parser = JsonSchemaParser(schema)
        except Exception as err:
            # Unreachable via `validate_schema`, which just compiled this same
            # root. Kept so that if it ever becomes reachable it is a 400 in
            # this project's own words rather than a 500 in the library's --
            # `errors.py` is explicit that a caught exception's text is
            # machinery talking to itself and does not get published.
            raise GrammarError(_unenforceable("", schema, schema)) from err

        import json

        self.tokenizer = tokenizer
        self.data = data if data is not None else tokenizer_data(tokenizer)
        self.enforcer = TokenEnforcer(self.data, parser)
        written = json.dumps(schema, sort_keys=True)
        self.trace = Trace(
            schema=written[:MAX_SCHEMA_CHARS],
            schema_bytes=len(written),
            schema_truncated=len(written) > MAX_SCHEMA_CHARS,
            temperature=float(temperature),
            vocab_size=self.data.vocab_size
            if hasattr(self.data, "vocab_size")
            else len(tokenizer.get_vocab()),
        )

        # A LIMIT THE CALLER NEVER WROTE, said out loud.
        #
        # An array with no `maxItems` is not unbounded here: the compiler caps
        # it, and past the cap the mask simply stops permitting a comma, so
        # the array closes and the completion looks finished. Nothing in the
        # output distinguishes "the model was done" from "somebody else's
        # default ran out", which is exactly the silent truncation this module
        # exists to refuse -- and the schema the caller sent has been quietly
        # rewritten to produce it.
        unbounded = _unbounded_arrays(schema)
        if unbounded:
            cap = _array_cap()
            how_many = (
                f"caps an unbounded array at {cap} items"
                if cap is not None
                else "applies a cap of its own to an unbounded array, and this "
                "build could not read what that cap is"
            )
            self.trace.notes.append(
                f"THE SCHEMA DOES NOT BOUND {', '.join(unbounded)} WITH "
                f"'maxItems', and the grammar compiler {how_many}. Past that "
                f"the mask stops permitting another element and the array "
                f"closes -- a limit that came from the compiler, not from this "
                f"schema. Set 'maxItems' to say what you meant."
            )

        # A LIMIT THE CALLER DID WRITE, AND THE COMPILER THREW AWAY.
        #
        # The mirror of the note above and the more misleading of the two. That
        # one discloses something the compiler added; these disclose something
        # the caller asked for and did not get -- and the completion still
        # parses, still carries `output_parses_as_json: true`, and still comes
        # back beside a sentence about what the mask cost. Nothing else in the
        # response would say the bound was never compiled, and the caller has
        # every reason to believe it was.
        for pointer, names in _dropped_constraints(schema):
            self.trace.notes.append(
                f"THE SCHEMA CONSTRAINS {pointer} WITH "
                f"{', '.join(repr(n) for n in names)}, AND THE GRAMMAR "
                f"COMPILER READS NONE OF THOSE KEYWORDS: the mask permits "
                f"values at that position which this schema forbids, so a "
                f"completion that parses is not a completion that validates. "
                f"The rest of the schema is enforced as written."
            )
        for pointer in _free_dictionaries(schema):
            self.trace.notes.append(
                f"THE SCHEMA SETS 'additionalProperties' TO FALSE AT {pointer} "
                f"AND NAMES NO 'properties', AND THE GRAMMAR COMPILER READS "
                f"THAT AS 'any key, any value': it tests the keyword for truth, "
                f"so false falls through to its any-JSON default. The mask "
                f"permits arbitrary keys under a schema that asked for none. "
                f"Name the properties you want, or ask for "
                f'\'{{"type": "json_object"}}\' if any object will do.'
            )

        # The end-of-sequence id(s) the enforcer appends to every allowed set,
        # sorted the same way `allowed_ids` is so the two compare directly.
        # `TokenEnforcerTokenizerData` takes either one id or several.
        eos = getattr(self.data, "eos_token_id", None)
        self._eos_ids = sorted(
            {int(i) for i in (eos if isinstance(eos, (list, tuple, set)) else [eos])}
            if eos is not None
            else set()
        )
        self._n_prompt = None
        # Steps SEEN, which keeps counting after `steps` stops growing at
        # MAX_STEPS. `record_choice` needs to know the difference: past the
        # cap there is no row for the token being reported, and writing it to
        # the last one there is corrupts a row that already described a
        # different step.
        self._steps_seen = 0

    def __call__(self, input_ids, scores):
        import torch

        # The prompt length, fixed at the first call. Everything after it is
        # the generated prefix the enforcer needs -- and it needs EXACTLY the
        # generated part, because the grammar starts at the first new token.
        row = input_ids[0]
        if self._n_prompt is None:
            self._n_prompt = int(row.shape[0])
        generated = [int(t) for t in row[self._n_prompt :]]

        allowed = self.enforcer.get_allowed_tokens(generated).allowed_tokens
        # Clamped to the width of the logits row, not to the tokenizer's
        # length. The two differ -- a tokenizer with added tokens is longer
        # than the head that has to produce them -- and `keep[ids] = True`
        # with an id past the end raises IndexError inside `generate`'s worker
        # thread, where nothing catches it and the consumer waits out the
        # streamer's timeout. Ids with no logit are tokens this model cannot
        # emit at all, so dropping them removes nothing that was reachable.
        width = int(scores.shape[-1])
        allowed_ids = sorted({int(i) for i in allowed if 0 <= int(i) < width})

        # TEMPERATURE IS APPLIED HERE, not by transformers.
        #
        # `generate` appends its sampling warpers AFTER every custom processor
        # and offers no way to ask for another position, so a recorder that
        # left temperature to HF would be reporting the mass removed from a
        # temperature-1 distribution the model never had -- while the module
        # docstring claimed the opposite. One division, the same one
        # `TemperatureLogitsWarper` does, and the claim becomes true.
        # `runtime.generate_stream` neutralises HF's warpers to match; a
        # caller that does not would apply temperature twice.
        temperature = float(self.trace.temperature)
        scaled = scores if temperature <= 0 else scores / temperature

        probs = torch.softmax(scaled[0].float(), dim=-1)
        top = torch.topk(probs, min(TOP_K, probs.shape[-1]))
        wanted_id = int(top.indices[0])
        allowed_set = set(allowed_ids)

        keep = torch.zeros_like(scaled[0], dtype=torch.bool)
        if allowed_ids:
            keep[torch.tensor(allowed_ids, device=scores.device)] = True
        # The mass the mask removes, from the distribution the model actually
        # had at this step.
        #
        # SUMMED IN FLOAT64, then clamped. Adding ~50,000 float32 terms
        # accumulates enough error to exceed 1.0 -- measured, a step where the
        # mask removed essentially everything reported 1.000023, and a
        # probability above 1 is not a thing to print at a reader. float64
        # removes the accumulation; the clamp is the belt for the remaining
        # ulp, not a substitute for it.
        deleted = float(probs.double()[~keep].sum())
        deleted = min(1.0, max(0.0, deleted))

        self._steps_seen += 1
        if len(self.trace.steps) < MAX_STEPS:
            self.trace.steps.append(
                Step(
                    step=len(self.trace.steps),
                    wanted=self.tokenizer.decode([wanted_id]),
                    wanted_id=wanted_id,
                    wanted_p=round(float(top.values[0]), 6),
                    wanted_was_allowed=wanted_id in allowed_set,
                    allowed_count=len(allowed_ids),
                    vocab_size=width,
                    deleted_mass=round(deleted, 6),
                    eos_only=allowed_ids == self._eos_ids,
                    top=[
                        {
                            "token": self.tokenizer.decode([int(i)]),
                            "p": round(float(p), 6),
                            "allowed": int(i) in allowed_set,
                        }
                        for p, i in zip(top.values, top.indices, strict=True)
                    ],
                )
            )
        else:
            self.trace.truncated += 1

        # -inf, not a large negative: a finite penalty leaves a forbidden token
        # reachable under sampling, and the whole promise of constrained
        # decoding is that it is not.
        masked = scaled.clone()
        masked[0][~keep] = float("-inf")
        return masked

    def record_choice(self, token_id: int) -> None:
        """What was actually emitted at the last recorded step.

        Separate from `__call__` because the processor runs BEFORE the token is
        chosen; the choice is only knowable afterwards. A step whose `chosen`
        is never set keeps `chosen_id = -1`, which reads as "not recorded"
        rather than as token 0.
        """
        # Only while steps are still being RECORDED. `__call__` stops
        # appending at MAX_STEPS and this went on writing to `steps[-1]`, so
        # every token past the cap overwrote the last recorded step's choice
        # -- step 1,999 ended up showing the token emitted at step 5,000,
        # beside its own `wanted` and its own `top`. A row describing two
        # different moments is worse than a missing row.
        if not self.trace.steps or self._steps_seen > len(self.trace.steps):
            return
        step = self.trace.steps[-1]
        step.chosen_id = int(token_id)
        step.chosen = self.tokenizer.decode([int(token_id)])
        for entry in step.top:
            if entry["token"] == step.chosen:
                step.chosen_p = entry["p"]
                break


def plan(schema: dict, tokenizer, data=None) -> dict:
    """What the grammar permits at step 0, before anything is generated.

    It answers the question people actually have first: how much of the
    vocabulary does this schema rule out before the model says anything?

    `data` is not optional in the way it looks. Without it this builds a whole
    `tokenizer_data` -- 50k decodes on gpt2 -- for one lookup, which is the
    exact cost `tokenizer_data`'s own docstring says must not be paid per
    call. Anything on a request path passes the cached one.
    """
    recorder = MaskRecorder(tokenizer, schema, data=data)
    allowed = recorder.enforcer.get_allowed_tokens([]).allowed_tokens
    ids = list(allowed)
    total = len(tokenizer.get_vocab())
    return {
        "allowed_at_start": len(ids),
        "vocab_size": total,
        "masked_fraction": round(1 - len(ids) / total, 6) if total else 0.0,
        "examples": [tokenizer.decode([i]) for i in ids[:8]],
        "note": _STEP_ORDER_NOTE,
    }
