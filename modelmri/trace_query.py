"""Parse a search box into a query, without ever building SQL from user text.

The search box accepts free text plus structured filters — `kind:tool_call`,
`error:true`, `duration>2000`, `name:pytest` — which is the shape people expect
from every issue tracker they have used. It is also the shape that invites
string interpolation into SQL, so nothing here returns SQL. It returns a parsed
object of allow-listed fields, and the caller builds a parameterised statement
from it.

**Allow-list, not escaping.** A field name that is not in `FIELDS` is not
escaped and passed through, it is refused. Escaping is a bet that the escaper
is correct for every input; an allow-list is a fact about which strings can
ever reach the statement, and there are five of them.

**Unparseable filters are refused, not dropped.** `kind:tolcall` silently
matching nothing looks exactly like "there are no tool calls", and a person
searching a trace for the step that failed is the last person who should be
handed a confident empty list. The typo is named and the valid values listed.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field

from .errors import BadRequest
from .traces import VALID_KINDS

# The only column names that can ever reach a statement built from this.
FIELDS = ("kind", "name", "error", "duration")

# `field:value`, `field>value`, `field<value`. Values may be quoted to carry
# spaces. Anything not matching this is free text.
#
# **No whitespace around the operator**, which is the convention every issue
# tracker uses and is load-bearing here rather than cosmetic. With `\s*` around
# the colon, pasting an ordinary log line — "error: connection refused" — parsed
# as the filter `error:connection` and was refused, so the single most likely
# thing anybody pastes into a search box could not be searched for. Tight
# binding makes `error:true` a filter and `error: connection refused` prose.
_FILTER = re.compile(
    r"""(?<![^\s(])(?P<field>[a-z_]+)(?P<op>[:><])(?P<value>"[^"]*"|'[^']*'|\S+)""",
    re.IGNORECASE,
)

_TRUTHY = {"true", "yes", "1", "on"}
_FALSY = {"false", "no", "0", "off"}


@dataclass
class Query:
    """A parsed search: free text plus zero or more allow-listed filters."""

    text: str = ""
    kind: str | None = None
    name: str | None = None
    error: bool | None = None
    duration_gt: int | None = None
    duration_lt: int | None = None
    filters_used: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)

    @property
    def is_empty(self) -> bool:
        return not self.text and not self.filters_used


def parse(raw: str) -> Query:
    """Split a search string into free text and structured filters.

    Free text is whatever is left after the filters are removed, so
    `pytest kind:tool_call` searches for "pytest" among tool calls rather than
    searching for the literal string "kind:tool_call".
    """
    query = Query()
    leftovers: list[str] = []
    cursor = 0
    raw = raw or ""

    for match in _FILTER.finditer(raw):
        field_name = match.group("field").lower()
        if field_name not in FIELDS:
            # Not a filter we know — treat it as free text rather than
            # refusing, because a colon in ordinary prose ("error: timeout"
            # pasted from a log) is far more common than a typo'd field, and
            # refusing that would make the box hostile to the obvious use.
            continue
        leftovers.append(raw[cursor : match.start()])
        cursor = match.end()

        op = match.group("op")
        value = match.group("value").strip("\"'")
        _apply(query, field_name, op, value)
        query.filters_used.append(f"{field_name}{op}{value}")

    leftovers.append(raw[cursor:])
    query.text = " ".join(" ".join(leftovers).split())
    return query


def _apply(query: Query, name: str, op: str, value: str) -> None:
    if name == "kind":
        if value not in VALID_KINDS:
            raise BadRequest(
                f"unknown step kind {value!r} — use one of "
                f"{', '.join(sorted(VALID_KINDS))}"
            )
        query.kind = value
        return

    if name == "name":
        query.name = value
        return

    if name == "error":
        lowered = value.lower()
        if lowered in _TRUTHY:
            query.error = True
        elif lowered in _FALSY:
            query.error = False
        else:
            raise BadRequest(
                f"error: takes true or false, not {value!r}. A filter that "
                "quietly matched nothing would look exactly like a trace with "
                "no failures."
            )
        return

    if name == "duration":
        if op == ":":
            raise BadRequest(
                "duration needs a comparison — write duration>2000 or "
                "duration<50 (milliseconds). `duration:2000` would match only "
                "steps that took exactly that long, which is almost never "
                "what anybody means."
            )
        try:
            millis = int(value)
        except ValueError:
            raise BadRequest(
                f"duration takes a whole number of milliseconds, not {value!r}"
            ) from None
        if op == ">":
            query.duration_gt = millis
        else:
            query.duration_lt = millis


def where(query: Query) -> tuple[str, list]:
    """(SQL fragment, parameters). Every value is a placeholder, never inlined.

    The fragment names only columns from `FIELDS`, and those names are literals
    in this function's own source rather than anything derived from input.
    """
    clauses: list[str] = []
    params: list = []

    if query.kind:
        clauses.append("s.kind = ?")
        params.append(query.kind)
    if query.name:
        clauses.append("s.name LIKE ?")
        params.append(f"%{query.name}%")
    if query.error is not None:
        clauses.append("s.error = ?")
        params.append(1 if query.error else 0)
    if query.duration_gt is not None:
        # `IS NOT NULL` explicitly: duration is nullable now, and in SQL a
        # comparison against NULL is NULL rather than false. Without this a
        # step whose duration was never recorded would be excluded by
        # `duration<50` AND by `duration>50`, which is not a filter, it is a
        # disappearance.
        clauses.append("s.duration_ms IS NOT NULL AND s.duration_ms > ?")
        params.append(query.duration_gt)
    if query.duration_lt is not None:
        clauses.append("s.duration_ms IS NOT NULL AND s.duration_ms < ?")
        params.append(query.duration_lt)

    return (" AND ".join(clauses), params)
