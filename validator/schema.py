"""The CoefficientSet schema and its strict validation rules.

A ``CoefficientSet`` is a named, immutable set of coefficients that feeds ONE backend
and may ``extend`` another set. Validation is strict: an unknown key is an error, never
a silent default, and every rejection names the offending entry or key so a committer
can find it without guessing.

The rules encoded here are exactly the acceptance criteria of registry issue #1:

  * an entry missing ``method``/``units``/``scope`` is rejected, naming the entry;
  * a method lacking its companion field (``rationale``/``sources``/``copied_from``)
    is rejected;
  * a zero ``value`` is rejected unless ``method: not_charged``;
  * an unknown top-level key, entry field, or ``scope`` key is rejected;
  * a set that omits a coefficient its backend consumes is refused, naming it;
  * ``ci95`` is accepted when present and never required.

``check_set`` returns a sorted list of human-readable error strings — empty means the
set is valid. It never raises on bad data; malformed input becomes an error string so a
CI run reports the problem instead of crashing.
"""

from __future__ import annotations

import math
from typing import Any

# ---------------------------------------------------------------------------
# Enumerations. Adding a member is a deliberate schema addition, not a redesign.
# ---------------------------------------------------------------------------

UNITS = frozenset(
    {
        "dimensionless",
        "us_per_layer",
        "us_per_request",
        "us_per_step",
        "us_per_hop",
    }
)

METHODS = frozenset(
    {
        "measured",
        "literature",
        "vendor_spec",
        "copied",
        "assumed",
        "not_charged",
    }
)

# Scope keys are OPEN by design: a key the resolver does not know is an error (strict),
# but adding one here is a schema addition, not a redesign.
SCOPE_KEYS = frozenset({"hardware", "tp", "ep", "nodes_spanned", "model_class"})

# Top-level keys of a CoefficientSet document.
TOP_LEVEL_REQUIRED = frozenset({"name", "backend", "coefficients"})
TOP_LEVEL_OPTIONAL = frozenset({"extends"})
TOP_LEVEL_KEYS = TOP_LEVEL_REQUIRED | TOP_LEVEL_OPTIONAL

# Entry fields.
ENTRY_REQUIRED = frozenset({"value", "units", "method", "fitted", "scope"})
ENTRY_OPTIONAL = frozenset(
    {"ci95", "sources", "rationale", "copied_from", "supersedes"}
)
ENTRY_KEYS = ENTRY_REQUIRED | ENTRY_OPTIONAL


def _is_finite_number(v: Any) -> bool:
    # bool is a subclass of int; a bare True/False is not a coefficient value.
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        return False
    return math.isfinite(v)


def _check_entry(name: str, entry: Any) -> list[str]:
    """Validate a single coefficient entry. Returns a list of error strings."""
    errors: list[str] = []
    if not isinstance(entry, dict):
        return [f"coefficient {name!r}: entry must be a mapping, got {type(entry).__name__}"]

    # Strict parse: unknown entry fields are errors, never ignored. Sort by str so a
    # mixed-type key set (e.g. an unquoted numeric field name that YAML parsed to int)
    # never crashes the comparison.
    for key in sorted(entry, key=str):
        if key not in ENTRY_KEYS:
            errors.append(f"coefficient {name!r}: unknown field {key!r}")

    # Required fields present.
    for field in sorted(ENTRY_REQUIRED):
        if field not in entry:
            errors.append(f"coefficient {name!r}: missing required field {field!r}")

    units = entry.get("units")
    if "units" in entry and units not in UNITS:
        errors.append(
            f"coefficient {name!r}: units {units!r} not one of {sorted(UNITS)}"
        )

    method = entry.get("method")
    if "method" in entry and method not in METHODS:
        errors.append(
            f"coefficient {name!r}: method {method!r} not one of {sorted(METHODS)}"
        )

    # value: finite number; zero only when the backend explicitly does not charge it.
    if "value" in entry:
        value = entry["value"]
        if not _is_finite_number(value):
            errors.append(
                f"coefficient {name!r}: value must be a finite number, got {value!r}"
            )
        elif value == 0 and method != "not_charged":
            errors.append(
                f"coefficient {name!r}: value 0 requires method 'not_charged' "
                f"(a priced-at-zero term), got method {method!r}"
            )

    # fitted: bool, and true only for measured (a value fitted against its own term).
    if "fitted" in entry:
        fitted = entry["fitted"]
        if not isinstance(fitted, bool):
            errors.append(
                f"coefficient {name!r}: fitted must be a boolean, got {fitted!r}"
            )
        elif fitted and method != "measured":
            errors.append(
                f"coefficient {name!r}: fitted: true requires method 'measured', "
                f"got method {method!r}"
            )

    # scope: an open-keyed, non-empty mapping; unknown keys are strict errors.
    if "scope" in entry:
        scope = entry["scope"]
        if not isinstance(scope, dict):
            errors.append(
                f"coefficient {name!r}: scope must be a mapping, got {type(scope).__name__}"
            )
        elif not scope:
            errors.append(f"coefficient {name!r}: scope must not be empty")
        else:
            for key in sorted(scope, key=str):
                if key not in SCOPE_KEYS:
                    errors.append(
                        f"coefficient {name!r}: unknown scope key {key!r} "
                        f"(known: {sorted(SCOPE_KEYS)})"
                    )
                    continue
                # A known scope key must carry an actual range/value. A null or empty
                # value conveys no scoping yet would satisfy a mere presence check —
                # the "truthy but empty" hole the provenance fields already guard against.
                val = scope[key]
                if val is None or (isinstance(val, (list, dict, str)) and not val):
                    errors.append(
                        f"coefficient {name!r}: scope key {key!r} must have a non-empty value"
                    )

    # ci95: optional, but when present it must be a 2-element [lower, upper] interval of
    # finite numbers with lower <= upper. "Accepted when present" must not mean "ignored":
    # a NaN/Inf or non-numeric confidence interval is bad provenance, checked with the same
    # rigor as `value`.
    if "ci95" in entry:
        ci95 = entry["ci95"]
        if (
            not isinstance(ci95, list)
            or len(ci95) != 2
            or not all(_is_finite_number(x) for x in ci95)
        ):
            errors.append(
                f"coefficient {name!r}: ci95 must be a 2-element list of finite numbers "
                f"[lower, upper], got {ci95!r}"
            )
        elif ci95[0] > ci95[1]:
            errors.append(
                f"coefficient {name!r}: ci95 lower bound {ci95[0]!r} exceeds upper "
                f"bound {ci95[1]!r}"
            )

    # supersedes: optional; when present, names a prior coefficient it replaces, so it
    # must be a non-empty string (mirroring copied_from). An arbitrary map/list is not a
    # coefficient name.
    if "supersedes" in entry:
        supersedes = entry["supersedes"]
        if not isinstance(supersedes, str) or not supersedes.strip():
            errors.append(
                f"coefficient {name!r}: supersedes must be a non-empty string "
                f"(the coefficient name it replaces)"
            )

    # sources: well-formed whenever PRESENT (a list of non-empty strings), independent of
    # whether the method REQUIRES it below. A committer who typed sources under any method
    # meant to cite something; junk there should not pass silently.
    if "sources" in entry:
        sources = entry["sources"]
        if (
            not isinstance(sources, list)
            or not sources
            or not all(isinstance(s, str) and s.strip() for s in sources)
        ):
            errors.append(
                f"coefficient {name!r}: sources must be a non-empty list of strings"
            )

    # Required-by-method companion fields. Provenance must be human-readable text, so
    # these are checked as non-empty STRINGS (or a list of strings for sources) — a bare
    # number or list would satisfy a truthiness test while carrying no provenance.
    if method in METHODS:
        if method != "measured":
            rationale = entry.get("rationale")
            if not isinstance(rationale, str) or not rationale.strip():
                errors.append(
                    f"coefficient {name!r}: method {method!r} requires a non-empty "
                    f"string 'rationale'"
                )
        if method in ("literature", "vendor_spec"):
            # The "present ⇒ well-formed" check above already validates shape; here we only
            # additionally REQUIRE its presence for these two methods.
            sources = entry.get("sources")
            if not isinstance(sources, list) or not sources:
                errors.append(
                    f"coefficient {name!r}: method {method!r} requires a non-empty "
                    f"'sources' list of strings"
                )
        if method == "copied":
            copied_from = entry.get("copied_from")
            if not isinstance(copied_from, str) or not copied_from.strip():
                errors.append(
                    f"coefficient {name!r}: method 'copied' requires a non-empty "
                    f"string 'copied_from'"
                )

    return errors


def check_set(data: Any, consumed: frozenset[str] | None, inherited: frozenset[str]) -> list[str]:
    """Validate a parsed CoefficientSet.

    ``consumed`` is the set of coefficient names the declared backend consumes (from its
    manifest), or ``None`` when the backend is unknown — in which case the omitted-by-name
    check is skipped but a backend error is reported. ``inherited`` is the set of
    coefficient names available from the resolved ``extends`` chain.

    Returns a sorted list of error strings; empty means valid.
    """
    errors: list[str] = []
    if not isinstance(data, dict):
        return [f"top level must be a mapping, got {type(data).__name__}"]

    # Strict parse: unknown top-level keys are errors. Sort by str so mixed-type keys
    # never crash the comparison (a non-string key is itself an "unknown key").
    for key in sorted(data, key=str):
        if key not in TOP_LEVEL_KEYS:
            errors.append(f"unknown top-level key {key!r}")
    for key in sorted(TOP_LEVEL_REQUIRED):
        if key not in data:
            errors.append(f"missing required top-level key {key!r}")

    # name must be a non-empty string. A non-string name (e.g. an unquoted number) not
    # only carries no identity, it is silently dropped from the CLI's set index — which
    # would bypass the duplicate-name guard and make the set invisible as an `extends`
    # target. Rejecting it here closes that silent hole at the schema level.
    if "name" in data:
        set_name = data["name"]
        if not isinstance(set_name, str) or not set_name.strip():
            errors.append(f"'name' must be a non-empty string, got {set_name!r}")

    coefficients = data.get("coefficients")
    own_names: frozenset[str] = frozenset()
    if "coefficients" in data:
        if not isinstance(coefficients, dict):
            errors.append(
                f"'coefficients' must be a mapping, got {type(coefficients).__name__}"
            )
        elif not coefficients:
            errors.append("'coefficients' must not be empty")
        else:
            own_names = frozenset(coefficients)
            # Sort by str so a mixed-type key set never crashes the comparison.
            for name in sorted(coefficients, key=str):
                # Coefficient names are matched against a backend's consumed-names list,
                # which is strings; a non-string name (e.g. an unquoted numeric key YAML
                # parsed to int) can never match and is a strict error.
                if not isinstance(name, str):
                    errors.append(
                        f"coefficient name {name!r} must be a string, got {type(name).__name__}"
                    )
                errors.extend(_check_entry(str(name), coefficients[name]))

    # Backend consumed-names check: a coefficient the backend consumes but the set
    # (after resolving extends) does not provide is refused by name. A coefficient
    # present but NOT consumed is allowed — the ABSENT case and the shared-base "drop".
    if consumed is not None:
        available = own_names | inherited
        for wanted in sorted(consumed):
            if wanted not in available:
                errors.append(
                    f"backend consumes {wanted!r} but the set does not provide it "
                    f"(add the entry or extend a set that defines it)"
                )

    return sorted(errors)
