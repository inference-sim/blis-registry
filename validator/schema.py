"""The CoefficientSet schema and its strict validation rules.

A ``CoefficientSet`` is a standalone, immutable set of coefficients. It is a
self-identifying document: its identity is its top-level ``name``, not its filename and
not an external manifest. There is no inheritance between sets — each set is complete on
its own. Validation is strict: an unknown key is an error, never a silent default, and
every rejection names the offending entry or key so a committer can find it without
guessing.

The shape:

  * a document declares ``kind: CoefficientSet``, a unique ``name``, and its
    ``coefficients`` — a LIST of single-key maps, each keyed by the coefficient name;
  * every entry carries ``value``/``units``/``method``/``fitted``/``scope``; a missing
    one is rejected, naming the entry;
  * a method lacking its companion (``rationale`` for all but ``measured``; ``sources``
    for ``literature``/``vendor_spec``; ``copied_from`` for ``copied``) is rejected;
  * a zero ``value`` is rejected unless ``method: not_charged``;
  * ``sources`` is a list of ``{kind, cite, role}`` provenance objects;
  * ``ci95`` and ``supersedes`` are optional and may be an explicit ``null``.

The registry validator checks SHAPE ONLY. It no longer verifies that a set provides every
coefficient some backend consumes: with the per-backend manifest (``backends/``) gone,
per-backend *completeness* ("every coefficient a backend needs is present") moves to the
simulator-side loader, which is the path that actually knows what each backend consumes.

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
# but adding one here is a schema addition, not a redesign. `model` scopes an entry to a
# model or model class (design: `scope: {model: [...]}`).
SCOPE_KEYS = frozenset({"hardware", "tp", "ep", "nodes_spanned", "model"})

# Provenance source-object fields and their vocabularies (design: each source is a
# {kind, cite, role} object; the value is triangulated across a list of them).
SOURCE_KINDS = frozenset(
    {"discussion", "publication", "datasheet", "model", "vendor_doc"}
)
SOURCE_ROLES = frozenset({"primary", "supporting", "upper_bound"})
SOURCE_REQUIRED = frozenset({"kind", "cite", "role"})

# Top-level keys of a CoefficientSet document. The set's identity is its `name`; `kind`
# is the type discriminator. There is no `backend` (no per-backend manifest anymore) and
# no `extends` (sets are standalone) — either one is now an unknown key, hence rejected.
TOP_LEVEL_REQUIRED = frozenset({"kind", "name", "coefficients"})
TOP_LEVEL_OPTIONAL = frozenset()
TOP_LEVEL_KEYS = TOP_LEVEL_REQUIRED | TOP_LEVEL_OPTIONAL

# Entry fields. `validated`/`unsupported` carry the honest-asymmetry scoping the design
# introduces for transfer/LoRA entries (a term may be validated for one metric and
# explicitly unsupported for another).
ENTRY_REQUIRED = frozenset({"value", "units", "method", "fitted", "scope"})
ENTRY_OPTIONAL = frozenset(
    {"ci95", "sources", "rationale", "copied_from", "supersedes",
     "validated", "unsupported"}
)
ENTRY_KEYS = ENTRY_REQUIRED | ENTRY_OPTIONAL


def _is_finite_number(v: Any) -> bool:
    # bool is a subclass of int; a bare True/False is not a coefficient value.
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        return False
    return math.isfinite(v)


def _in_enum(value: Any, allowed: frozenset[str]) -> bool:
    """True iff ``value`` is a member of ``allowed``.

    A non-string (or unhashable, e.g. a list/dict from a malformed YAML value) can never
    be a valid enum member, so it is reported as "not in the enum" rather than being fed
    to ``value in allowed`` — where an unhashable value would raise ``TypeError`` and
    escape ``check_set``'s no-raise contract as a traceback.
    """
    return isinstance(value, str) and value in allowed


def _check_sources(name: str, sources: Any) -> list[str]:
    """Validate a `sources` list of {kind, cite, role} provenance objects.

    The design models a value as triangulated across a LIST of sources, each a structured
    object rather than a bare string, so a reviewer can see what kind of evidence it is
    (a discussion, a publication, a datasheet …) and what role it plays (the primary
    basis, a supporting agreement, or an upper bound). Returns error strings.
    """
    errors: list[str] = []
    if not isinstance(sources, list) or not sources:
        return [f"coefficient {name!r}: 'sources' must be a non-empty list"]
    for i, src in enumerate(sources):
        if not isinstance(src, dict):
            errors.append(
                f"coefficient {name!r}: sources[{i}] must be a {{kind, cite, role}} "
                f"mapping, got {type(src).__name__}"
            )
            continue
        for key in sorted(src, key=str):
            if key not in SOURCE_REQUIRED:
                errors.append(f"coefficient {name!r}: sources[{i}] unknown field {key!r}")
        kind = src.get("kind")
        if not _in_enum(kind, SOURCE_KINDS):
            errors.append(
                f"coefficient {name!r}: sources[{i}] kind {kind!r} not one of "
                f"{sorted(SOURCE_KINDS)}"
            )
        cite = src.get("cite")
        if not isinstance(cite, str) or not cite.strip():
            errors.append(
                f"coefficient {name!r}: sources[{i}] requires a non-empty string 'cite'"
            )
        role = src.get("role")
        if not _in_enum(role, SOURCE_ROLES):
            errors.append(
                f"coefficient {name!r}: sources[{i}] role {role!r} not one of "
                f"{sorted(SOURCE_ROLES)}"
            )
    return errors


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
    if "units" in entry and not _in_enum(units, UNITS):
        errors.append(
            f"coefficient {name!r}: units {units!r} not one of {sorted(UNITS)}"
        )

    method = entry.get("method")
    if "method" in entry and not _in_enum(method, METHODS):
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
                elif isinstance(val, list):
                    # Each element of a scope range must be a non-empty scalar: a null, a
                    # nested container, or a blank/whitespace-only string conveys no range
                    # and must not slip through inside an otherwise non-empty list.
                    for elem in val:
                        if (
                            elem is None
                            or isinstance(elem, (list, dict))
                            or (isinstance(elem, str) and not elem.strip())
                        ):
                            errors.append(
                                f"coefficient {name!r}: scope key {key!r} has an empty or "
                                f"non-scalar element {elem!r}"
                            )

    # ci95: optional; an explicit `null` is the canonical "no interval claimed" (design).
    # When it carries a value it must be a 2-element [lower, upper] of finite numbers with
    # lower <= upper — checked with the same rigor as `value`, not ignored.
    if entry.get("ci95") is not None:
        ci95 = entry["ci95"]
        if (
            not isinstance(ci95, list)
            or len(ci95) != 2
            or not all(_is_finite_number(x) for x in ci95)
        ):
            errors.append(
                f"coefficient {name!r}: ci95 must be null or a 2-element list of finite "
                f"numbers [lower, upper], got {ci95!r}"
            )
        elif ci95[0] > ci95[1]:
            errors.append(
                f"coefficient {name!r}: ci95 lower bound {ci95[0]!r} exceeds upper "
                f"bound {ci95[1]!r}"
            )

    # supersedes: optional; an explicit `null` is canonical (design). When set, it names a
    # prior coefficient it replaces, so it must be a non-empty string.
    if entry.get("supersedes") is not None:
        supersedes = entry["supersedes"]
        if not isinstance(supersedes, str) or not supersedes.strip():
            errors.append(
                f"coefficient {name!r}: supersedes must be null or a non-empty string "
                f"(the coefficient name it replaces)"
            )

    # sources: well-formed whenever PRESENT (a list of {kind, cite, role} objects),
    # independent of whether the method REQUIRES it below.
    if "sources" in entry:
        errors.extend(_check_sources(name, entry["sources"]))

    # String-valued optional fields: well-formed (a non-empty string) WHENEVER PRESENT,
    # independent of whether any method requires them. A `measured` entry carrying
    # `rationale: 123` or `copied_from: []` is malformed even though neither is required
    # there — the same "present ⇒ well-formed" rule sources/ci95/supersedes already follow.
    # `validated`/`unsupported` are the honest-asymmetry metric names (design).
    for field in ("rationale", "copied_from", "validated", "unsupported"):
        if field in entry:
            v = entry[field]
            if not isinstance(v, str) or not v.strip():
                errors.append(
                    f"coefficient {name!r}: {field!r} must be a non-empty string, "
                    f"got {v!r}"
                )

    # Required-by-method: which companion fields must be PRESENT. Shape is enforced above
    # (or in _check_sources); here we only require presence for the relevant methods.
    # _in_enum guards against a non-string/unhashable method (already flagged above) that
    # would otherwise raise on the set-membership test.
    if _in_enum(method, METHODS):
        if method != "measured" and "rationale" not in entry:
            errors.append(
                f"coefficient {name!r}: method {method!r} requires a 'rationale'"
            )
        if method in ("literature", "vendor_spec") and "sources" not in entry:
            errors.append(
                f"coefficient {name!r}: method {method!r} requires 'sources'"
            )
        if method == "copied" and "copied_from" not in entry:
            errors.append(
                f"coefficient {name!r}: method 'copied' requires 'copied_from'"
            )

    return errors


def _check_coefficients(coefficients: Any) -> list[str]:
    """Validate the ``coefficients`` list of single-key maps. Returns error strings.

    Each element is a mapping with EXACTLY ONE key — the coefficient name — whose value is
    the entry. The list shape (rather than a map) means the strict LOADER can no longer
    catch a repeated coefficient name — two ``- mfu_prefill:`` items are two distinct
    dicts, not one duplicated mapping key — so duplicate-name detection lives HERE instead,
    preserving the "a second definition never silently shadows the first" guarantee.
    """
    errors: list[str] = []
    if not isinstance(coefficients, list):
        return [
            f"'coefficients' must be a list, got {type(coefficients).__name__}"
        ]
    if not coefficients:
        return ["'coefficients' must not be empty"]

    seen: set[str] = set()
    for i, item in enumerate(coefficients):
        if not isinstance(item, dict):
            errors.append(
                f"coefficients[{i}] must be a single-key {{name: entry}} mapping, "
                f"got {type(item).__name__}"
            )
            continue
        if len(item) != 1:
            errors.append(
                f"coefficients[{i}] must have exactly one key (the coefficient name), "
                f"got {len(item)}"
            )
            continue
        # The sole key is the coefficient name. A non-string name (e.g. an unquoted
        # numeric key YAML parsed to int) is a strict error; stringify for the entry
        # checks so a name typo never crashes downstream formatting.
        (name, entry), = item.items()
        if not isinstance(name, str):
            errors.append(
                f"coefficients[{i}]: coefficient name {name!r} must be a string, "
                f"got {type(name).__name__}"
            )
        name_str = str(name)
        if name_str in seen:
            errors.append(f"duplicate coefficient name {name_str!r}")
        seen.add(name_str)
        errors.extend(_check_entry(name_str, entry))
    return errors


def check_set(data: Any) -> list[str]:
    """Validate a parsed CoefficientSet. Returns a sorted list of error strings.

    Empty means valid. The check is SHAPE ONLY — there is no per-backend completeness
    check anymore (that moved to the simulator-side loader with ``backends/`` removed).
    """
    errors: list[str] = []
    if not isinstance(data, dict):
        return [f"top level must be a mapping, got {type(data).__name__}"]

    # Strict parse: unknown top-level keys are errors. `backend` and `extends`, accepted by
    # the old format, are now unknown keys and rejected here. Sort by str so mixed-type
    # keys never crash the comparison (a non-string key is itself an "unknown key").
    for key in sorted(data, key=str):
        if key not in TOP_LEVEL_KEYS:
            errors.append(f"unknown top-level key {key!r}")
    for key in sorted(TOP_LEVEL_REQUIRED):
        if key not in data:
            errors.append(f"missing required top-level key {key!r}")

    # kind is the type discriminator; it must be exactly 'CoefficientSet' so a file that
    # is actually some other object (or a typo) is refused rather than half-validated.
    if "kind" in data and data["kind"] != "CoefficientSet":
        errors.append(f"'kind' must be 'CoefficientSet', got {data['kind']!r}")

    # name is the set's identity; it must be a non-empty string.
    if "name" in data and (
        not isinstance(data["name"], str) or not data["name"].strip()
    ):
        errors.append(f"'name' must be a non-empty string, got {data['name']!r}")

    if "coefficients" in data:
        errors.extend(_check_coefficients(data["coefficients"]))

    return sorted(errors)
