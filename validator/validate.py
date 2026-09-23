#!/usr/bin/env python3
"""Validate coefficient sets against the CoefficientSet schema.

Usage:
    validate.py [PATH ...]

Each PATH is a coefficient-set YAML file or a directory scanned recursively for YAML
files (``*.yaml``/``*.yml``, case-insensitive). With no
PATH, the real sets in ``operators/`` and the synthetic schema fixtures in ``fixtures/``
are validated (``operators/`` may be empty until real sets are transcribed). The tool
loads each set, resolves its ``extends`` chain and its backend manifest, runs the strict
schema checks, and prints one line per problem. Exit code is 0 iff every set is valid —
this is the gate CI runs on every committed set.

A coefficient set is identified by its **filename stem** (``operators/roofline.yaml`` is
the set ``roofline``), not by a field in the document. ``extends:`` names the stem of a
base set to inherit from.

Backend manifests live in ``backends/<name>.yaml`` and declare the coefficient names a
backend consumes:

    consumes: [mfu_prefill, mfu_decode]

A set names its backend with ``backend:`` and may inherit entries with ``extends:``. The
manifest is what lets the validator refuse an omitted coefficient BY NAME rather than let
it default silently.
"""

from __future__ import annotations

import sys
from pathlib import Path

import yaml

# Support running both as a module (``python -m validator.validate``) and as a script
# (``python validator/validate.py``), where the package is not on sys.path.
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from validator.loader import DuplicateKeyError, load_strict
    from validator.schema import check_set
else:
    from .loader import DuplicateKeyError, load_strict
    from .schema import check_set

REPO_ROOT = Path(__file__).resolve().parent.parent
BACKENDS_DIR = REPO_ROOT / "backends"
OPERATORS_DIR = REPO_ROOT / "operators"   # the real coefficient sets
FIXTURES_DIR = REPO_ROOT / "fixtures"     # synthetic schema fixtures (not registry data)

# With no explicit path, validate both the real sets and the fixtures. `operators/` may be
# empty (no real sets shipped yet); the fixtures keep the CI gate non-vacuous meanwhile.
DEFAULT_TARGETS = [OPERATORS_DIR, FIXTURES_DIR]


# Every way a YAML file can fail to load into usable data. UnicodeDecodeError (a
# ValueError subclass, NOT an OSError) covers a non-UTF-8 file. Complex/unhashable keys
# are named by the loader as a DuplicateKeyError, so TypeError is retained only as a
# defensive catch-all for any unhashable value that might still surface. Catching these
# keeps a malformed committed file a NAMED error instead of a stack trace escaping to CI.
LOAD_ERRORS = (yaml.YAMLError, DuplicateKeyError, OSError, ValueError, TypeError)


def _load_file(path: Path):
    """Load and strictly parse one YAML file. Raises a LOAD_ERRORS member on bad input."""
    return load_strict(path.read_text(encoding="utf-8"))


def _backend_consumes(backend: str) -> tuple[frozenset[str] | None, str | None]:
    """Resolve a backend's consumed-names list.

    Returns ``(names, None)`` on success, or ``(None, reason)`` where ``reason``
    distinguishes a *missing* manifest from a *malformed* one — so the caller can refuse
    a set naming the real problem instead of misreporting a present-but-broken manifest
    as absent (which would also silently skip the omitted-by-name check).
    """
    manifest = BACKENDS_DIR / f"{backend}.yaml"
    if not manifest.is_file():
        return None, f"backend {backend!r} has no manifest (add backends/{backend}.yaml)"
    try:
        data = _load_file(manifest)
    except LOAD_ERRORS as exc:
        return None, f"backend manifest backends/{backend}.yaml could not be parsed: {exc}"
    if not isinstance(data, dict):
        return None, f"backend manifest backends/{backend}.yaml must be a mapping"
    consumes = data.get("consumes")
    if not isinstance(consumes, list) or not all(isinstance(c, str) for c in consumes):
        return None, (
            f"backend manifest backends/{backend}.yaml must have a 'consumes' list of "
            f"coefficient-name strings"
        )
    return frozenset(consumes), None


def _resolve_inherited(
    data: dict, sets_by_stem: dict[str, dict]
) -> tuple[frozenset[str], list[str]]:
    """Collect coefficient names available via the ``extends`` chain.

    ``sets_by_stem`` maps a set's filename stem to its parsed document. ``extends`` names
    the stem of a base set. Returns (inherited_names, errors). An ``extends`` naming an
    absent set, a non-string ``extends``, or a cycle is reported as an error and stops the
    walk. Walked iteratively (not recursively) so an arbitrarily deep acyclic chain cannot
    overflow the stack and escape as a traceback; ``seen`` bounds the walk to the number
    of distinct sets.
    """
    names: set[str] = set()
    seen: set[str] = set()
    current = data
    while True:
        parent_stem = current.get("extends")
        if parent_stem is None:
            return frozenset(names), []
        if not isinstance(parent_stem, str):
            return frozenset(names), [f"'extends' must be a string, got {parent_stem!r}"]
        if parent_stem in seen:
            return frozenset(names), [
                f"'extends' cycle detected involving {parent_stem!r}"
            ]
        parent = sets_by_stem.get(parent_stem)
        if parent is None:
            return frozenset(names), [
                f"'extends' names {parent_stem!r}, which is not a known coefficient set "
                f"(expected a file operators/{parent_stem}.yaml)"
            ]
        seen.add(parent_stem)
        parent_coeffs = parent.get("coefficients")
        if isinstance(parent_coeffs, dict):
            names.update(parent_coeffs)
        current = parent


# YAML extensions a coefficient set may use. Matched case-insensitively so a set named
# `roofline.YAML` is not skipped. A set the gate does not SEE is worse than one it
# rejects, so discovery must not depend on an exact-case, single-spelling extension.
YAML_SUFFIXES = frozenset({".yaml", ".yml"})


def _scan_dir(directory: Path) -> list[Path]:
    """Every YAML file under ``directory``, recursively.

    Recursive (``rglob``) so a set dropped in a subdirectory is still validated rather
    than silently unchecked; extension match is case-insensitive over YAML_SUFFIXES.
    """
    return sorted(
        p for p in directory.rglob("*")
        if p.is_file() and p.suffix.lower() in YAML_SUFFIXES
    )


def _discover(paths: list[str]) -> list[Path]:
    files: list[Path] = []
    targets = [Path(p) for p in paths] if paths else DEFAULT_TARGETS
    for target in targets:
        if target.is_dir():
            files.extend(_scan_dir(target))
        elif target.is_file():
            files.append(target)
        else:
            # A missing DEFAULT target (e.g. an as-yet-uncreated operators/) is not an
            # error — there may simply be no real sets yet. An explicitly-named missing
            # path IS reported, as a load error, so a bad argument fails loudly.
            if paths:
                files.append(target)
    return files


def _index_sets(files: list[Path]) -> tuple[dict[str, dict], list[str]]:
    """Load every set once and index by filename STEM so ``extends`` can resolve siblings.

    A set's identity is its stem, so two files with the same stem in different scanned
    directories collide — reported here so the ambiguity fails loudly rather than one
    silently shadowing the other in the index.
    """
    sets_by_stem: dict[str, dict] = {}
    errors: list[str] = []
    for path in files:
        try:
            data = _load_file(path)
        except LOAD_ERRORS:
            continue  # per-file errors are reported in the main validation pass
        if isinstance(data, dict):
            stem = path.stem
            if stem in sets_by_stem:
                errors.append(f"duplicate coefficient-set stem {stem!r}")
            sets_by_stem[stem] = data
    return sets_by_stem, errors


def validate_paths(paths: list[str]) -> tuple[int, list[str]]:
    """Validate the given paths. Returns (exit_code, output_lines)."""
    files = _discover(paths)
    if not files:
        return 1, ["no coefficient sets found to validate"]

    sets_by_stem, index_errors = _index_sets(files)
    lines: list[str] = []
    ok = not index_errors
    for err in index_errors:
        lines.append(f"registry: {err}")

    for path in files:
        rel = path
        try:
            data = _load_file(path)
        except FileNotFoundError:
            ok = False
            lines.append(f"{rel}: file not found")
            continue
        except LOAD_ERRORS as exc:
            ok = False
            lines.append(f"{rel}: could not parse: {exc}")
            continue

        if data is None:
            ok = False
            lines.append(f"{rel}: empty file")
            continue
        if not isinstance(data, dict):
            ok = False
            lines.append(f"{rel}: top level must be a mapping, got {type(data).__name__}")
            continue

        backend = data.get("backend")
        file_errors: list[str] = []
        consumed: frozenset[str] | None = None
        # The schema (check_set) reports a missing/non-string backend; here we additionally
        # resolve the manifest when the backend is a usable string.
        if isinstance(backend, str) and backend.strip():
            consumed, backend_reason = _backend_consumes(backend)
            if backend_reason is not None:
                file_errors.append(backend_reason)

        inherited, extend_errors = _resolve_inherited(data, sets_by_stem)
        file_errors.extend(extend_errors)
        file_errors.extend(check_set(data, consumed, inherited))

        if file_errors:
            ok = False
            for err in sorted(file_errors):
                lines.append(f"{rel}: {err}")
        else:
            lines.append(f"{rel}: OK")

    return (0 if ok else 1), lines


def main(argv: list[str]) -> int:
    exit_code, lines = validate_paths(argv)
    for line in lines:
        print(line)
    if exit_code == 0:
        print("All coefficient sets valid.")
    else:
        print("Validation FAILED.", file=sys.stderr)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
