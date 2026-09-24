#!/usr/bin/env python3
"""Validate coefficient sets against the CoefficientSet schema.

Usage:
    validate.py [PATH ...]

Each PATH is a coefficient-set YAML file or a directory scanned recursively for YAML
files (``*.yaml``/``*.yml``, case-insensitive). With no PATH, the real sets in
``coefficients/`` are validated; that directory may be empty until real sets are
transcribed. The tool loads each set, runs the strict schema checks, and prints one line
per problem. Exit code is 0 iff every set is valid — this is the gate CI runs on every
committed set.

A coefficient set is a standalone, self-identifying document: its identity is its
top-level ``name``, not its filename and not an external manifest. There is no inheritance
between sets and no per-backend manifest — the validator checks SHAPE ONLY. Per-backend
completeness ("every coefficient a backend needs is present") is enforced by the
simulator-side loader, which knows what each backend consumes.
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
COEFFICIENTS_DIR = REPO_ROOT / "coefficients"   # the real coefficient sets

# With no explicit path, validate the real sets in coefficients/. That directory may be
# sparse before a given task transcribes its numbers; a missing default dir is not an
# error.
DEFAULT_TARGETS = [COEFFICIENTS_DIR]


# Every way a YAML file can fail to load into usable data. UnicodeDecodeError (a
# ValueError subclass, NOT an OSError) covers a non-UTF-8 file. Complex/unhashable keys
# are named by the loader as a DuplicateKeyError, so TypeError is retained only as a
# defensive catch-all for any unhashable value that might still surface. Catching these
# keeps a malformed committed file a NAMED error instead of a stack trace escaping to CI.
LOAD_ERRORS = (yaml.YAMLError, DuplicateKeyError, OSError, ValueError, TypeError)


def _load_file(path: Path):
    """Load and strictly parse one YAML file. Raises a LOAD_ERRORS member on bad input."""
    return load_strict(path.read_text(encoding="utf-8"))


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
    """Discover the coefficient-set files to validate.

    A directory is scanned recursively; an explicitly-named file is taken as-is. A missing
    DEFAULT target (e.g. an as-yet-uncreated coefficients/) is not an error — there may
    simply be no real sets yet. An explicitly-named missing path IS reported (as a load
    error) so a bad argument fails loudly.
    """
    found: list[Path] = []
    targets = [Path(p) for p in paths] if paths else DEFAULT_TARGETS
    for target in targets:
        if target.is_dir():
            found.extend(_scan_dir(target))
        elif target.is_file():
            found.append(target)
        elif paths:
            found.append(target)
    return found


def validate_paths(paths: list[str]) -> tuple[int, list[str]]:
    """Validate the given paths. Returns (exit_code, output_lines)."""
    files = _discover(paths)
    if not files:
        # No explicit paths => scanning the default registry, which may legitimately be
        # empty (no real sets shipped yet): that is a clean pass, not a failure. An
        # explicit path that matched nothing IS an error — the caller asked for something.
        if paths:
            return 1, ["no coefficient sets found to validate"]
        return 0, ["no coefficient sets to validate (registry is empty)"]

    lines: list[str] = []
    ok = True

    # A set's identity is its `name` (schema.py enforces it is a non-empty string within a
    # file). Since identity moved from the filesystem-unique filename stem to a free-form
    # document field, uniqueness is no longer automatic — two files could declare the same
    # `name` and each pass its own shape check. This registry-level pass refuses that
    # collision by name, the successor to the old duplicate-stem check.
    name_to_path: dict[str, Path] = {}

    for path in files:
        try:
            data = _load_file(path)
        except FileNotFoundError:
            ok = False
            lines.append(f"{path}: file not found")
            continue
        except LOAD_ERRORS as exc:
            ok = False
            lines.append(f"{path}: could not parse: {exc}")
            continue

        if data is None:
            ok = False
            lines.append(f"{path}: empty file")
            continue
        if not isinstance(data, dict):
            ok = False
            lines.append(f"{path}: top level must be a mapping, got {type(data).__name__}")
            continue

        file_errors = check_set(data)
        if file_errors:
            ok = False
            for err in sorted(file_errors):
                lines.append(f"{path}: {err}")
        else:
            lines.append(f"{path}: OK")

        # Track `name` for the cross-file uniqueness check. Only a usable (non-empty
        # string) name participates — a missing/malformed name is already reported by
        # check_set above, so it need not also be reported here as a collision.
        name = data.get("name")
        if isinstance(name, str) and name.strip():
            prior = name_to_path.get(name)
            if prior is not None:
                ok = False
                lines.append(
                    f"{path}: duplicate set name {name!r} (already declared by {prior})"
                )
            else:
                name_to_path[name] = path

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
