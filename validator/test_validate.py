"""Behavioral tests for the CoefficientSet validator.

Each test maps to a behavioral contract (BC-n) from the plan, which maps to an
acceptance criterion of registry issue #1. Tests assert on BEHAVIOR (is it rejected, and
does the message name the offending entry/key), not on internal structure, so they
survive a refactor of the validator.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from validator.loader import DuplicateKeyError, load_strict
from validator.schema import check_set
from validator import validate as validate_mod

REPO_ROOT = Path(__file__).resolve().parent.parent

# roofline consumes exactly these; used for most single-entry tests.
ROOFLINE = frozenset({"mfu_prefill", "mfu_decode"})
NO_INHERIT = frozenset()


def a_valid_entry(**overrides):
    """A minimal valid entry; override fields per test."""
    entry = {
        "value": 0.45,
        "units": "dimensionless",
        "method": "literature",
        "fitted": False,
        "sources": ["https://example.invalid/src"],
        "rationale": "why",
        "scope": {"hardware": ["H100"]},
    }
    entry.update(overrides)
    return entry


def a_valid_set(coeffs=None, **top):
    data = {
        "name": "t",
        "backend": "roofline",
        "coefficients": coeffs
        if coeffs is not None
        else {"mfu_prefill": a_valid_entry(), "mfu_decode": a_valid_entry()},
    }
    data.update(top)
    return data


def errors_for(entry_name="c", entry=None, consumed=None, inherited=NO_INHERIT, **top):
    """Validate a set containing one named entry; return the error list."""
    coeffs = {entry_name: entry if entry is not None else a_valid_entry()}
    # Ensure the consumed check doesn't fire unless a test asks for it.
    data = a_valid_set(coeffs=coeffs, **top)
    return check_set(data, consumed, inherited)


# --- BC-4: strict loader (duplicate keys) ---------------------------------------


def test_loader_rejects_duplicate_keys():
    with pytest.raises(DuplicateKeyError):
        load_strict("a: 1\na: 2\n")


def test_loader_accepts_normal_mapping():
    assert load_strict("a: 1\nb: 2\n") == {"a": 1, "b": 2}


# --- BC-1: missing required entry fields ----------------------------------------


@pytest.mark.parametrize("field", ["method", "units", "scope", "value", "fitted"])
def test_missing_required_field_is_rejected_naming_entry(field):
    entry = a_valid_entry()
    del entry[field]
    errs = errors_for("mfu_prefill", entry)
    assert any(field in e and "mfu_prefill" in e for e in errs), errs


# --- BC-7: enum + type validation -----------------------------------------------


def test_bad_units_rejected():
    errs = errors_for("c", a_valid_entry(units="furlongs"))
    assert any("units" in e and "c" in e for e in errs), errs


def test_bad_method_rejected():
    errs = errors_for("c", a_valid_entry(method="guessed"))
    assert any("method" in e and "c" in e for e in errs), errs


def test_non_bool_fitted_rejected():
    errs = errors_for("c", a_valid_entry(fitted="yes"))
    assert any("fitted" in e for e in errs), errs


def test_fitted_true_requires_measured():
    # literature + fitted:true is contradictory.
    errs = errors_for("c", a_valid_entry(method="literature", fitted=True))
    assert any("fitted" in e and "measured" in e for e in errs), errs


def test_fitted_true_with_measured_is_ok():
    entry = a_valid_entry(method="measured", fitted=True)
    entry.pop("sources", None)  # measured needs no companion
    errs = errors_for("c", entry)
    assert errs == [], errs


# --- BC-3: zero value -----------------------------------------------------------


def test_zero_value_rejected_for_non_not_charged():
    errs = errors_for("c", a_valid_entry(value=0))
    assert any(("value 0" in e) and ("not_charged" in e) for e in errs), errs


def test_zero_value_ok_for_not_charged():
    entry = a_valid_entry(value=0, method="not_charged", units="us_per_hop")
    entry.pop("sources", None)
    errs = errors_for("c", entry)
    assert errs == [], errs


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf"), "0.4", True])
def test_non_finite_or_non_numeric_value_rejected(bad):
    entry = a_valid_entry(value=bad)
    errs = errors_for("c", entry)
    assert any("value" in e for e in errs), (bad, errs)


# --- BC-2: required-by-method companions ----------------------------------------


def test_non_measured_requires_rationale():
    entry = a_valid_entry(method="assumed")
    entry.pop("rationale")
    errs = errors_for("c", entry)
    assert any("rationale" in e for e in errs), errs


def test_literature_requires_sources():
    entry = a_valid_entry(method="literature")
    entry.pop("sources")
    errs = errors_for("c", entry)
    assert any("sources" in e for e in errs), errs


def test_vendor_spec_requires_sources():
    entry = a_valid_entry(method="vendor_spec")
    entry.pop("sources")
    errs = errors_for("c", entry)
    assert any("sources" in e for e in errs), errs


def test_copied_requires_copied_from():
    entry = a_valid_entry(method="copied")
    errs = errors_for("c", entry)
    assert any("copied_from" in e for e in errs), errs


def test_measured_needs_no_companion():
    entry = a_valid_entry(method="measured", fitted=True)
    entry.pop("sources", None)
    errs = errors_for("c", entry)
    assert errs == [], errs


# --- BC-4: strict top-level / entry / scope keys --------------------------------


def test_unknown_top_level_key_rejected():
    errs = check_set(a_valid_set(surprise=1), ROOFLINE, NO_INHERIT)
    assert any("surprise" in e for e in errs), errs


def test_unknown_entry_field_rejected():
    errs = errors_for("c", a_valid_entry(surprise=1))
    assert any("surprise" in e and "c" in e for e in errs), errs


def test_unknown_scope_key_rejected():
    errs = errors_for("c", a_valid_entry(scope={"planet": ["mars"]}))
    assert any("scope" in e and "planet" in e for e in errs), errs


def test_empty_scope_rejected():
    errs = errors_for("c", a_valid_entry(scope={}))
    assert any("scope" in e for e in errs), errs


# --- BC-5: backend consumed-names -----------------------------------------------


def test_backend_omitted_coefficient_refused_by_name():
    # roofline consumes mfu_prefill + mfu_decode; provide only one.
    coeffs = {"mfu_prefill": a_valid_entry()}
    errs = check_set(a_valid_set(coeffs=coeffs), ROOFLINE, NO_INHERIT)
    assert any("mfu_decode" in e for e in errs), errs


def test_present_but_not_consumed_is_allowed():
    coeffs = {
        "mfu_prefill": a_valid_entry(),
        "mfu_decode": a_valid_entry(),
        "extra_term": a_valid_entry(),  # not in ROOFLINE consumes
    }
    errs = check_set(a_valid_set(coeffs=coeffs), ROOFLINE, NO_INHERIT)
    assert errs == [], errs


def test_inherited_coefficient_satisfies_consumed():
    # Child provides only mfu_decode; mfu_prefill comes from the extends chain.
    coeffs = {"mfu_decode": a_valid_entry()}
    errs = check_set(
        a_valid_set(coeffs=coeffs, extends="base"),
        ROOFLINE,
        inherited=frozenset({"mfu_prefill"}),
    )
    assert errs == [], errs


# --- BC-6: ci95 optional --------------------------------------------------------


def test_ci95_present_is_accepted():
    errs = errors_for("c", a_valid_entry(ci95=[0.4, 0.5]))
    assert errs == [], errs


def test_ci95_absent_is_accepted():
    entry = a_valid_entry()
    assert "ci95" not in entry
    errs = errors_for("c", entry)
    assert errs == [], errs


# --- BC-8: extends resolution (CLI-level) ---------------------------------------


def test_unknown_extends_rejected():
    data = a_valid_set(extends="nope")
    inherited, errs = validate_mod._resolve_inherited(data, {})
    assert any("nope" in e for e in errs), errs
    assert inherited == frozenset()


def test_extends_cycle_rejected():
    a = {"name": "a", "backend": "roofline", "extends": "b", "coefficients": {}}
    b = {"name": "b", "backend": "roofline", "extends": "a", "coefficients": {}}
    _, errs = validate_mod._resolve_inherited(a, {"a": a, "b": b})
    assert any("cycle" in e for e in errs), errs


# --- BC-9 / BC-10: committed set + CLI ------------------------------------------


def test_committed_example_set_validates():
    code, lines = validate_mod.validate_paths([str(REPO_ROOT / "operators")])
    assert code == 0, "\n".join(lines)


def test_cli_exits_nonzero_on_bad_set(tmp_path):
    bad = tmp_path / "bad.yaml"
    bad.write_text(
        "name: b\nbackend: roofline\ncoefficients:\n"
        "  mfu_prefill: {value: 0.4, units: dimensionless, method: assumed, "
        "fitted: false, scope: {hardware: [H100]}}\n"  # assumed w/o rationale + missing mfu_decode
    )
    code, lines = validate_mod.validate_paths([str(bad)])
    assert code == 1
    joined = "\n".join(lines)
    assert "rationale" in joined and "mfu_decode" in joined, joined


def test_cli_rejects_empty_file(tmp_path):
    empty = tmp_path / "empty.yaml"
    empty.write_text("")
    code, lines = validate_mod.validate_paths([str(empty)])
    assert code == 1, lines


# --- Robustness: malformed inputs become NAMED errors, never a stack trace ----------


def test_mixed_type_coefficient_keys_do_not_crash(tmp_path):
    # An unquoted numeric key parses to int; sorting str+int must not raise.
    f = tmp_path / "mixed.yaml"
    f.write_text("name: m\nbackend: roofline\ncoefficients:\n  1: {}\n  a: {}\n")
    code, lines = validate_mod.validate_paths([str(f)])
    assert code == 1
    joined = "\n".join(lines)
    assert "could not parse" not in joined or "Traceback" not in joined
    assert any("must be a string" in ln for ln in lines), lines


def test_non_utf8_file_is_named_error(tmp_path):
    f = tmp_path / "nonutf8.yaml"
    f.write_bytes(b"name: x\nbackend: roofline\n\xff\n")
    code, lines = validate_mod.validate_paths([str(f)])
    assert code == 1
    assert any("could not parse" in ln for ln in lines), lines


def test_complex_yaml_key_is_named_error(tmp_path):
    f = tmp_path / "complex.yaml"
    f.write_text("? [a, b]\n: 1\n")
    code, lines = validate_mod.validate_paths([str(f)])
    assert code == 1
    assert any("could not parse" in ln or "complex" in ln for ln in lines), lines


def test_non_mapping_root_is_named_error(tmp_path):
    f = tmp_path / "list.yaml"
    f.write_text("- a\n- b\n")
    code, lines = validate_mod.validate_paths([str(f)])
    assert code == 1
    assert any("mapping" in ln for ln in lines), lines


def test_malformed_manifest_is_reported_not_treated_as_missing(tmp_path, monkeypatch):
    # A present-but-malformed manifest must be named as malformed and must NOT silently
    # skip the consumed-names check by looking "missing".
    backends = tmp_path / "backends"
    backends.mkdir()
    (backends / "roofline.yaml").write_text("consumes: not-a-list\n")
    monkeypatch.setattr(validate_mod, "BACKENDS_DIR", backends)
    names, reason = validate_mod._backend_consumes("roofline")
    assert names is None
    assert reason is not None and "consumes" in reason and "roofline.yaml" in reason


def test_missing_manifest_distinguished_from_malformed(tmp_path, monkeypatch):
    backends = tmp_path / "backends"
    backends.mkdir()
    monkeypatch.setattr(validate_mod, "BACKENDS_DIR", backends)
    names, reason = validate_mod._backend_consumes("ghost")
    assert names is None
    assert reason is not None and "no manifest" in reason


def test_non_string_sources_rejected():
    errs = errors_for("c", a_valid_entry(method="literature", sources=[123, None]))
    assert any("sources" in e for e in errs), errs


def test_non_string_rationale_rejected():
    errs = errors_for("c", a_valid_entry(method="assumed", rationale=123))
    assert any("rationale" in e for e in errs), errs


def test_committed_extends_set_validates():
    # BC-8 end-to-end over a COMMITTED set: example-extends inherits mfu_prefill.
    code, lines = validate_mod.validate_paths([str(REPO_ROOT / "operators")])
    assert code == 0, "\n".join(lines)


def test_cli_subprocess_smoke():
    """The script runs as a subprocess (as CI invokes it) and passes on operators/."""
    result = subprocess.run(
        [sys.executable, str(REPO_ROOT / "validator" / "validate.py"),
         str(REPO_ROOT / "operators")],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
