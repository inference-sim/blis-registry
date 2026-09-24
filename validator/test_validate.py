"""Behavioral tests for the CoefficientSet validator.

Tests assert on BEHAVIOR (is it rejected, and does the message name the offending
entry/key), not on internal structure, so they survive a refactor of the validator.

The format under test: a standalone, self-identifying document declaring
``kind: CoefficientSet``, a unique ``name``, and ``coefficients`` as a LIST of single-key
maps. There is no ``backend`` and no ``extends`` — either is now an unknown key. The
validator checks SHAPE ONLY; per-backend completeness moved to the simulator-side loader.
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


def a_source(**overrides):
    """A minimal valid {kind, cite, role} provenance object."""
    src = {"kind": "discussion", "cite": "some#ref", "role": "primary"}
    src.update(overrides)
    return src


def a_valid_entry(**overrides):
    """A minimal valid entry; override fields per test."""
    entry = {
        "value": 0.45,
        "units": "dimensionless",
        "method": "literature",
        "fitted": False,
        "sources": [a_source()],
        "rationale": "why",
        "scope": {"hardware": ["H100"]},
    }
    entry.update(overrides)
    return entry


def a_valid_set(coeffs=None, **top):
    """A minimal valid set. ``coeffs`` is the coefficients LIST of single-key maps."""
    data = {
        "kind": "CoefficientSet",
        "name": "roofline-test",
        "coefficients": coeffs
        if coeffs is not None
        else [{"mfu_prefill": a_valid_entry()}, {"mfu_decode": a_valid_entry()}],
    }
    data.update(top)
    return data


def errors_for(entry_name="c", entry=None, **top):
    """Validate a set containing one named entry; return the error list."""
    coeffs = [{entry_name: entry if entry is not None else a_valid_entry()}]
    data = a_valid_set(coeffs=coeffs, **top)
    return check_set(data)


# --- BC-4: strict loader (duplicate keys) ---------------------------------------


def test_loader_rejects_duplicate_keys():
    with pytest.raises(DuplicateKeyError):
        load_strict("a: 1\na: 2\n")


def test_loader_accepts_normal_mapping():
    assert load_strict("a: 1\nb: 2\n") == {"a": 1, "b": 2}


def test_loader_rejects_multi_document_yaml():
    # A multi-document stream (--- separated) must NOT silently load only the first doc;
    # yaml.load raises a ComposerError (a YAMLError), which the CLI reports as a parse
    # failure. Lock this in so a committed file can't smuggle a second hidden document.
    import yaml
    with pytest.raises(yaml.YAMLError):
        load_strict("kind: CoefficientSet\n---\nkind: CoefficientSet\n")


def test_multi_document_file_is_named_error(tmp_path):
    f = tmp_path / "multi.yaml"
    f.write_text("kind: CoefficientSet\nname: s\n---\nkind: CoefficientSet\n")
    code, lines = validate_mod.validate_paths([str(f)])
    assert code == 1
    assert any("could not parse" in ln for ln in lines), lines


def test_loader_accepts_aliases_without_bypassing_duplicate_detection():
    # YAML aliases reuse a value (valid); they must not become a hole in duplicate-key
    # detection, and a duplicate key in flow style is still caught.
    assert load_strict("a: &x 1\nb: *x\n") == {"a": 1, "b": 1}
    with pytest.raises(DuplicateKeyError):
        load_strict("m: {k: 1, k: 2}\n")


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


@pytest.mark.parametrize("bad", [[], {}, 123, None, True])
def test_non_string_units_rejected_without_crash(bad):
    # A non-string (or unhashable) units value must be a NAMED error, not a TypeError
    # from a set-membership test escaping check_set's no-raise contract.
    errs = errors_for("c", a_valid_entry(units=bad))
    assert any("units" in e for e in errs), (bad, errs)


@pytest.mark.parametrize("bad", [[], {}, 123, None, True])
def test_non_string_method_rejected_without_crash(bad):
    errs = errors_for("c", a_valid_entry(method=bad))
    assert any("method" in e for e in errs), (bad, errs)


def test_unhashable_enum_values_do_not_crash_end_to_end(tmp_path):
    # CLI-level regression: units:[] and method:{} in a committed file must be reported,
    # never a traceback.
    coeffs = tmp_path / "coefficients"
    coeffs.mkdir()
    _write_set(
        coeffs, "s.yaml",
        "kind: CoefficientSet\nname: s\ncoefficients:\n"
        "  - mfu_prefill: {value: 0.4, units: [], method: {}, fitted: false, "
        "scope: {hardware: [X]}}\n"
        "  - mfu_decode: {value: 0.3, units: dimensionless, method: measured, "
        "fitted: true, scope: {hardware: [X]}}\n",
    )
    code, lines = validate_mod.validate_paths([str(coeffs)])
    assert code == 1
    joined = "\n".join(lines)
    assert "Traceback" not in joined, joined
    assert any("units" in ln for ln in lines) and any("method" in ln for ln in lines), lines


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
    errs = check_set(a_valid_set(surprise=1))
    assert any("surprise" in e for e in errs), errs


def test_backend_key_now_rejected():
    # `backend` was accepted by the old format; it is now an unknown top-level key.
    errs = check_set(a_valid_set(backend="roofline"))
    assert any("backend" in e and "unknown" in e for e in errs), errs


def test_extends_key_now_rejected():
    # `extends` was accepted by the old format; it is now an unknown top-level key.
    errs = check_set(a_valid_set(extends="base"))
    assert any("extends" in e and "unknown" in e for e in errs), errs


def test_unknown_entry_field_rejected():
    errs = errors_for("c", a_valid_entry(surprise=1))
    assert any("surprise" in e and "c" in e for e in errs), errs


def test_unknown_scope_key_rejected():
    errs = errors_for("c", a_valid_entry(scope={"planet": ["mars"]}))
    assert any("scope" in e and "planet" in e for e in errs), errs


def test_empty_scope_rejected():
    errs = errors_for("c", a_valid_entry(scope={}))
    assert any("scope" in e for e in errs), errs


# --- name (set identity) --------------------------------------------------------


def test_missing_name_rejected():
    data = a_valid_set()
    del data["name"]
    errs = check_set(data)
    assert any("name" in e and "missing" in e for e in errs), errs


@pytest.mark.parametrize("bad_name", [123, "", "   ", None, [], {}])
def test_non_string_or_empty_name_rejected(bad_name):
    data = a_valid_set()
    data["name"] = bad_name
    errs = check_set(data)
    assert any("name" in e for e in errs), (bad_name, errs)


def test_valid_name_accepted():
    errs = check_set(a_valid_set(name="roofline-h100"))
    assert errs == [], errs


# --- coefficients: list of single-key maps --------------------------------------


def test_coefficients_map_form_rejected():
    # The old map form is no longer valid; coefficients must be a list.
    data = a_valid_set()
    data["coefficients"] = {"mfu_prefill": a_valid_entry()}
    errs = check_set(data)
    assert any("coefficients" in e and "list" in e for e in errs), errs


def test_empty_coefficients_rejected():
    data = a_valid_set(coeffs=[])
    errs = check_set(data)
    assert any("coefficients" in e and "empty" in e for e in errs), errs


def test_coefficients_list_form_accepted():
    errs = check_set(a_valid_set())  # default is the list form
    assert errs == [], errs


@pytest.mark.parametrize("item", ["not-a-mapping", 123, [1, 2], None])
def test_coefficient_item_must_be_a_mapping(item):
    data = a_valid_set(coeffs=[item])
    errs = check_set(data)
    assert any("coefficients[0]" in e and "mapping" in e for e in errs), (item, errs)


@pytest.mark.parametrize("bad", [{}, {"a": 1, "b": 2}])
def test_coefficient_item_must_be_single_key(bad):
    # A coefficient item is a {name: entry} map with EXACTLY one key.
    data = a_valid_set(coeffs=[bad])
    errs = check_set(data)
    assert any("coefficients[0]" in e and "one key" in e for e in errs), (bad, errs)


def test_duplicate_coefficient_name_rejected():
    # Two list items with the same name: the loader can't catch this (distinct dicts), so
    # the schema must — a second definition never silently shadows the first.
    coeffs = [{"mfu_prefill": a_valid_entry()}, {"mfu_prefill": a_valid_entry()}]
    errs = check_set(a_valid_set(coeffs=coeffs))
    assert any("duplicate" in e and "mfu_prefill" in e for e in errs), errs


def test_non_string_coefficient_name_rejected():
    coeffs = [{123: a_valid_entry()}]
    errs = check_set(a_valid_set(coeffs=coeffs))
    assert any("must be a string" in e for e in errs), errs


def test_non_mapping_entry_rejected():
    errs = check_set(a_valid_set(coeffs=[{"c": "not-a-mapping"}]))
    assert any("c" in e and "mapping" in e for e in errs), errs


def test_non_mapping_scope_rejected():
    errs = errors_for("c", a_valid_entry(scope="H100"))
    assert any("scope" in e and "mapping" in e for e in errs), errs


# --- CLI-level list-format regressions ------------------------------------------


_SET_BODY = (
    "kind: CoefficientSet\nname: roofline-test\ncoefficients:\n"
    "  - mfu_prefill: {value: 0.4, units: dimensionless, method: measured, "
    "fitted: true, scope: {hardware: [H100]}}\n"
    "  - mfu_decode: {value: 0.3, units: dimensionless, method: measured, "
    "fitted: true, scope: {hardware: [H100]}}\n"
)


def test_duplicate_set_name_across_files_rejected(tmp_path):
    # A set's identity is its `name`. Since identity is a document field (not the unique
    # filename), two files declaring the same `name` must be refused by name — the
    # successor to the old duplicate-stem check.
    coeffs = tmp_path / "coefficients"
    coeffs.mkdir()
    body = (
        "kind: CoefficientSet\nname: roofline-dup\ncoefficients:\n"
        "  - mfu_prefill: {value: 0.4, units: dimensionless, method: measured, "
        "fitted: true, scope: {hardware: [H100]}}\n"
    )
    _write_set(coeffs, "one.yaml", body)
    _write_set(coeffs, "two.yaml", body)
    code, lines = validate_mod.validate_paths([str(coeffs)])
    assert code == 1
    assert any("duplicate set name" in ln and "roofline-dup" in ln for ln in lines), lines


def test_distinct_set_names_across_files_accepted(tmp_path):
    # Two files with DISTINCT names are fine — uniqueness is per name, not a blanket cap.
    coeffs = tmp_path / "coefficients"
    coeffs.mkdir()
    tmpl = (
        "kind: CoefficientSet\nname: {name}\ncoefficients:\n"
        "  - mfu_prefill: {{value: 0.4, units: dimensionless, method: measured, "
        "fitted: true, scope: {{hardware: [H100]}}}}\n"
    )
    _write_set(coeffs, "a.yaml", tmpl.format(name="roofline-a"))
    _write_set(coeffs, "b.yaml", tmpl.format(name="roofline-b"))
    code, lines = validate_mod.validate_paths([str(coeffs)])
    assert code == 0, "\n".join(lines)


def test_duplicate_coefficient_name_reported_end_to_end(tmp_path):
    # A repeated coefficient name in a committed file is reported, not silently shadowed.
    coeffs = tmp_path / "coefficients"
    coeffs.mkdir()
    _write_set(
        coeffs, "dup.yaml",
        "kind: CoefficientSet\nname: dup\ncoefficients:\n"
        "  - mfu_prefill: {value: 0.4, units: dimensionless, method: measured, "
        "fitted: true, scope: {hardware: [H100]}}\n"
        "  - mfu_prefill: {value: 0.3, units: dimensionless, method: measured, "
        "fitted: true, scope: {hardware: [H100]}}\n",
    )
    code, lines = validate_mod.validate_paths([str(coeffs)])
    assert code == 1
    assert any("duplicate coefficient name" in ln and "mfu_prefill" in ln for ln in lines), lines


# --- BC-6: ci95 optional --------------------------------------------------------


def test_ci95_present_is_accepted():
    errs = errors_for("c", a_valid_entry(ci95=[0.4, 0.5]))
    assert errs == [], errs


def test_ci95_absent_is_accepted():
    entry = a_valid_entry()
    assert "ci95" not in entry
    errs = errors_for("c", entry)
    assert errs == [], errs


@pytest.mark.parametrize(
    "bad",
    [
        "banana",                    # non-list
        [0.4],                       # wrong arity
        [0.4, 0.5, 0.6],             # wrong arity
        [float("nan"), 0.5],         # non-finite
        [0.4, float("inf")],         # non-finite
        ["a", "b"],                  # non-numeric
    ],
)
def test_ci95_malformed_rejected(bad):
    errs = errors_for("c", a_valid_entry(ci95=bad))
    assert any("ci95" in e for e in errs), (bad, errs)


def test_ci95_lower_exceeds_upper_rejected():
    errs = errors_for("c", a_valid_entry(ci95=[0.9, 0.1]))
    assert any("ci95" in e and ("exceeds" in e or "lower" in e) for e in errs), errs


# --- Optional-field value validation (C2-C5 hardening) --------------------------


def test_supersedes_must_be_non_empty_string():
    errs = errors_for("c", a_valid_entry(supersedes={"arbitrary": "map"}))
    assert any("supersedes" in e for e in errs), errs


def test_supersedes_valid_string_accepted():
    errs = errors_for("c", a_valid_entry(supersedes="old_coeff"))
    assert errs == [], errs


def test_sources_malformed_rejected_even_when_not_required():
    # method: assumed does not REQUIRE sources, but junk in it must not pass silently.
    entry = a_valid_entry(method="assumed", rationale="why")
    entry["sources"] = 999
    errs = errors_for("c", entry)
    assert any("sources" in e for e in errs), errs


def test_scope_null_value_under_known_key_rejected():
    errs = errors_for("c", a_valid_entry(scope={"hardware": None}))
    assert any("scope" in e and "hardware" in e for e in errs), errs


def test_scope_empty_list_value_rejected():
    errs = errors_for("c", a_valid_entry(scope={"hardware": []}))
    assert any("scope" in e and "hardware" in e for e in errs), errs


@pytest.mark.parametrize("key", ["hardware", "tp", "ep", "nodes_spanned", "model"])
def test_known_scope_keys_accepted(key):
    # Every known scope key (including `model`) is accepted with a non-empty value.
    errs = errors_for("c", a_valid_entry(scope={key: ["x"]}))
    assert errs == [], (key, errs)


@pytest.mark.parametrize("bad_list", [
    [" "],       # whitespace-only string
    [""],        # empty string
    [[]],        # nested empty container
    [None],      # null element
    ["H100", ""],  # one good, one blank — still rejected
])
def test_scope_list_with_empty_or_nonscalar_element_rejected(bad_list):
    # A non-empty scope list must not carry empty/blank/non-scalar elements, which convey
    # no range yet would pass a mere "list is non-empty" check.
    errs = errors_for("c", a_valid_entry(scope={"hardware": bad_list}))
    assert any("scope" in e and "hardware" in e for e in errs), (bad_list, errs)


# --- committed set + CLI --------------------------------------------------------


def _write_set(dirpath, filename, text):
    p = dirpath / filename
    p.write_text(text)
    return p


def test_valid_set_validates_end_to_end(tmp_path):
    # A well-formed set, validated through the CLI over a directory, passes cleanly.
    coeffs = tmp_path / "coefficients"
    coeffs.mkdir()
    _write_set(
        coeffs, "roofline-x.yaml",
        "kind: CoefficientSet\nname: roofline-x\ncoefficients:\n"
        "  - mfu_prefill: {value: 0.45, units: dimensionless, method: measured, "
        "fitted: true, scope: {hardware: [H100]}}\n"
        "  - mfu_decode: {value: 0.30, units: dimensionless, method: measured, "
        "fitted: true, scope: {hardware: [H100]}}\n",
    )
    code, lines = validate_mod.validate_paths([str(coeffs)])
    assert code == 0, "\n".join(lines)


def test_no_arg_default_validates_committed_sets():
    # No-arg run scans coefficients/; the committed roofline sets make this a real
    # validation of committed artifacts, and it must pass on the current tree.
    code, lines = validate_mod.validate_paths([])
    assert code == 0, "\n".join(lines)


def test_committed_coefficients_validate():
    # The committed sets in coefficients/ validate.
    code, lines = validate_mod.validate_paths([str(REPO_ROOT / "coefficients")])
    assert code == 0, "\n".join(lines)


def test_empty_registry_dir_is_a_clean_pass(monkeypatch, tmp_path):
    # An empty default registry (no committed sets) is a clean pass, not a failure —
    # verified by pointing DEFAULT_TARGETS at an empty dir so the check is real.
    empty = tmp_path / "coefficients"
    empty.mkdir()
    monkeypatch.setattr(validate_mod, "DEFAULT_TARGETS", [empty])
    code, lines = validate_mod.validate_paths([])
    assert code == 0, "\n".join(lines)


# --- kind top-level validation --------------------------------------------------


@pytest.mark.parametrize("bad_kind", ["Coefficient", "coefficientset", 123, ""])
def test_wrong_kind_rejected(bad_kind):
    data = a_valid_set()
    data["kind"] = bad_kind
    errs = check_set(data)
    assert any("kind" in e for e in errs), (bad_kind, errs)


def test_missing_kind_rejected():
    data = a_valid_set()
    del data["kind"]
    errs = check_set(data)
    assert any("kind" in e for e in errs), errs


def test_cli_exits_nonzero_on_bad_set(tmp_path):
    bad = tmp_path / "bad.yaml"
    bad.write_text(
        "kind: CoefficientSet\nname: bad\ncoefficients:\n"
        "  - mfu_prefill: {value: 0.4, units: dimensionless, method: assumed, "
        "fitted: false, scope: {hardware: [H100]}}\n"  # assumed w/o rationale
    )
    code, lines = validate_mod.validate_paths([str(bad)])
    assert code == 1
    joined = "\n".join(lines)
    assert "rationale" in joined, joined


def test_cli_rejects_empty_file(tmp_path):
    empty = tmp_path / "empty.yaml"
    empty.write_text("")
    code, lines = validate_mod.validate_paths([str(empty)])
    assert code == 1, lines


# --- Robustness: malformed inputs become NAMED errors, never a stack trace ----------


def test_mixed_type_coefficient_keys_do_not_crash(tmp_path):
    # An unquoted numeric key parses to int; a non-string coefficient name is a NAMED
    # error, and sorting str+int must not raise.
    f = tmp_path / "mixed.yaml"
    f.write_text(
        "kind: CoefficientSet\nname: mixed\ncoefficients:\n"
        "  - 1: {value: 0.4, units: dimensionless, method: measured, fitted: true, "
        "scope: {hardware: [X]}}\n"
    )
    code, lines = validate_mod.validate_paths([str(f)])
    assert code == 1
    joined = "\n".join(lines)
    assert "Traceback" not in joined, joined
    assert any("must be a string" in ln for ln in lines), lines


def test_non_utf8_file_is_named_error(tmp_path):
    f = tmp_path / "nonutf8.yaml"
    f.write_bytes(b"kind: CoefficientSet\nname: s\n\xff\n")
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


# --- Structured sources ({kind, cite, role} objects) ---------------------------


def test_bare_string_sources_rejected():
    # The old string-list shape is no longer valid; sources are structured objects.
    errs = errors_for("c", a_valid_entry(method="literature",
                                         sources=["https://example.invalid"]))
    assert any("sources" in e for e in errs), errs


def test_valid_structured_sources_accepted():
    srcs = [a_source(role="primary"),
            a_source(kind="publication", cite="Paper 2024", role="upper_bound")]
    errs = errors_for("c", a_valid_entry(method="literature", sources=srcs))
    assert errs == [], errs


@pytest.mark.parametrize("bad_field,value", [
    ("kind", "blog"),          # not in SOURCE_KINDS
    ("kind", []),              # unhashable — must not crash the membership test
    ("kind", {}),              # unhashable
    ("role", "footnote"),      # not in SOURCE_ROLES
    ("role", []),              # unhashable
    ("cite", ""),              # empty cite
    ("cite", 123),             # non-string cite
])
def test_source_object_field_validated(bad_field, value):
    src = a_source()
    src[bad_field] = value
    errs = errors_for("c", a_valid_entry(method="literature", sources=[src]))
    assert any("sources[0]" in e for e in errs), (bad_field, errs)


def test_source_unknown_field_rejected():
    src = a_source()
    src["surprise"] = 1
    errs = errors_for("c", a_valid_entry(method="literature", sources=[src]))
    assert any("sources[0]" in e and "surprise" in e for e in errs), errs


def test_source_missing_field_rejected():
    src = a_source()
    del src["role"]
    errs = errors_for("c", a_valid_entry(method="literature", sources=[src]))
    assert any("sources[0]" in e and "role" in e for e in errs), errs


def test_non_string_rationale_rejected():
    errs = errors_for("c", a_valid_entry(method="assumed", rationale=123))
    assert any("rationale" in e for e in errs), errs


def test_rationale_malformed_rejected_even_when_not_required():
    # A measured entry does not REQUIRE rationale, but if present it must be well-formed.
    entry = a_valid_entry(method="measured", fitted=True, rationale=123)
    entry.pop("sources", None)
    errs = errors_for("c", entry)
    assert any("rationale" in e for e in errs), errs


def test_copied_from_malformed_rejected_even_when_not_required():
    # A measured entry does not REQUIRE copied_from, but junk in it must not pass silently.
    entry = a_valid_entry(method="measured", fitted=True, copied_from=[])
    entry.pop("sources", None)
    errs = errors_for("c", entry)
    assert any("copied_from" in e for e in errs), errs


# --- null-valued optional fields (design: explicit null is canonical) -----------


def test_ci95_null_accepted():
    errs = errors_for("c", a_valid_entry(ci95=None))
    assert errs == [], errs


def test_supersedes_null_accepted():
    errs = errors_for("c", a_valid_entry(supersedes=None))
    assert errs == [], errs


def test_validated_unsupported_fields_accepted():
    errs = errors_for("c", a_valid_entry(validated="throughput",
                                         unsupported="absolute_ttft"))
    assert errs == [], errs


def test_cli_subprocess_smoke():
    """The script runs as a subprocess (as CI invokes it) with no args and passes."""
    result = subprocess.run(
        [sys.executable, str(REPO_ROOT / "validator" / "validate.py")],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
