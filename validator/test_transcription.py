"""Transcription test for the roofline MFU coefficient sets (R2G2).

R2 is *value-preserving*: transcribing the MFU discounts into the registry must not change
a single number. This test is the frozen-snapshot gate that proves it — each committed
``coefficients/roofline-<gpu>.yaml`` set is checked to reproduce the currently-shipped MFU
values to full precision, so the guarantee is enforced by a test rather than by eye.

``SHIPPED_MFU`` below is a frozen snapshot of the values as they ship TODAY, taken from the
two places they currently live:

  * the catalog hardware files (``hardware/<gpu>.yaml`` in blis-catalog), and
  * the ``mfuPrefill``/``mfuDecode`` fields of each GPU block in the simulator's
    ``hardware_config.json`` (inference-sim),

which agree by construction. The registry cannot reach those repos at test time, so the
snapshot is embedded here as literals — that embedding *is* the frozen snapshot. This is
the one temporary fixture R2 adds; it is retired in R2H1, once the catalog copy is deleted
and the registry becomes the single source of these numbers.

The values are compared with exact ``==``: both the YAML and the snapshot originate from
the same decimal literals (0.45, 0.30, ...), so equality is exact and "full precision" is
literal, not approximate.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from validator.loader import load_strict

REPO_ROOT = Path(__file__).resolve().parent.parent
COEFFICIENTS_DIR = REPO_ROOT / "coefficients"

# Frozen snapshot: the shipped MFU pair and the hardware scope for every GPU BLIS supports
# today. Keyed by coefficient-set filename stem (the set's identity). H200's pair is the
# H100 carry-over it currently ships (UNVALIDATED); the snapshot freezes the shipped number
# regardless of provenance, because the invariant under test is byte-identity, not calibration.
SHIPPED_MFU = {
    "roofline-h100":     {"hardware": "H100",     "mfu_prefill": 0.45, "mfu_decode": 0.30},
    "roofline-a100-sxm": {"hardware": "A100-SXM", "mfu_prefill": 0.38, "mfu_decode": 0.18},
    "roofline-a100-80":  {"hardware": "A100-80",  "mfu_prefill": 0.38, "mfu_decode": 0.18},
    "roofline-l40s":     {"hardware": "L40S",     "mfu_prefill": 0.32, "mfu_decode": 0.08},
    "roofline-h200":     {"hardware": "H200",     "mfu_prefill": 0.45, "mfu_decode": 0.30},
}


def _load_set(stem: str) -> dict:
    path = COEFFICIENTS_DIR / f"{stem}.yaml"
    assert path.is_file(), f"missing roofline set: {path}"
    return load_strict(path.read_text(encoding="utf-8"))


def _coeffs_by_name(data: dict) -> dict:
    """Index a set's ``coefficients`` list (single-key maps) by coefficient name.

    The new format stores coefficients as a LIST of ``{name: entry}`` maps; these tests
    look entries up by name, so collapse the list into a name->entry dict here. A duplicate
    name would silently overwrite, but the schema rejects duplicates, so a set that reaches
    these value checks has unique names.
    """
    by_name: dict = {}
    for item in data["coefficients"]:
        (name, entry), = item.items()
        by_name[name] = entry
    return by_name


def test_every_covered_gpu_has_a_set():
    # Coverage spans every GPU BLIS ships MFU for today — not H100 alone. A missing set is a
    # coverage gap; an unexpected set means the snapshot and the tree have drifted apart.
    # rglob to match validate.py's recursive scan, so the two tools agree on what "the
    # tree" is: a set dropped in a coefficients/ subdirectory is caught here too.
    present = {p.stem for p in COEFFICIENTS_DIR.rglob("roofline-*.yaml")}
    assert present == set(SHIPPED_MFU), (present, set(SHIPPED_MFU))


@pytest.mark.parametrize("stem", sorted(SHIPPED_MFU))
def test_mfu_values_match_shipped_to_full_precision(stem):
    # The core R2 invariant: the transcribed values equal the shipped values exactly.
    expected = SHIPPED_MFU[stem]
    coeffs = _coeffs_by_name(_load_set(stem))
    for name in ("mfu_prefill", "mfu_decode"):
        got = coeffs[name]["value"]
        want = expected[name]
        assert got == want, f"{stem}:{name} = {got!r}, shipped {want!r}"
        # Guard against a silent int/float slip (e.g. 0 vs 0.0 elsewhere): types match too.
        assert type(got) is type(want), f"{stem}:{name} type {type(got)} != {type(want)}"


@pytest.mark.parametrize("stem", sorted(SHIPPED_MFU))
def test_each_entry_scoped_to_its_gpu_with_literature_provenance(stem):
    # Acceptance criteria: every mfu_* entry carries method: literature, its #589 sources,
    # a rationale, fitted: false, and a hardware scope naming exactly its GPU.
    expected = SHIPPED_MFU[stem]
    coeffs = _coeffs_by_name(_load_set(stem))
    # Exactly the two MFU discounts are entries — no more, no less. This is also the
    # vendor-spec-exclusion guard: a TFlopsPeak/TFlopsFP8/BwPeakTBs key leaking in as an
    # entry (they are catalog references, cited not copied) fails here by name.
    assert set(coeffs) == {"mfu_prefill", "mfu_decode"}, coeffs.keys()
    for name, entry in coeffs.items():
        assert entry["method"] == "literature", (stem, name)
        assert entry["fitted"] is False, (stem, name)
        assert entry["units"] == "dimensionless", (stem, name, entry.get("units"))
        assert entry["scope"] == {"hardware": [expected["hardware"]]}, (stem, name)
        assert isinstance(entry.get("rationale"), str) and entry["rationale"].strip()
        # #589 is the PRIMARY basis, not merely present: the discount is a #589 estimate,
        # so a demotion to a supporting role would misstate provenance.
        cite_roles = [(s["cite"], s["role"]) for s in entry["sources"]]
        assert ("inference-sim#589", "primary") in cite_roles, (stem, name, cite_roles)
