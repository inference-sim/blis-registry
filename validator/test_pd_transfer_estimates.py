"""Transcription test for the pd-transfer-estimates set (blis-registry#10, R2 follow-up).

The two PD-transfer numbers `inference-sim` prices KV-cache transfer with are ESTIMATES, not
catalog nominal facts, so by the R2 catalog↔registry split they live here:

  * ``pd_transfer_base_latency`` — the ``--pd-transfer-base-latency`` default (0.05 ms), an
    order-of-magnitude modeling placeholder. Recorded in ``us_per_transfer`` (µs), the schema's
    per-transfer-latency unit — the ``UNITS`` enum has no ``ms`` member — so the physical value
    is 0.05 ms == 50.0 µs. ``method: assumed``. This is the SOLE owner: blis-catalog#12 removes
    ``PDTransferBaseLatencyMs`` from the fabric classes, so there is no catalog term to compose.
  * ``pd_transfer_overhead`` — the effective ÷ nominal correction for KV transfer over a fabric.
    Ships at the multiplicative identity 1.0 (a placeholder, not a measurement) until a campaign
    measures it; ``method: assumed``. Composes with the catalog nominal ``InterNodeBwGBps``.

This test pins the DATA (value / unit / method / scope) against the values the issue specifies.
It does NOT exercise the simulator, so it does not prove runtime step-time parity — that gate is
the N-track's (R2 tracker), as with the other transcription tests in this directory.

The unit choice (``us_per_transfer`` rather than the issue's literal ``ms``) is a deliberate
CORRECTION: ``ms`` is not a member of the schema's closed ``UNITS`` enum, so a coefficient
carrying it would be rejected. This mirrors ``output_token_processing`` in the since-removed
``trained-physics`` set (semantically µs/token, recorded under an in-schema unit with the
value preserved and the reason stated in its rationale).
"""

from __future__ import annotations

from pathlib import Path

from validator.loader import load_strict
from validator.schema import check_set

REPO_ROOT = Path(__file__).resolve().parent.parent
SET_PATH = REPO_ROOT / "coefficients" / "pd-transfer-estimates.yaml"

# The two coefficients this set owns — exactly these, no more, no less.
EXPECTED_NAMES = {"pd_transfer_base_latency", "pd_transfer_overhead"}

# 0.05 ms == 50.0 µs. The registry has no `ms` unit, so the base latency is recorded in
# `us_per_transfer` with its physical value preserved by this conversion.
BASE_LATENCY_US = 50.0

# The overhead placeholder is an exact multiplicative identity (effective == nominal until a
# measurement lowers it), so it does not perturb any product it multiplies.
OVERHEAD_IDENTITY = 1.0


def _coeffs_by_name() -> dict:
    data = load_strict(SET_PATH.read_text(encoding="utf-8"))
    by_name: dict = {}
    for item in data["coefficients"]:
        (name, entry), = item.items()
        by_name[name] = entry
    return by_name


def test_set_validates_as_standalone():
    # The set passes the strict schema and is name-identified, with no inheritance keys.
    data = load_strict(SET_PATH.read_text(encoding="utf-8"))
    assert check_set(data) == []
    assert data["name"] == "pd-transfer-estimates"
    assert "extends" not in data and "backend" not in data


def test_names_are_exactly_the_two_pd_estimates():
    # Exactly the base-latency and overhead entries — a leaked or missing name fails here.
    assert set(_coeffs_by_name()) == EXPECTED_NAMES


def test_base_latency_value_unit_and_provenance():
    # The 0.05 ms placeholder, recorded as 50.0 µs under the schema's per-transfer unit.
    entry = _coeffs_by_name()["pd_transfer_base_latency"]
    assert entry["value"] == BASE_LATENCY_US, entry["value"]
    assert type(entry["value"]) is float, type(entry["value"])
    assert entry["units"] == "us_per_transfer", entry["units"]
    assert entry["method"] == "assumed", entry["method"]
    assert entry["fitted"] is False, entry["fitted"]


def test_overhead_value_unit_and_provenance():
    # The effective ÷ nominal correction, an assumed dimensionless identity until measured.
    entry = _coeffs_by_name()["pd_transfer_overhead"]
    assert entry["value"] == OVERHEAD_IDENTITY, entry["value"]
    assert type(entry["value"]) is float, type(entry["value"])
    assert entry["units"] == "dimensionless", entry["units"]
    assert entry["method"] == "assumed", entry["method"]
    assert entry["fitted"] is False, entry["fitted"]


def test_both_are_assumed_not_measured():
    # Neither number is a measurement yet: both are `assumed` and unfitted. When a campaign
    # measures either, it moves to `method: measured` — this test then flips deliberately.
    for name, entry in _coeffs_by_name().items():
        assert entry["method"] == "assumed", (name, entry["method"])
        assert entry["fitted"] is False, (name, entry["fitted"])


def test_scope_is_all_supported_hardware():
    # Both terms derive from GLOBAL CLI flags, not a per-GPU fit, so they hold across every
    # supported GPU — NOT H100-only. (A set scoped to one GPU is scoped that way because it was
    # fitted there; these are unfitted placeholders that apply wherever the simulator
    # runs.) There is no `fabric`/`network` scope key today to express the per-fabric variation
    # the eventual measured overhead will carry (issue #10), so all-hardware is the honest scope.
    all_hw = {"hardware": ["H100", "H200", "A100-80", "A100-SXM", "L40S"]}
    for name, entry in _coeffs_by_name().items():
        assert entry["scope"] == all_hw, (name, entry["scope"])
