"""Transcription test for the estimated-coefficient sets (R2G3b).

R2 is *value-preserving*: transcribing the estimated numbers into the registry must not change
a single number. This test is the frozen-snapshot gate that proves it for the two sets R2G3b
authors — the LoRA adapter-cost set (family #1) and the legacy CPU↔GPU transfer set (family
#2) — checked to reproduce the currently-shipped values to full precision, so the guarantee is
enforced by a test rather than by eye.

``SHIPPED_LORA``/``SHIPPED_TRANSFER`` below are a frozen snapshot of the values as they ship
TODAY in inference-sim: the ``lora:`` block of ``defaults.yaml`` (family #1) and the
``--kv-transfer-bandwidth``/``--kv-transfer-base-latency`` flag defaults in ``cmd/root.go``
(family #2). The registry cannot reach that repo at test time, so the snapshot is embedded here
as literals — that embedding *is* the frozen snapshot. Values are compared with exact ``==``
and matching numeric TYPE (guarding a silent int/float slip, e.g. 0 vs 0.0), because both the
YAML and the snapshot originate from the same decimal literals.

Family #3 (tier-deviation factors) authors NO set: no committed config — in inference-sim or
the catalog — ships a value for saturation_queue_depth / single_transfer_fraction /
latency_jitter_stddev / direct_io (only the hardcoded ramp-off sentinel Qsat=1/f1=1.0/σ=0 and
test fixtures exist). A value-preserving transcription therefore produces nothing, matching the
issue's "(all absent, so all inert)". There is nothing to snapshot, so this file covers the two
sets that carry data.
"""

from __future__ import annotations

from pathlib import Path

from validator.loader import load_strict
from validator.schema import check_set

REPO_ROOT = Path(__file__).resolve().parent.parent
COEFFICIENTS_DIR = REPO_ROOT / "coefficients"
LORA_PATH = COEFFICIENTS_DIR / "lora-adapter-costs.yaml"
TRANSFER_PATH = COEFFICIENTS_DIR / "legacy-kv-transfer.yaml"

# Frozen snapshot of the shipped LoRA values (defaults.yaml `lora:` block). The registry
# stores each shipped field as a NAMED entry; the k6/k7 tier maps at ranks 8/16/32 expand to
# one entry per (coefficient, rank). Values are floats, as they ship.
SHIPPED_LORA = {
    "load_base_latency_us":    1500.0,      # load_base_latency_us
    "load_bandwidth_bytes_us": 2000000.0,   # load_bandwidth_bytes_us (2.0e6)
    "footprint_bytes_per_rank": 2000000.0,  # footprint_bytes_per_rank (2.0e6)
    "step_overhead_k6_rank8":  0.02,        # step_overhead_tiers[8].k6
    "step_overhead_k6_rank16": 0.035,       # step_overhead_tiers[16].k6
    "step_overhead_k6_rank32": 0.06,        # step_overhead_tiers[32].k6
    "step_overhead_k7_rank8":  1.0,         # step_overhead_tiers[8].k7
    "step_overhead_k7_rank16": 1.0,         # step_overhead_tiers[16].k7
    "step_overhead_k7_rank32": 1.0,         # step_overhead_tiers[32].k7
}

# Frozen snapshot of the shipped legacy-transfer flag defaults (cmd/root.go). Bandwidth is a
# float default (100.0); base-latency is an int64 default (0), preserved as int here so a
# silent int→float slip fails the type check below.
SHIPPED_TRANSFER = {
    "kv_transfer_bandwidth":    100.0,  # --kv-transfer-bandwidth default (float)
    "kv_transfer_base_latency": 0,      # --kv-transfer-base-latency default (int)
}

# The LoRA numbers derived from the Agullo Digital Twin (method: literature); the two byte/
# rate numbers are assumed (a modelling choice, not a DT measurement — #1826).
LORA_LITERATURE = {
    "load_base_latency_us",
    "step_overhead_k6_rank8", "step_overhead_k6_rank16", "step_overhead_k6_rank32",
    "step_overhead_k7_rank8", "step_overhead_k7_rank16", "step_overhead_k7_rank32",
}
LORA_ASSUMED = {"load_bandwidth_bytes_us", "footprint_bytes_per_rank"}


def _coeffs_by_name(path: Path) -> dict:
    data = load_strict(path.read_text(encoding="utf-8"))
    by_name: dict = {}
    for item in data["coefficients"]:
        (name, entry), = item.items()
        by_name[name] = entry
    return by_name


# --- both sets validate as standalone ------------------------------------------------


def test_both_sets_validate_as_standalone():
    # The R2G1 shape check passes with no extends/backend key, and the identities are the
    # names the loader will look them up by.
    for path, name in ((LORA_PATH, "lora-adapter-costs"),
                        (TRANSFER_PATH, "legacy-kv-transfer")):
        data = load_strict(path.read_text(encoding="utf-8"))
        assert check_set(data) == [], (name, check_set(data))
        assert data["name"] == name
        assert "extends" not in data and "backend" not in data


# --- names are exactly the transcribed set -------------------------------------------


def test_lora_names_are_exactly_the_transcribed_set():
    # Exactly the 9 shipped LoRA numbers — a leaked or missing entry fails here by name.
    assert set(_coeffs_by_name(LORA_PATH)) == set(SHIPPED_LORA)


def test_transfer_names_are_exactly_the_transcribed_set():
    assert set(_coeffs_by_name(TRANSFER_PATH)) == set(SHIPPED_TRANSFER)


# --- values match shipped to full precision (value AND numeric type) -----------------


def test_lora_values_match_shipped_to_full_precision():
    coeffs = _coeffs_by_name(LORA_PATH)
    for name, want in SHIPPED_LORA.items():
        got = coeffs[name]["value"]
        assert got == want, f"{name} = {got!r}, shipped {want!r}"
        assert type(got) is type(want), f"{name} type {type(got)} != {type(want)}"


def test_transfer_values_match_shipped_to_full_precision():
    coeffs = _coeffs_by_name(TRANSFER_PATH)
    for name, want in SHIPPED_TRANSFER.items():
        got = coeffs[name]["value"]
        assert got == want, f"{name} = {got!r}, shipped {want!r}"
        # base_latency ships as int 0; guard the int/float slip explicitly.
        assert type(got) is type(want), f"{name} type {type(got)} != {type(want)}"


# --- methods, fitted discipline, and the honest-asymmetry scope ----------------------


def test_lora_methods_and_fitted():
    # DT-derived numbers are literature; the two byte/rate numbers are assumed (a modelling
    # choice, not a measurement — #1826). Nothing here is in-repo fitted, so fitted is false
    # everywhere (the DT does not fit per rank: `rank = x[1]  # not used`).
    coeffs = _coeffs_by_name(LORA_PATH)
    for name in LORA_LITERATURE:
        assert coeffs[name]["method"] == "literature", (name, coeffs[name]["method"])
    for name in LORA_ASSUMED:
        assert coeffs[name]["method"] == "assumed", (name, coeffs[name]["method"])
    for name, entry in coeffs.items():
        assert entry["fitted"] is False, (name, entry["fitted"])


def test_lora_scope_and_honest_asymmetry():
    # Every entry is scoped to the two DT-validated configs and carries the validated/
    # unsupported asymmetry (throughput validated, absolute TTFT unsupported — SC-007).
    coeffs = _coeffs_by_name(LORA_PATH)
    for name, entry in coeffs.items():
        assert entry["scope"] == {
            "model": ["Llama-3.1-8B-Instruct", "Qwen-2.5-7B-Instruct"]
        }, (name, entry["scope"])
        assert entry.get("validated") == "throughput", (name, entry.get("validated"))
        assert entry.get("unsupported") == "absolute_ttft", (name, entry.get("unsupported"))


def test_transfer_methods_and_scope():
    # The bandwidth default is assumed (a round default, no source); the zero base-latency is
    # not_charged (the zero-value rule). Neither is fitted. Both scope to the cpu_dram bus.
    coeffs = _coeffs_by_name(TRANSFER_PATH)
    assert coeffs["kv_transfer_bandwidth"]["method"] == "assumed"
    assert coeffs["kv_transfer_base_latency"]["method"] == "not_charged"
    for name, entry in coeffs.items():
        assert entry["fitted"] is False, (name, entry["fitted"])
        assert entry["scope"] == {"hardware": ["cpu_dram"]}, (name, entry["scope"])
