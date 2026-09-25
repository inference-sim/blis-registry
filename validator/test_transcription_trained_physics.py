"""Transcription test for the trained-physics coefficient set (R2G3).

R2 is *value-preserving*: transcribing the iter29 fit into the registry must not change a
single number. This test is the frozen-snapshot gate that proves it — the committed
``coefficients/trained-physics.yaml`` set is checked to reproduce the currently-shipped
``alpha_coeffs``/``beta_coeffs`` to full precision, so the guarantee is enforced by a test
rather than by eye.

``SHIPPED_ALPHA``/``SHIPPED_BETA`` below are a frozen snapshot of the two positional arrays
as they ship TODAY in inference-sim ``defaults.yaml`` (``trained_physics_coefficients``),
which also equal the frozen ``testdata/trained_physics_iter29.json`` used by the backend's
golden test. The registry cannot reach that repo at test time, so the snapshot is embedded
here as literals — that embedding *is* the frozen snapshot. Values are compared with exact
``==``: both the YAML and the snapshot originate from the same decimal literals, so "full
precision" is literal, not approximate.

The array→name mapping is the meaning the ``defaults.yaml`` comments carry positionally
(β₁ₐ, β₂ₐ, β₃, β₄, β₅, β₆, β₇, β₈, β₁ᵦ, β₂ᵦ, β_EP and α₀, α₁, α₂). The test also pins the
honest-``method`` discipline the issue requires: the two zeros are PRESENT/``not_charged``
(never absent — presence selects the additive formula), β_EP is ``copied`` from
``tp_allreduce_attention``, and ``cross_node_hop_latency`` is ``not_charged``.
"""

from __future__ import annotations

from pathlib import Path

from validator.loader import load_strict
from validator.schema import check_set

REPO_ROOT = Path(__file__).resolve().parent.parent
SET_PATH = REPO_ROOT / "coefficients" / "trained-physics.yaml"

# Frozen snapshot of the shipped positional arrays (defaults.yaml @ 2ebef6a). The registry
# stores each position as a NAMED entry; this maps position → the name the set gives it.
SHIPPED_BETA = {
    "prefill_compute":        0.152128,     # β₁ₐ  [0]
    "decode_compute":         0.0,          # β₂ₐ  [1]  priced-at-zero, PRESENT
    "weight_load":            1.36252915,   # β₃   [2]
    "tp_allreduce_attention": 0.752037,     # β₄   [3]
    "per_layer_overhead":     32.09546717,  # β₅   [4]
    "per_request_overhead":   4.41684444,   # β₆   [5]
    "per_step_overhead":      126.024825,   # β₇   [6]
    "per_moe_layer_overhead": 481.8613888,  # β₈   [7]
    "prefill_kv_read":        0.0,          # β₁ᵦ  [8]  priced-at-zero, PRESENT
    "decode_kv_read":         1.94710771,   # β₂ᵦ  [9]
    "moe_dispatch_alltoall":  0.752037,     # β_EP [10] seeded to β₄
}
SHIPPED_ALPHA = {
    "queueing":                 15563.199579,  # α₀
    "post_decode_fixed":        777.3455,      # α₁
    "output_token_processing":  45.907545,     # α₂
}
# Not in the alpha_coeffs array; a HardwareCalib field, zero on every supported GPU today.
SHIPPED_HOP = {"cross_node_hop_latency": 0.0}

SHIPPED = {**SHIPPED_BETA, **SHIPPED_ALPHA, **SHIPPED_HOP}

# The zeros that must ship PRESENT (not absent) with method not_charged, and the entries
# whose provenance the issue pins.
PRICED_AT_ZERO = {"decode_compute", "prefill_kv_read", "cross_node_hop_latency"}
# β_EP is a seeded copy of β₄, not a fitted number of its own — excluded from the live set.
COPIED = {"moe_dispatch_alltoall"}


def _coeffs_by_name() -> dict:
    data = load_strict(SET_PATH.read_text(encoding="utf-8"))
    by_name: dict = {}
    for item in data["coefficients"]:
        (name, entry), = item.items()
        by_name[name] = entry
    return by_name


def test_set_validates_as_standalone():
    # BC-4: the set passes the R2G1 shape check with no extends/backend key.
    data = load_strict(SET_PATH.read_text(encoding="utf-8"))
    assert check_set(data) == []
    assert data["name"] == "trained-physics"
    assert "extends" not in data and "backend" not in data


def test_names_are_exactly_the_transcribed_set():
    # BC-5: exactly the 14 coefficients — 11 betas + 3 alphas — plus the hop latency the
    # acceptance criteria list. A leaked or missing entry fails here by name.
    assert set(_coeffs_by_name()) == set(SHIPPED)


def test_every_value_matches_shipped_to_full_precision():
    # BC-1: the core R2 invariant — every transcribed value equals the shipped value
    # exactly, with matching numeric type (guards a silent 0 vs 0.0 / int vs float slip).
    coeffs = _coeffs_by_name()
    for name, want in SHIPPED.items():
        got = coeffs[name]["value"]
        assert got == want, f"{name} = {got!r}, shipped {want!r}"
        assert type(got) is type(want), f"{name} type {type(got)} != {type(want)}"


def test_priced_at_zero_are_present_not_absent():
    # BC-2: decode_compute and prefill_kv_read (and the hop latency) ship PRESENT and priced
    # at 0.0 with method not_charged. Presence selects the additive formula; absence would
    # select the MAX form and change output (INV-6) — so "present" is the invariant, not a
    # nicety. Each carries a rationale naming the co-occurrence (schema requires it for
    # not_charged).
    coeffs = _coeffs_by_name()
    for name in PRICED_AT_ZERO:
        entry = coeffs[name]
        assert entry["value"] == 0.0, (name, entry["value"])
        assert entry["method"] == "not_charged", (name, entry["method"])
        assert entry["fitted"] is False, (name, entry["fitted"])
        assert isinstance(entry.get("rationale"), str) and entry["rationale"].strip()


def test_live_coefficients_are_measured_and_fitted():
    # The 8 live betas + 3 alphas are the fitted numbers: method measured, fitted true.
    # The two priced-at-zero betas (not_charged) and β_EP (copied) are excluded — they are
    # the non-fitted numbers the issue's method table calls out separately.
    coeffs = _coeffs_by_name()
    live = (set(SHIPPED_BETA) | set(SHIPPED_ALPHA)) - PRICED_AT_ZERO - COPIED
    assert len(live) == 11, live  # 8 live betas + 3 alphas
    for name in live:
        entry = coeffs[name]
        assert entry["method"] == "measured", (name, entry["method"])
        assert entry["fitted"] is True, (name, entry["fitted"])


def test_moe_dispatch_is_copied_from_tp_allreduce():
    # BC-3: β_EP is seeded to β₄, recorded honestly as copied with copied_from naming the
    # source — and the two values are in fact equal, so the seeding is real, not nominal.
    coeffs = _coeffs_by_name()
    ep = coeffs["moe_dispatch_alltoall"]
    assert ep["method"] == "copied"
    assert ep["copied_from"] == "tp_allreduce_attention"
    assert ep["fitted"] is False
    assert ep["value"] == coeffs["tp_allreduce_attention"]["value"]


def test_units_follow_the_schema_families():
    # The units discipline R2G1 enforces: the four additive-rate betas carry per-layer /
    # per-request / per-step rates; the three alphas are per-request overheads; every
    # correction is dimensionless; the hop latency is per-hop. None is a bare "correction".
    coeffs = _coeffs_by_name()
    expected_units = {
        "prefill_compute": "dimensionless",
        "decode_compute": "dimensionless",
        "weight_load": "dimensionless",
        "tp_allreduce_attention": "dimensionless",
        "per_layer_overhead": "us_per_layer",
        "per_request_overhead": "us_per_request",
        "per_step_overhead": "us_per_step",
        "per_moe_layer_overhead": "us_per_layer",
        "prefill_kv_read": "dimensionless",
        "decode_kv_read": "dimensionless",
        "moe_dispatch_alltoall": "dimensionless",
        "queueing": "us_per_request",
        "post_decode_fixed": "us_per_request",
        "output_token_processing": "us_per_request",
        "cross_node_hop_latency": "us_per_hop",
    }
    for name, want in expected_units.items():
        assert coeffs[name]["units"] == want, (name, coeffs[name]["units"], want)
