"""Transcription test for the trained-physics coefficient sets (R2G3).

R2 is *value-preserving*: transcribing the trained-physics coefficients into the registry
must not change a single number. This test is the frozen-snapshot gate that proves it —
each committed ``operators/trained-physics-<gpu>.yaml`` set is checked to reproduce the
currently-shipped ``alpha_coeffs`` / ``beta_coeffs`` to full precision, so the guarantee is
enforced by a test rather than by eye (the same shape as ``test_transcription.py`` for the
roofline MFU sets).

``SHIPPED_ALPHAS`` / ``SHIPPED_BETAS`` below are a frozen snapshot of the values as they ship
TODAY, taken from inference-sim's ``defaults.yaml`` at the pinned commit the R2G3 issue links:

    defaults.yaml @ 2ebef6a17b95c3c41480f0f90529bb95cf9a0e5b, L163-L186
      alpha_coeffs: [15563.199579, 777.3455, 45.907545]
      beta_coeffs:  [0.152128, 0.0, 1.36252915, 0.752037, 32.09546717, 4.41684444,
                     126.024825, 481.8613888, 0.0, 1.94710771, 0.752037]

The registry cannot reach inference-sim at test time, so the snapshot is embedded here as
literals — that embedding *is* the frozen snapshot, exactly as the roofline MFU snapshot is.

IMPORTANT — this snapshot is ELEVEN betas, not ten. Production ships eleven values (the
eleventh, ``beta_EP``, seeded equal to ``beta_4``), which runs the ADDITIVE-with-zero form.
The ``training``-branch ``trained_physics_iter29.json`` fixture carries only ten (``beta_EP``
absent) and would exercise the back-fill path production does not run. We freeze against the
shipped ``defaults.yaml`` array, which is the value R2 must preserve.

Values are compared with exact ``==`` and a type check: the YAML literals and these snapshot
literals originate from the same decimals, so equality is exact and "full precision" is
literal, not approximate. A ``0`` vs ``0.0`` slip is caught by the type assertion.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from validator.loader import load_strict

REPO_ROOT = Path(__file__).resolve().parent.parent
OPERATORS_DIR = REPO_ROOT / "operators"

# Frozen snapshot of defaults.yaml@2ebef6a L163-L186. Indices are 1-based positions, which
# map directly to the coefficient names the trained-physics manifest consumes (alpha_coeffs[0]
# -> alpha_1, beta_coeffs[0] -> beta_1, ...).
SHIPPED_ALPHAS = [15563.199579, 777.3455, 45.907545]
SHIPPED_BETAS = [
    0.152128,      # beta_1  (b1a) prefill compute
    0.0,           # beta_2  (b2a) decode compute -- not_charged
    1.36252915,    # beta_3       weight load
    0.752037,      # beta_4       TP all-reduce (attention)
    32.09546717,   # beta_5       per-layer overhead
    4.41684444,    # beta_6       per-request scheduling overhead
    126.024825,    # beta_7       per-step constant overhead
    481.8613888,   # beta_8       per-MoE-layer overhead
    0.0,           # beta_9  (b1b) prefill kv-read -- not_charged
    1.94710771,    # beta_10 (b2b) decode kv-read
    0.752037,      # beta_11 (bEP) MoE dispatch/combine comm -- copied from beta_4
]

# The two step-time positions that ship at 0.0 and MUST be declared (not omitted): omitting
# prefill_kv_read (beta_9) selects the MAX form, declaring it 0.0 selects the ADDITIVE form,
# and the two differ ~1.67x on memory-bound traffic (north-star Problem 4). Plus the
# cross-node hop latency, which is priced at zero on every GPU.
ZERO_NOT_CHARGED = ("beta_2", "beta_9", "alpha_hop")

# Every GPU BLIS ships a roofline set for today (test_transcription.py::SHIPPED_MFU). A
# trained-physics set extends its GPU's roofline set, so coverage must match one-for-one.
# Keyed by coefficient-set filename stem (the set's identity) -> its hardware scope token.
TP_GPUS = {
    "trained-physics-h100":     "H100",
    "trained-physics-a100-sxm": "A100-SXM",
    "trained-physics-a100-80":  "A100-80",
    "trained-physics-l40s":     "L40S",
    "trained-physics-h200":     "H200",
}

# The roofline stem each trained-physics set must extend to inherit its GPU's MFU pair.
ROOFLINE_PARENT = {stem: stem.replace("trained-physics-", "roofline-") for stem in TP_GPUS}


def _load_set(stem: str) -> dict:
    path = OPERATORS_DIR / f"{stem}.yaml"
    assert path.is_file(), f"missing trained-physics set: {path}"
    return load_strict(path.read_text(encoding="utf-8"))


def test_every_covered_gpu_has_a_trained_physics_set():
    # Coverage spans every GPU BLIS ships a roofline set for. A missing set is a coverage
    # gap; an unexpected one means the snapshot and the tree have drifted apart. rglob to
    # match validate.py's recursive scan, so a set dropped in a subdirectory is caught too.
    present = {p.stem for p in OPERATORS_DIR.rglob("trained-physics-*.yaml")}
    assert present == set(TP_GPUS), (present, set(TP_GPUS))


@pytest.mark.parametrize("stem", sorted(TP_GPUS))
def test_beta_values_match_shipped_to_full_precision(stem):
    # The core R2 invariant for the eleven betas: transcribed values equal shipped exactly.
    coeffs = _load_set(stem)["coefficients"]
    for i, want in enumerate(SHIPPED_BETAS, 1):
        name = f"beta_{i}"
        got = coeffs[name]["value"]
        assert got == want, f"{stem}:{name} = {got!r}, shipped {want!r}"
        # Guard a silent int/float slip (0 vs 0.0): the shipped decimals are all floats.
        assert type(got) is type(want), f"{stem}:{name} type {type(got)} != {type(want)}"


@pytest.mark.parametrize("stem", sorted(TP_GPUS))
def test_alpha_values_match_shipped_to_full_precision(stem):
    # The three per-request-overhead alphas, same exact-equality invariant as the betas.
    coeffs = _load_set(stem)["coefficients"]
    for i, want in enumerate(SHIPPED_ALPHAS, 1):
        name = f"alpha_{i}"
        got = coeffs[name]["value"]
        assert got == want, f"{stem}:{name} = {got!r}, shipped {want!r}"
        assert type(got) is type(want), f"{stem}:{name} type {type(got)} != {type(want)}"


@pytest.mark.parametrize("stem", sorted(TP_GPUS))
def test_zero_positions_declared_not_charged(stem):
    # Absent-vs-zero (INV-6 / Problem 4): the zero positions are PRESENT and priced at zero,
    # never omitted -- omission would select a different formula and change output.
    coeffs = _load_set(stem)["coefficients"]
    for name in ZERO_NOT_CHARGED:
        assert name in coeffs, f"{stem}: {name} must be declared, not absent"
        entry = coeffs[name]
        assert entry["value"] == 0.0, (stem, name, entry.get("value"))
        assert entry["method"] == "not_charged", (stem, name, entry.get("method"))
        assert entry["fitted"] is False, (stem, name, entry.get("fitted"))
        # A not_charged term must explain why it is priced at zero (co-occurrence for the
        # betas, uncalibrated-hop for alpha_hop); the schema requires a rationale, and we
        # assert it is genuinely non-empty here rather than merely present.
        assert isinstance(entry.get("rationale"), str) and entry["rationale"].strip(), (stem, name)


@pytest.mark.parametrize("stem", sorted(TP_GPUS))
def test_beta_ep_copied_from_beta_4(stem):
    # beta_11 (beta_EP) is seeded equal to beta_4, honestly recorded as copied rather than
    # presented as an independent measurement. Its value must equal beta_4's, to full
    # precision, so the copy is real and not a transcription typo.
    coeffs = _load_set(stem)["coefficients"]
    ep = coeffs["beta_11"]
    assert ep["method"] == "copied", (stem, ep.get("method"))
    assert ep["copied_from"] == "beta_4", (stem, ep.get("copied_from"))
    assert ep["fitted"] is False, (stem, ep.get("fitted"))
    assert ep["value"] == coeffs["beta_4"]["value"], (stem, ep["value"], coeffs["beta_4"]["value"])


@pytest.mark.parametrize("stem", sorted(TP_GPUS))
def test_live_entries_measured_and_fitted(stem):
    # The eight live betas and three alphas came from the iter29 regression: method measured,
    # fitted true. (The zero/copied entries are asserted separately above.)
    coeffs = _load_set(stem)["coefficients"]
    live_betas = [f"beta_{i}" for i in range(1, 12) if f"beta_{i}" not in ("beta_2", "beta_9", "beta_11")]
    live = live_betas + ["alpha_1", "alpha_2", "alpha_3"]
    for name in live:
        entry = coeffs[name]
        assert entry["method"] == "measured", (stem, name, entry.get("method"))
        assert entry["fitted"] is True, (stem, name, entry.get("fitted"))


@pytest.mark.parametrize("stem", sorted(TP_GPUS))
def test_extends_its_gpu_roofline_set_and_declares_backend(stem):
    # Each set feeds the trained-physics backend and inherits its GPU's MFU pair by extending
    # exactly the matching roofline set -- the mechanism that lets a global beta/alpha vector
    # carry a per-GPU MFU pair.
    doc = _load_set(stem)
    assert doc["backend"] == "trained-physics", (stem, doc.get("backend"))
    assert doc["extends"] == ROOFLINE_PARENT[stem], (stem, doc.get("extends"))


@pytest.mark.parametrize("stem", sorted(TP_GPUS))
def test_every_entry_scoped_to_its_gpu(stem):
    # Scope discipline: the global vector is transcribed identically per GPU but each entry
    # names its own hardware, so a set states the range it holds over. alpha_hop additionally
    # scopes nodes_spanned (it is a cross-node term), so we only assert hardware is present
    # and names this GPU rather than requiring an exact scope equality for every entry.
    hw = TP_GPUS[stem]
    coeffs = _load_set(stem)["coefficients"]
    for name, entry in coeffs.items():
        scope = entry["scope"]
        assert scope.get("hardware") == [hw], (stem, name, scope)
