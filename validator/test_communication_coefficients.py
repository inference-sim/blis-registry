"""Transcription test for the communication-coefficients set (R2G4).

R2G4 is *value-preserving*: naming/splitting the communication coefficients must not change a
single number, and step time must stay byte-identical. This test is the frozen-snapshot gate
that proves it.

Two invariants are pinned, both against a PINNED SOURCE SNAPSHOT embedded in this file (not
against the live other-repo source, and not against the running simulator):

  * the three all-reduce/dispatch entries that split (or rename) the single β₄ slot all hold
    β₄'s value ``0.752037`` — a value-preserving split of one array position into three names;
    and
  * the seven ``moe_comm_scale_<backend>`` entries hold ``1.0`` and their backend names match
    this file's snapshot of the vLLM backend set.

``BETA4``/``MOE_COMM_SCALE_BACKENDS`` below are a frozen snapshot of the source as it shipped
when this set was authored: β₄ = ``beta_coeffs[3]`` in inference-sim ``defaults.yaml`` (also the
value R2G3 transcribed as ``tp_allreduce_attention``), and the seven backends of
``moeCommBackends`` in ``sim/latency/moe_comm_backend.go`` (all sharing ``commScale = 1.0``).
The registry cannot reach that repo at test time, so the snapshot is embedded here as literals —
that embedding *is* the frozen snapshot. These tests assert exact value/type equality between the
committed YAML and this snapshot; they do NOT detect a later change to ``moeCommBackends`` in the
other repository, and they do NOT exercise the simulator, so they do not by themselves prove
bit-for-bit runtime step-time parity — that runtime byte-identity gate is the later N-track's
(R2 tracker). What ``1.0`` guarantees here is only that it is an exact multiplicative identity in
the data.

``tp_allreduce_attention`` and ``cross_node_hop_latency`` are NOT in this set — they already
carry their final R2G4 names in ``coefficients/trained-physics.yaml`` and are unchanged by R2G4.
"""

from __future__ import annotations

from pathlib import Path

from validator.loader import load_strict
from validator.schema import check_set

REPO_ROOT = Path(__file__).resolve().parent.parent
SET_PATH = REPO_ROOT / "coefficients" / "communication-coefficients.yaml"

# β₄ = beta_coeffs[3] (defaults.yaml), the one fitted value the split shares.
BETA4 = 0.752037

# The three entries that split/rename the single β₄ slot; each copied_from the one fitted member.
SPLIT_FROM_BETA4 = {
    "tp_allreduce_dense_ffn",   # β₄ split: dense-FFN all-reduce
    "tp_allreduce_moe_ffn",     # β₄ split: MoE-FFN all-reduce
    "moe_dispatch",             # β_EP rename (was moe_dispatch_alltoall), seeded to β₄
}

# Frozen snapshot of the seven vLLM all-to-all backends (moeCommBackends, in source order), all
# sharing commScale = 1.0. A distinct moe_comm_scale_<backend> entry per name (the validator
# rejects duplicate coefficient names within a set).
MOE_COMM_SCALE_BACKENDS = [
    "naive",
    "allgather_reducescatter",
    "pplx",
    "deepep_high_throughput",
    "deepep_low_latency",
    "mori",
    "flashinfer_all2allv",
]
MOE_COMM_SCALE_NAMES = {f"moe_comm_scale_{b}" for b in MOE_COMM_SCALE_BACKENDS}


def _coeffs_by_name() -> dict:
    data = load_strict(SET_PATH.read_text(encoding="utf-8"))
    by_name: dict = {}
    for item in data["coefficients"]:
        (name, entry), = item.items()
        by_name[name] = entry
    return by_name


def test_set_validates_as_standalone():
    data = load_strict(SET_PATH.read_text(encoding="utf-8"))
    assert check_set(data) == []
    assert data["name"] == "communication-coefficients"
    assert "extends" not in data and "backend" not in data


def test_names_are_exactly_the_added_and_renamed_set():
    # Exactly the three split/renamed entries plus the seven per-backend dials — no more, no
    # less. tp_allreduce_attention / cross_node_hop_latency are NOT here (they stay in
    # trained-physics with their final names); a leaked restatement fails by name.
    assert set(_coeffs_by_name()) == SPLIT_FROM_BETA4 | MOE_COMM_SCALE_NAMES


def test_split_and_rename_values_and_provenance():
    # The core R2 invariant for the split: each of the three holds β₄'s value to full precision
    # and type, is method: copied from the one fitted member, and is not itself fitted.
    coeffs = _coeffs_by_name()
    for name in SPLIT_FROM_BETA4:
        entry = coeffs[name]
        assert entry["value"] == BETA4, f"{name} = {entry['value']!r}, β₄ {BETA4!r}"
        assert type(entry["value"]) is float, (name, type(entry["value"]))
        assert entry["method"] == "copied", (name, entry["method"])
        assert entry["copied_from"] == "tp_allreduce_attention", (name, entry.get("copied_from"))
        assert entry["fitted"] is False, (name, entry["fitted"])


def test_moe_dispatch_supersedes_its_r2g3_name():
    # The one genuine rename records its lineage machine-readably (issue #5): moe_dispatch is the
    # final name for R2G3's moe_dispatch_alltoall. The two new split siblings supersede nothing.
    coeffs = _coeffs_by_name()
    assert coeffs["moe_dispatch"]["supersedes"] == "moe_dispatch_alltoall"
    for name in ("tp_allreduce_dense_ffn", "tp_allreduce_moe_ffn"):
        assert "supersedes" not in coeffs[name], (name, "new name supersedes nothing")


def test_moe_comm_scale_backends_match_source():
    # The seven per-backend dials match this file's PINNED SNAPSHOT of the vLLM backend set (a
    # missing or extra backend vs the snapshot fails here); each an assumed, unfitted 1.0. This
    # does not detect a later change to moeCommBackends in the other repo — see the module
    # docstring.
    coeffs = _coeffs_by_name()
    present = {n for n in coeffs if n.startswith("moe_comm_scale_")}
    assert present == MOE_COMM_SCALE_NAMES, (present, MOE_COMM_SCALE_NAMES)
    for name in MOE_COMM_SCALE_NAMES:
        entry = coeffs[name]
        assert entry["value"] == 1.0, (name, entry["value"])
        assert type(entry["value"]) is float, (name, type(entry["value"]))
        assert entry["method"] == "assumed", (name, entry["method"])
        assert entry["fitted"] is False, (name, entry["fitted"])


def test_values_are_preserved():
    # R2G4 is value-preserving: every split/rename holds β₄'s value and every dial holds the 1.0
    # multiplicative identity. This asserts the DATA is unchanged; runtime step-time parity is the
    # N-track's gate (see the module docstring), not proven here.
    coeffs = _coeffs_by_name()
    for name in SPLIT_FROM_BETA4:
        assert coeffs[name]["value"] == BETA4
    for name in MOE_COMM_SCALE_NAMES:
        assert coeffs[name]["value"] == 1.0


def test_scope_is_h100_throughout():
    # Consistent with trained-physics: H100 is the only calibrated target. Nothing new is
    # in-repo fitted, so fitted is false on every entry (checked per-family above).
    coeffs = _coeffs_by_name()
    for name, entry in coeffs.items():
        assert entry["scope"] == {"hardware": ["H100"]}, (name, entry["scope"])
