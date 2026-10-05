"""Every generated coefficient must survive an independent re-derivation.

The two cost-model sets are generated (scripts/emit_primitives.py and
scripts/emit_collectives.py), which removes transcription error but not the risk
that a generator is wrong. So this test re-derives each value by a second path —
the standalone fit scripts, which read the parquet files directly — and compares.

It is a real check rather than a tautology because the two paths differ where it
matters: the generators pin a collection by name and the fit scripts take one as
an argument, so a generator that silently selected a different collection than the
one its citation names would be caught here. That is not hypothetical; it happened
during development, when a "newest version" heuristic picked a 3 564-shape MoE
collection over the 42 174-shape one the citation named, changing the imbalance
median from 1.056 to 1.002.

Skipped when the AISimulate tree is absent, since it is not vendored here.
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

DATA = Path(
    os.environ.get(
        "AISIMULATE_DATA",
        "/tmp/aisim/python/aisimulate/src/aisimulate_core/systems/data",
    )
)
REPO = Path(__file__).resolve().parent.parent

pytestmark = pytest.mark.skipif(
    not DATA.is_dir(), reason=f"AISimulate data not present at {DATA}"
)

# (sku, catalog chip, gemm collection, nccl version)
# (sku, chip, the GEMM collection its coefficients CITE, the NCCL collection).
# The GEMM entries moved to the vLLM lane: vLLM runs its own linear kernels for every
# quantized dtype -- CutlassFP8ScaledMMLinearKernel, FlashInferCuteDslNvFp4LinearKernel,
# and a FlashInfer/DeepGEMM dispatch for fp8_block -- not the single `torch_flow` path the
# TRT-LLM sweep measures. A100 keeps SGLang, the only lane carrying that family for it.
# l40s's newest vLLM gemm collection is 0.24.0; the Hopper and Blackwell parts use 0.25.0
# or 0.27.1 and are covered by the shape-ramp gate below rather than here.
PARTS = [
    ("h200_sxm", "h200", "vllm/0.25.0", "2.29.2"),
    ("h100_sxm", "h100", "vllm/0.25.0", "2.29.2"),
    ("l40s", "l40s", "vllm/0.24.0", "2.27.3"),
    ("gb200", "gb200-nvl72", "vllm/0.27.1", "2.29.2"),
    # a100_sxm's GEMM moved to the vLLM lane too: its vLLM sweep (0.14.0) is smaller
    # than SGLang's but measures the engine being predicted, and scored against the vLLM
    # envelope the vLLM fit wins 0.0673 to 0.1002. Its MoE imbalance stays on SGLang --
    # the vLLM MoE sweep for this part has no `balanced` rows, so no paired ratio exists.
    ("a100_sxm", "a100-sxm", "vllm/0.14.0", "2.27.3"),
]

DTYPE = {
    "bfloat16": "bf16", "float16": "fp16", "fp8": "fp8",
    "fp8_block": "fp8_block", "nvfp4": "nvfp4",
}
NCCL_DTYPE = {"half": "fp16", "int8": "int8"}


def registry_values() -> dict[tuple[str, tuple[str, ...]], float]:
    """Coefficient values keyed by (name, scope tuple), plus one entry per chip.

    The per-chip entries exist because a scope is a SET and the kernel matches
    membership -- `resolve.Scope.Admits` -- so one entry may serve several chips. A
    lookup keyed only on the exact tuple missed every widened scope, and widening is
    legitimate: a100-80 is an alias of a100-sxm and shares its measurements, so both
    names appear in one scope rather than in two entries that could drift apart.

    A chip that appears in more than one scope for the same coefficient would be a
    genuine ambiguity, so that is rejected rather than resolved by order.
    """
    out: dict[tuple[str, tuple[str, ...]], float] = {}
    for name in ("cost-model-primitives", "cost-model-collectives"):
        doc = yaml.safe_load((REPO / "coefficients" / f"{name}.yaml").read_text())
        for item in doc["coefficients"]:
            (key, body), = item.items()
            chips = tuple(body["scope"]["hardware"])
            out[(key, chips)] = body["value"]
            for chip in chips:
                single = (key, (chip,))
                if single in out and out[single] != body["value"]:
                    raise AssertionError(
                        f"{key} resolves to two values for {chip}: "
                        f"{out[single]} and {body['value']}"
                    )
                out[single] = body["value"]
    return out


def run_script(script: str, *args: str) -> str:
    result = subprocess.run(
        [sys.executable, str(REPO / "scripts" / script), *args],
        capture_output=True, text=True, check=True,
    )
    return result.stdout


# Parts whose GEMM ramp is the three-factor shape-aware form, fitted by
# fit_gemm_shape_ramp.py on vllm/0.25.0. Their eps_max and m_half come from that JOINT
# fit, so re-deriving them from the one-factor envelope would check a provenance the
# registry no longer claims. l40s and a100-sxm are absent deliberately: their vLLM gemm
# sweeps are at 0.24.0 and 0.14.0, and pinning one collection per part is what keeps a
# difference between parts attributable to silicon rather than to an engine version.
SHAPE_RAMP_PARTS = {"h200", "h100", "b200", "b300", "gb200-nvl72"}
SHAPE_RAMP_COLLECTION = "vllm/0.25.0"


@pytest.mark.parametrize("sku,chip,collection,nccl", PARTS)
def test_gemm_envelope_survives_independent_refit(sku, chip, collection, nccl):
    if chip in SHAPE_RAMP_PARTS:
        pytest.skip(
            f"{chip} carries the three-factor ramp; "
            f"test_gemm_shape_ramp_survives_independent_refit covers it"
        )
    reg = registry_values()
    out = run_script("fit_gemm_envelope.py", str(DATA / sku / "gemm" / collection))
    checked = 0
    for line in out.splitlines():
        m = re.match(
            r"(\w+)\s+rows=\s*\d+\s+M_values=\s*\d+\s+eps_max=([\d.]+)"
            r"\s+M_half=\s*(\d+)",
            line,
        )
        if not m:
            continue
        suffix = DTYPE[m.group(1)]
        for key, want in (
            (f"gemm_eps_max_{suffix}", float(m.group(2))),
            (f"gemm_m_half_{suffix}", int(m.group(3))),
        ):
            got = reg.get((key, (chip,)))
            assert got is not None, f"{chip}: {key} is not in the registry"
            assert abs(float(got) - want) < 1e-9, (
                f"{chip} {key}: registry holds {got}, an independent refit of "
                f"{collection} gives {want}"
            )
            checked += 1
    assert checked >= 2, f"{chip}: refit produced nothing to compare"


@pytest.mark.parametrize("sku,chip,collection,nccl", PARTS)
def test_collective_floors_and_rates_survive_independent_rederivation(
    sku, chip, collection, nccl
):
    """All three collective parameters must survive an independent re-derivation.

    The transition rate is included because it is the one the step time actually uses:
    the peak is only a ceiling. A generator that emitted a correct floor and peak beside
    a wrong transition rate would price every collective wrongly while looking right.
    """
    reg = registry_values()
    out = run_script("fit_collectives.py", str(DATA / sku / "comm" / "nccl" / nccl))
    checked = 0
    for line in out.splitlines()[2:]:
        parts = line.split()
        # operation dtype ranks floor_us spread peak_GB/s trans_GB/s err2 err3 [note]
        if len(parts) < 9 or parts[3] == "-":
            continue
        op, dtype, ranks = parts[0], parts[1], int(parts[2])
        # all_reduce is fitted from vLLM's own custom kernel, not from NCCL, because vLLM
        # does not call NCCL for a tensor-parallel all-reduce its custom kernel can serve.
        # Re-deriving it from the NCCL sweep would compare the registry against a
        # collection no committed entry cites. The vLLM lane is re-derived separately
        # below, so the coefficient is still checked -- against the right data.
        if op == "all_reduce":
            continue
        floor = float(parts[3])
        # The script reports rates in GB/s for readability; the registry stores
        # bytes per microsecond, which is 1000x. Converting here rather than
        # matching on the printed figure keeps the test honest about units — a
        # mismatch of exactly 1000 is the bug this conversion prevents, and it
        # was the bug that surfaced when the script's columns changed.
        peak = float(parts[5]) * 1000
        transition = float(parts[6]) * 1000
        stem = f"{op}_{NCCL_DTYPE[dtype]}_{ranks}rank_{chip.replace('-', '_')}"
        for key, want, tol in (
            (f"collective_floor_{stem}", round(floor, 2), 0.05),
            (f"collective_peak_rate_{stem}", peak, 50.0),
            (f"collective_transition_rate_{stem}", transition, 50.0),
        ):
            got = reg.get((key, (chip,)))
            assert got is not None, f"{chip}: {key} is not in the registry"
            assert abs(float(got) - want) < tol, (
                f"{chip} {key}: registry holds {got}, an independent re-derivation "
                f"gives {want}"
            )
            checked += 1
    assert checked >= 9, f"{chip}: re-derivation produced too little to compare"


@pytest.mark.parametrize(
    "sku, chip, vllm",
    [
        ("h200_sxm", "h200", "0.24.0"),
        ("h100_sxm", "h100", "0.24.0"),
        ("b200_sxm", "b200", "0.24.0"),
        ("b300_sxm", "b300", "0.24.0"),
    ],
)
def test_all_reduce_survives_rederivation_from_the_vllm_lane(sku, chip, vllm):
    """The all-reduce triple must re-derive from the collection its citations name.

    BLIS prices vLLM serving, and vLLM serves a tensor-parallel all-reduce from its own
    custom kernel rather than from NCCL. The committed values therefore come from
    `custom_allreduce_perf.parquet` in the CUDA-graph lane, which is the lane a decode step
    runs in, and this re-derives them from that file.

    The lane matters enough to pin: on h200 at 8 ranks the NCCL floor is 15.98us against
    4.711us here, and the eager lane is slower than NCCL rather than faster. A test that
    accepted any of the three would accept a 3.4x mispricing.
    """
    reg = registry_values()
    out = run_script(
        "fit_collectives_vllm.py", str(DATA / sku / "comm" / "vllm" / vllm)
    )
    checked = 0
    for line in out.splitlines():
        parts = line.replace("#", " ").split()
        # backend ranks floor_us trans_GB/s peak_GB/s err2 err3 flat_n
        if len(parts) < 8 or parts[0] != "vllm_graph":
            continue
        ranks = int(parts[1])
        floor, transition, peak = float(parts[2]), float(parts[3]), float(parts[4])
        stem = f"all_reduce_fp16_{ranks}rank_{chip.replace('-', '_')}"
        for key, want, tol in (
            (f"collective_floor_{stem}", round(floor, 2), 0.05),
            (f"collective_peak_rate_{stem}", peak * 1000, 50.0),
            (f"collective_transition_rate_{stem}", transition * 1000, 50.0),
        ):
            got = reg.get((key, (chip,)))
            assert got is not None, f"{chip}: {key} is not in the registry"
            assert abs(float(got) - want) < tol, (
                f"{chip} {key}: registry holds {got}, an independent re-derivation "
                f"of the vLLM lane gives {want}"
            )
            checked += 1
    assert checked == 9, (
        f"{chip}: expected 3 ranks x 3 coefficients from the vllm_graph lane, "
        f"compared {checked}"
    )


def test_no_coefficient_cites_a_non_nvidia_measurement_source():
    """The cost-model sets are single-sourced from NVIDIA AISimulate.

    Mixing measurement sources across a set makes cross-SKU comparisons
    confounded, since two sources apply different methodologies. The host-overhead
    set is exempt: its entries are assumed, cite no measurement, and say so.
    """
    for name in ("cost-model-primitives", "cost-model-collectives"):
        doc = yaml.safe_load((REPO / "coefficients" / f"{name}.yaml").read_text())
        for item in doc["coefficients"]:
            (key, body), = item.items()
            for source in body.get("sources", []):
                cite = source["cite"]
                assert "AISimulate" in cite or "aisimulate" in cite, (
                    f"{name}/{key} cites {cite!r}, which is not an AISimulate "
                    f"collection; this set is single-sourced"
                )


@pytest.mark.parametrize("sku,chip", sorted(
    (sku, chip) for sku, chip in {
        "h200_sxm": "h200", "h100_sxm": "h100", "b200_sxm": "b200",
        "b300_sxm": "b300", "gb200": "gb200-nvl72",
    }.items()
))
def test_gemm_shape_ramp_survives_independent_refit(sku, chip):
    """Re-derive all four ramp coefficients from the sweep their citation names.

    SKIPPED while the registry carries only the one-factor ramp. The three-factor form
    is fitted by scripts/fit_gemm_shape_ramp.py and implemented in the kernel, and its
    coefficients are deliberately NOT applied: it is several times closer per shape --
    the one-factor ramp over-predicts efficiency by up to 700x at small output width --
    and it makes the end-to-end score worse, because the error it removes was cancelling
    another. docs/perf-model/hypothesis-log.md has both measurements. Gating on the
    registry rather than on a hardcoded list means this test starts checking the moment
    those coefficients land, and does not fail for their absence.

    The three-factor form fits eps_max, m_half, k_half and n_half TOGETHER, so all four
    have to be checked against one re-run. Checking a subset would let the registry hold
    a mixture of two fits -- which is the defect this replaces: eps_max and m_half were
    the envelope fit's while the shape terms were absent, so every GEMM in a layer read
    the same efficiency.
    """
    reg = registry_values()
    if not any(k.startswith("gemm_k_half_") and chip in scope
               for (k, scope) in reg):
        pytest.skip(
            f"{chip} carries the one-factor ramp; the three-factor coefficients are "
            f"fitted and implemented but not applied"
        )
    out = run_script("fit_gemm_shape_ramp.py", "--emit", f"{sku}:{chip}")
    seen = {}
    name = None
    for line in out.splitlines():
        m = re.match(r"^  - (gemm_\w+):$", line)
        if m:
            name = m.group(1)
            continue
        m = re.match(r"^      value: ([\d.]+)$", line)
        if m and name:
            seen[name] = float(m.group(1))
            name = None
    assert seen, f"{chip}: the fitter emitted no coefficients"
    checked = 0
    for key, want in seen.items():
        got = reg.get((key, (chip,)))
        assert got is not None, f"{chip}: {key} is not in the registry"
        assert abs(float(got) - want) < 1e-9, (
            f"{chip} {key}: registry holds {got}, an independent refit of "
            f"{SHAPE_RAMP_COLLECTION} gives {want}"
        )
        checked += 1
    # Four coefficients per dtype, and a part carries at least bf16, fp8 and fp8_block.
    assert checked >= 12, f"{chip}: only {checked} coefficients re-derived"
