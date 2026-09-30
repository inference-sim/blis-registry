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
PARTS = [
    ("h200_sxm", "h200", "trtllm/1.3.0rc20", "2.29.2"),
    ("h100_sxm", "h100", "trtllm/1.3.0rc20", "2.29.2"),
    ("l40s", "l40s", "trtllm/1.3.0rc20", "2.27.3"),
    ("gb200", "gb200-nvl72", "trtllm/1.3.0rc20", "2.29.2"),
    ("a100_sxm", "a100-sxm", "sglang/0.5.10", "2.27.3"),
]

DTYPE = {
    "bfloat16": "bf16", "float16": "fp16", "fp8": "fp8",
    "fp8_block": "fp8_block", "nvfp4": "nvfp4",
}
NCCL_DTYPE = {"half": "fp16", "int8": "int8"}


def registry_values() -> dict[tuple[str, tuple[str, ...]], float]:
    out = {}
    for name in ("cost-model-primitives", "cost-model-collectives"):
        doc = yaml.safe_load((REPO / "coefficients" / f"{name}.yaml").read_text())
        for item in doc["coefficients"]:
            (key, body), = item.items()
            out[(key, tuple(body["scope"]["hardware"]))] = body["value"]
    return out


def run_script(script: str, *args: str) -> str:
    result = subprocess.run(
        [sys.executable, str(REPO / "scripts" / script), *args],
        capture_output=True, text=True, check=True,
    )
    return result.stdout


@pytest.mark.parametrize("sku,chip,collection,nccl", PARTS)
def test_gemm_envelope_survives_independent_refit(sku, chip, collection, nccl):
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
    assert checked >= 12, f"{chip}: re-derivation produced too little to compare"


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
