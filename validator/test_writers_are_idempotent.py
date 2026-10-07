"""Every family with a writer must re-derive to the committed file, byte for byte.

WHY THIS FILE EXISTS. `test_generated_from_aisimulate.py` re-derives the GEMM and
collective families by a second path. Four families had no such gate -- decode
attention, prefill attention, sliding-window attention and KDA -- and the consequences
were real rather than hypothetical:

  * The prefill pair drifted on all seven parts. `work_scale` is a fraction of the bf16
    GEMM ramp, commit 5be809b relaned that ramp, and the prefill numerators were never
    refitted against the new denominator. Nothing could notice.
  * `docs/reproducing-coefficients.md` carried a KDA recipe (`kda/sglang/0.5.16`, the
    `kda_fused_decode` row) that regenerates SUPERSEDED values -- a rate 6.4x too fast,
    from an AMD-only kernel that runs on no CUDA deployment.

Each writer supports `--check`: it re-runs its own fitter and exits non-zero if the
committed file would change. That is exactly the gate these families were missing, so
this test is a thin wrapper over it. A refit that moves a value fails here rather than
shipping silently.

Skipped when the AISimulate tree is absent, since it is not vendored.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

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

# Each writer, and what it maintains. The timeout is generous because these shell out
# to fitters that read multi-hundred-megabyte parquet sweeps.
WRITERS = [
    ("relane_attention_decode.py", "full-attention decode floor and rate"),
    ("relane_attention_prefill.py", "prefill floor and work_scale"),
    ("relane_attention_swa.py", "sliding-window decode floor and rate"),
    ("relane_recurrent_kda.py", "the KDA chain pair"),
    ("relane_gemm_envelope.py", "the GEMM efficiency ramp"),
    ("relane_moe_imbalance.py", "the MoE routing-imbalance pair"),
]


@pytest.mark.parametrize("script,what", WRITERS, ids=[w[0] for w in WRITERS])
def test_writer_reproduces_the_committed_file(script, what):
    """`--check` must pass: the committed values are what the fitter produces."""
    env = dict(os.environ, AISIMULATE_DATA=str(DATA))
    r = subprocess.run(
        [sys.executable, str(REPO / "scripts" / script), "--check"],
        capture_output=True, text=True, env=env, cwd=str(REPO), timeout=1800,
    )
    assert r.returncode == 0, (
        f"{script} --check failed, so the committed {what} is not what its own fitter "
        f"produces. Either a fit moved and the file was not updated, or the file was "
        f"edited without re-running the writer.\n"
        f"--- stdout ---\n{r.stdout}\n--- stderr ---\n{r.stderr}"
    )


def test_every_fitted_family_has_a_writer():
    """A fitted coefficient family with no writer cannot be kept honest.

    Guards the gap this file closes from reopening: adding a new fitted family without
    a `--check`-capable writer should fail here rather than drift silently for months.
    """
    import yaml

    writers = {w for w, _ in WRITERS}
    # Family stem -> the writer that owns it.
    owned = {
        "attention_decode_floor": "relane_attention_decode.py",
        "attention_decode_rate": "relane_attention_decode.py",
        "attention_decode_floor_swa": "relane_attention_swa.py",
        "attention_decode_rate_swa": "relane_attention_swa.py",
        "attention_prefill_floor": "relane_attention_prefill.py",
        "attention_prefill_work_scale": "relane_attention_prefill.py",
        "recurrent_decode_floor_kda": "relane_recurrent_kda.py",
        "recurrent_decode_rate_kda": "relane_recurrent_kda.py",
        "gemm_eps_max": "relane_gemm_envelope.py",
        "gemm_m_half": "relane_gemm_envelope.py",
        "moe_routing_imbalance_median": "relane_moe_imbalance.py",
        "moe_routing_imbalance_p90": "relane_moe_imbalance.py",
    }
    # Families whose absence of a writer is recorded rather than accidental.
    exempt = {
        # The mamba2 pair is a documented LOWER BOUND from a single TRT-LLM sweep; it
        # has a fitter (fit_recurrent.py --model) and one part, and its value cannot
        # move without the sweep changing.
        "recurrent_decode_floor_mamba2",
        "recurrent_decode_rate_mamba2",
        # Collectives are covered by test_generated_from_aisimulate.py, which
        # re-derives floors, rates and the vLLM all-reduce triple directly.
    }

    unowned: dict[str, int] = {}
    for path in sorted((REPO / "coefficients").glob("*.yaml")):
        doc = yaml.safe_load(path.read_text())
        for e in doc["coefficients"]:
            (name, body), = e.items()
            if not body.get("fitted"):
                continue
            if name.startswith("collective_"):
                continue
            stem = name
            for suffix in ("_bf16", "_fp8_block", "_fp8", "_nvfp4"):
                if stem.endswith(suffix):
                    stem = stem[: -len(suffix)]
                    break
            if stem in exempt or stem in owned:
                continue
            unowned[stem] = unowned.get(stem, 0) + 1

    assert not unowned, (
        f"fitted families with no writer and no recorded exemption: {unowned}. "
        f"A fitted value that no script regenerates drifts silently -- that is how the "
        f"prefill pair went stale on seven parts. Add a writer with --check, or record "
        f"the exemption in this test with the reason."
    )
    for stem, w in owned.items():
        assert w in writers, f"{stem} claims writer {w}, which is not in WRITERS"
