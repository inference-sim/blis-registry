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

# A directory that EXISTS but is EMPTY must skip too, not run and fail.
#
# The tree is read from $AISIMULATE_DATA, which defaults under /tmp and so is reaped: it
# was found once with 1,427 directories and 0 files. `not DATA.is_dir()` passes for that
# hollow tree, so twenty re-derivation tests ran and failed with "no fit" / "no parts
# fitted" -- indistinguishable, at a glance, from a coefficient regression. Probing for
# one parquet tells the two apart, and the reason string says which state was found.
_PARQUETS = next(DATA.rglob("*.parquet"), None) if DATA.is_dir() else None
pytestmark = pytest.mark.skipif(
    _PARQUETS is None,
    reason=(f"AISimulate data not present at {DATA}" if not DATA.is_dir()
            else f"AISimulate tree at {DATA} holds no parquet: re-fetch with "
                 f"`git clone https://github.com/ai-dynamo/aisimulate`"),
)

# Each writer, and what it maintains. The timeout is generous because these shell out
# to fitters that read multi-hundred-megabyte parquet sweeps.
WRITERS = [
    ("relane_attention_decode.py", "full-attention decode floor and rate"),
    ("relane_attention_prefill.py", "prefill floor and work_scale"),
    ("relane_attention_swa.py", "sliding-window decode floor and rate"),
    ("relane_gemm_envelope.py", "the GEMM efficiency ramp"),
    ("relane_moe_imbalance.py", "the MoE routing-imbalance pair"),
    # Not a fitter -- it transcribes AISimulate's activation table and anchors the capture
    # bytes on a committed CSV -- but it renders a whole file, so byte-identity is the right
    # gate. The capture half needs no AISimulate tree and is also gated in CI by
    # test_coefficient_properties.py::test_capture_bytes_re_derive_from_the_committed_samples.
    ("emit_memory.py", "the activation multiples and CUDA-graph capture bytes"),
]

# relane_attention_mla.py is DELIBERATELY ABSENT from both lists, and this records why so
# the omission is not read as an oversight.
#
# It writes BOTH halves of the MLA pair from `mla_generation_module_perf.parquet`. The RATE
# half is sound and committed as fitted. The FLOOR half is not: that table is a MODULE
# measurement including the down-projections blis-catalog prices as separate GEMM nodes, so
# a module-derived floor charges them twice -- it cost 2.45 points of overall TPOT and 8.2
# on kimi-k2.5 end-to-end. `scripts/correct_mla_floor.py` owns the floor now and commits it
# as `method: assumed`, derived from the same part's measured attention-kernel floor.
#
# So the writer no longer reproduces the committed file, and listing it here would assert
# that it should. Its plain `--check` is expected to disagree about the floor.
#
# The two halves are gated separately instead, so neither drifts:
#   * the RATE by `relane_attention_mla.py --check-rate`, in RATE_ONLY below;
#   * the FLOOR by `correct_mla_floor.py --check`, plus the property test in
#     test_coefficient_properties.py::test_kind_specific_decode_floors_are_not_module_measurements.

# Writers where only PART of the pair is still committed from the fitter. The flag checks
# that part alone; the rest of the pair has its own gate, named in the comment above.
RATE_ONLY = [
    ("relane_attention_mla.py --check-rate", "the MLA decode rate"),
]

# Writers whose rewrite path renders a GENERIC rationale, over entries that carry
# hand-authored prose a fit cannot regenerate. For these, byte-identity is the wrong
# assertion -- it would demand that the committed file lose a finding. The h200 KDA
# entry records that a missing coefficient made a GLM-5.3-Flash prefill read 9.4x
# FASTER on h200 than h100 despite a shared die; that is worth more than the generic
# sentence a re-render would put in its place. So the VALUES are asserted instead,
# which is what a drift check is actually for.
VALUE_ONLY = [
    ("relane_recurrent_kda.py", "cost-model-recurrent.yaml", "the KDA chain pair"),
    # The mamba2 and GDN writers render a generic rationale too, and the committed
    # mamba2 h100 entry carries hand-authored prose about the missing selective-scan
    # kernel. Values are asserted; prose is not.
    ("relane_recurrent_family.py --family mamba2", "cost-model-recurrent.yaml",
     "the mamba2 convolution pair"),
    ("relane_recurrent_family.py --family gdn", "cost-model-recurrent.yaml",
     "the GDN chain pair"),
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


@pytest.mark.parametrize("script,target,what", VALUE_ONLY,
                         ids=[w[0] for w in VALUE_ONLY])
def test_writer_reproduces_the_committed_values(script, target, what, tmp_path):
    """Every VALUE must re-derive, even where the prose is deliberately hand-authored.

    Runs the writer against a scratch copy of the repo's coefficients directory, then
    compares (name, scope) -> value against the committed file. Prose is ignored by
    construction; a moved number fails.
    """
    import shutil

    import yaml

    def values(path):
        doc = yaml.safe_load(Path(path).read_text())
        out = {}
        for e in doc["coefficients"]:
            (n, b), = e.items()
            scope = tuple(sorted(
                (k, tuple(v) if isinstance(v, list) else v)
                for k, v in (b.get("scope") or {}).items()))
            out[(n, scope)] = b["value"]
        return out

    scratch = tmp_path / "repo"
    shutil.copytree(REPO / "coefficients", scratch / "coefficients")
    shutil.copytree(REPO / "scripts", scratch / "scripts")
    committed = values(REPO / "coefficients" / target)

    env = dict(os.environ, AISIMULATE_DATA=str(DATA))
    r = subprocess.run(
        # `script` may carry arguments (e.g. "relane_recurrent_family.py --family gdn"),
        # because one writer maintains two families and each needs its own gate.
        [sys.executable, str(scratch / "scripts" / script.split()[0]),
         *script.split()[1:]],
        capture_output=True, text=True, env=env, cwd=str(scratch), timeout=1800,
    )
    assert r.returncode == 0, f"{script} failed:\n{r.stdout}\n{r.stderr}"

    refit = values(scratch / "coefficients" / target)
    moved = {k: (committed[k], refit[k])
             for k in committed if k in refit and committed[k] != refit[k]}
    missing = [k for k in committed if k not in refit]
    assert not moved and not missing, (
        f"{script} re-derives different values for {what}: moved={moved} "
        f"missing={missing}. A value that its own fitter no longer produces is the "
        f"drift this file exists to catch."
    )


def test_every_fitted_family_has_a_writer():
    """A fitted coefficient family with no writer cannot be kept honest.

    Guards the gap this file closes from reopening: adding a new fitted family without
    a `--check`-capable writer should fail here rather than drift silently for months.
    """
    import yaml

    # Both gates count as owning a family: WRITERS asserts byte-identity, VALUE_ONLY
    # asserts value-identity where the prose is deliberately hand-authored. A family in
    # either is covered against drift.
    writers = ({w for w, _ in WRITERS}
               | {w.split()[0] for w, _, _ in VALUE_ONLY}
               # A split pair's fitted half is gated by its own flag; the writer still
               # owns that half, so it counts as covered here.
               | {w.split()[0] for w, _ in RATE_ONLY})
    # Family stem -> the writer that owns it.
    owned = {
        "attention_decode_floor": "relane_attention_decode.py",
        "attention_decode_rate": "relane_attention_decode.py",
        "attention_decode_floor_swa": "relane_attention_swa.py",
        "attention_decode_rate_swa": "relane_attention_swa.py",
        # The MLA floor is no longer fitted -- correct_mla_floor.py commits it as
        # `assumed` from the part's own attention floor -- so it never reaches this loop,
        # which only considers `fitted: true` entries. The RATE is still fitted and still
        # owned by the module-table writer.
        "attention_decode_rate_mla": "relane_attention_mla.py",
        "attention_prefill_floor": "relane_attention_prefill.py",
        "attention_prefill_work_scale": "relane_attention_prefill.py",
        "recurrent_decode_floor_kda": "relane_recurrent_kda.py",
        "recurrent_decode_rate_kda": "relane_recurrent_kda.py",
        "recurrent_decode_floor_gdn": "relane_recurrent_family.py",
        "recurrent_decode_rate_gdn": "relane_recurrent_family.py",
        "recurrent_decode_floor_mamba2": "relane_recurrent_family.py",
        "recurrent_decode_rate_mamba2": "relane_recurrent_family.py",
        "gemm_eps_max": "relane_gemm_envelope.py",
        "gemm_m_half": "relane_gemm_envelope.py",
        "moe_routing_imbalance_median": "relane_moe_imbalance.py",
        "moe_routing_imbalance_p90": "relane_moe_imbalance.py",
    }
    # Families whose absence of a writer is recorded rather than accidental.
    exempt = {
        # Collectives are covered by test_generated_from_aisimulate.py, which
        # re-derives floors, rates and the vLLM all-reduce triple directly.
        #
        # The mamba2 pair used to be exempt here on the grounds that it had one part and
        # no writer. It now has six parts and relane_recurrent_family.py, so the
        # exemption is gone rather than carried forward -- an exemption that outlives its
        # reason is how a gate quietly stops gating.
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


def test_the_mla_floor_correction_still_holds():
    """`correct_mla_floor.py --check`: each MLA floor equals its part's attention floor.

    The floor half of the MLA pair is not fitted, so neither re-derivation gate above
    covers it. This is its drift check, and it needs no AISimulate tree: the value is
    derived from a sibling coefficient in the same committed file, so the relation is
    checkable from the repository alone.
    """
    r = subprocess.run(
        [sys.executable, str(REPO / "scripts" / "correct_mla_floor.py"), "--check"],
        capture_output=True, text=True, cwd=str(REPO), timeout=300,
    )
    assert r.returncode == 0, (
        "correct_mla_floor.py --check failed, so a committed attention_decode_floor_mla "
        "no longer equals its part's measured attention_decode_floor. Either a part-wide "
        "floor was refitted without re-running the correction, or a module-derived floor "
        "was reintroduced.\n"
        f"--- stdout ---\n{r.stdout}\n--- stderr ---\n{r.stderr}"
    )


@pytest.mark.parametrize("script,what", RATE_ONLY, ids=[w[0] for w in RATE_ONLY])
def test_writer_reproduces_the_committed_half_it_still_owns(script, what):
    """For a split pair, the half still fitted must match its fitter.

    The other half has its own gate; see the RATE_ONLY comment for which.
    """
    env = dict(os.environ, AISIMULATE_DATA=str(DATA))
    r = subprocess.run(
        [sys.executable, str(REPO / "scripts" / script.split()[0])] + script.split()[1:],
        capture_output=True, text=True, env=env, cwd=str(REPO), timeout=1800,
    )
    assert r.returncode == 0, (
        f"{script} failed, so the committed {what} is not what its fitter produces.\n"
        f"--- stdout ---\n{r.stdout}\n--- stderr ---\n{r.stderr}"
    )
