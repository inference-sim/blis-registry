#!/usr/bin/env python3
"""Fit the GEMM ramp of vLLM's Marlin weight-only grouped GEMM (MXFP4 W4A16 experts).

    python scripts/fit_marlin_grouped_ramp.py            # print the fits, rewrite the file
    python scripts/fit_marlin_grouped_ramp.py --check    # exit 1 if the file would change

WHY THIS FAMILY EXISTS (inference-sim/blis-registry#38). On a part without a native FP4
tensor-core path, vLLM v0.31.0 runs MXFP4 routed experts through Marlin: the MXFP4
backend order is FlashInfer-TRTLLM, DeepGEMM, Marlin, and the first two require SM100
(`model_executor/layers/fused_moe/oracle/mxfp4.py`, `_get_priority_backends`). For a
compressed-tensors MXFP4 checkpoint (Kimi-K3) `CompressedTensorsW4A4Mxfp4MoEMethod` picks
`MarlinExperts` on EVERY SM unless `moe_backend == "b12x"`
(`quantization/compressed_tensors/compressed_tensors_moe/compressed_tensors_moe_w4a4_mxfp4.py`).
Marlin is WEIGHT-ONLY: the four-bit weights are dequantized in registers and multiplied
against BF16 activations with `mma.sync` (`csrc/libtorch_stable/quantization/marlin/
marlin_mma.h`), so its compute rate is a fraction of the part's BF16 peak, not of an FP8 or
FP4 one. No committed ramp describes it: bf16, fp8, fp8_block and nvfp4 are all dense
tensor-core paths.

THE QUANTITY. `eff(M) = eps_max * M / (M + M_half)`, as a fraction of the catalog's dense
BF16 peak (`TFlopsPeak`), CONDITIONED ON THE COMPOSITION blis-latency-kernel applies to a
routed-expert term (`kernel.go`, the routed SM and HBM halves are SUMMED):

    t = T*top_k*6*h*i / (BF16_peak * eff(T))  +  touched(T) * 3*h*i*B / (BwPeak * hbm_derate)
    touched(T) = E * (1 - ((E - top_k)/E)^T)          # price.ExpertsTouched

with M = T, the step's token count, as the kernel evaluates every ramp. B is the STORED
bytes per weight: 0.5 for the e2m1 values plus one E8M0 scale per 32, 0.53125. So the ramp
is the SM half that remains once the kernel's own memory half is charged, which is the
only way a fitted ramp composes back to the measured latency inside the kernel.

THE DATA. AISimulate `moe/vllm/0.25.0/moe_perf.parquet`, kernel_source
`vllm_compressedtensorsw4a4mxfp4moe_marlin_marlinexperts_situ_as_silu`, moe_dtype
`w4a16_mxfp4`: Kimi-K3's expert geometry (hidden 3584, inter 3072, top-16 of 896), the
`MarlinExperts` selection forced on the stock build with SiTU's elementwise epilogue
approximated as SiLU (AISimulate `collector/vllm/collect_moe.py`). The epilogue is an
elementwise pass between the two grouped GEMMs, so the approximation leaves the GEMMs the
ramp describes untouched. Fitted on balanced routing at moe_tp = moe_ep = 1, 27 token
counts from 1 to 16,384.

WHY vLLM 0.25.0 MEASURES THE v0.31.0 KERNEL. Between the two tags the Marlin MoE template
(`csrc/libtorch_stable/moe/marlin_moe_wna16/marlin_template.h`) changes only by removing
the runtime activation-ordering path (#54809): `has_act_order` was `group_blocks == 0`, a
compile-time constant, and MXFP4's group-32 instantiation never took that branch.

HELD OUT, AND WHAT IT SHOWS.
  * A DIFFERENT GEOMETRY AND WEIGHT TYPE. The same sweep carries vLLM's int4 Marlin MoE
    (`compressedtensorswna16marlinmoe_marlin_marlinexperts`, hidden 7168, inter 2048,
    top-8 of 384 -- the Kimi-K2 geometry, whose int4 checkpoints use group-32 BF16 scales,
    AISimulate model_configs/moonshotai--Kimi-K2.5_config.json). Same Marlin template,
    different expert shape: held-out error is reported beside the training error, and is
    the evidence that the asymptote is the kernel's rather than one shape's.
  * EXPERT AND TENSOR PARALLELISM ARE A FORM LIMIT, NOT THIS FIT'S. The error grows with
    moe_ep and moe_tp, because the kernel's routed term evaluates the ramp at the STEP's
    token count and carries no per-launch floor. The committed BF16 ramp shows the same
    growth on vLLM's BF16 Triton experts (2.8x at ep 8, 8.3x at ep 32 on h200, hidden 2048,
    inter 512, top-8 of 256), so it is a
    property of the composition and is reported, not tuned away.

LANE. vLLM, the engine BLIS prices. Units: `latency` is MILLISECONDS in every AISimulate
parquet.
"""

from __future__ import annotations

import argparse
import math
import os
import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import yaml

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import coefficient_blocks as cb  # noqa: E402

SET_PATH = HERE.parent / "coefficients" / "cost-model-primitives.yaml"
DEFAULT_DATA = os.environ.get(
    "AISIMULATE_DATA",
    "/tmp/aisim/python/aisimulate/src/aisimulate_core/systems/data")
# The vendored chip descriptors (validator/testdata/catalog), so the writer runs in CI;
# BLIS_CATALOG overrides it with a live checkout.
DEFAULT_CATALOG = (os.environ.get("BLIS_CATALOG")
                   or str(HERE.parent / "validator" / "testdata" / "catalog"))

COLLECTION = "vllm/0.25.0"
FIT_KERNEL = "vllm_compressedtensorsw4a4mxfp4moe_marlin_marlinexperts_situ_as_silu"
FIT_DTYPE = "w4a16_mxfp4"
HOLDOUT_KERNEL = "vllm_compressedtensorswna16marlinmoe_marlin_marlinexperts"
HOLDOUT_DTYPE = "int4_wo"

# Stored bytes per weight. MXFP4: e2m1 nibbles plus one E8M0 byte per 32 values (OCP MX
# block size). int4 holdout: nibbles plus one BF16 scale per group of 32.
BYTES_PER_WEIGHT = {FIT_DTYPE: 0.5 + 1 / 32, HOLDOUT_DTYPE: 0.5 + 2 / 32}

# The parts this family is committed for, and why each: h200 serves DeepSeek-V4.1-Flash's
# MXFP4 experts on Marlin (#38); gb300 serves Kimi-K3's, which the KDA validation in #25
# cannot score until the W4A16 rate is stated.
PARTS = [("h200_sxm", "h200"), ("gb300", "gb300")]

NAMES = ("gemm_eps_max_w4a16_mxfp4_grouped", "gemm_m_half_w4a16_mxfp4_grouped")

EPS_GRID = np.round(np.arange(0.001, 1.0005, 0.001), 3)
MHALF_GRID = np.arange(1, 4097)


def chip_facts(catalog: Path, chip: str) -> tuple[float, float]:
    """(BF16 dense peak in FLOP/s, datasheet HBM bandwidth in B/s), from the catalog."""
    facts = yaml.safe_load((catalog / "hardware" / f"{chip}.yaml").read_text())
    return float(facts["TFlopsPeak"]) * 1e12, float(facts["BwPeakTBs"]) * 1e12


def hbm_derate(chip: str) -> float:
    """The committed hbm_derate for this chip, which the kernel multiplies BwPeak by."""
    doc = yaml.safe_load(SET_PATH.read_text())
    for e in doc["coefficients"]:
        ((name, body),) = e.items()
        if name == "hbm_derate" and chip in body["scope"].get("hardware", []):
            return float(body["value"])
    raise SystemExit(f"{chip}: no committed hbm_derate; the memory half cannot be charged")


def load(data: Path, sku: str, kernel: str, dtype: str) -> list[dict]:
    path = data / sku / "moe" / COLLECTION / "moe_perf.parquet"
    if not path.is_file():
        raise SystemExit(f"{sku}: {path} is absent; the pinned collection is required")
    t = pq.read_table(path).to_pydict()
    rows = []
    for i in range(len(t["latency"])):
        if (t["kernel_source"][i] != kernel or t["moe_dtype"][i] != dtype
                or t["distribution"][i] != "balanced" or t["latency"][i] <= 0):
            continue
        rows.append({k: t[k][i] for k in (
            "num_tokens", "hidden_size", "inter_size", "topk", "num_experts",
            "moe_tp_size", "moe_ep_size")} | {"s": t["latency"][i] * 1e-3})
    return rows


def terms(rows: list[dict], dtype: str, bw: float) -> tuple[np.ndarray, ...]:
    """Per row: tokens, routed FLOPs, the kernel's memory-half seconds, measured seconds."""
    T, F, H, S = [], [], [], []
    for r in rows:
        tp, ep = r["moe_tp_size"], r["moe_ep_size"]
        h, i = r["hidden_size"], r["inter_size"] / tp
        k, e = r["topk"], r["num_experts"]
        tokens = r["num_tokens"]
        T.append(tokens)
        F.append(tokens * k / ep * 6 * h * i)
        touched = (e / ep) * (1 - ((e - k) / e) ** tokens)
        H.append(touched * 3 * h * i * BYTES_PER_WEIGHT[dtype] / bw)
        S.append(r["s"])
    return tuple(np.asarray(x, dtype=float) for x in (T, F, H, S))


def predict(T, F, H, peak, eps, mh):
    return F / (peak * eps * T / (T + mh)) + H


def errors(T, F, H, S, peak, eps, mh) -> tuple[float, float]:
    """(geometric mean |error|, geometric mean signed ratio), as the committed fits report."""
    r = np.log(predict(T, F, H, peak, eps, mh) / S)
    return math.exp(float(np.mean(np.abs(r)))), math.exp(float(np.mean(r)))


def fit(T, F, H, S, peak) -> tuple[float, int, float]:
    """Exhaustive grid over (eps_max, M_half), minimising the geometric error."""
    mh = MHALF_GRID[:, None].astype(float)
    best = (math.inf, 0.0, 0)
    for eps in EPS_GRID:
        pred = F[None, :] / (peak * eps * T[None, :] / (T[None, :] + mh)) + H[None, :]
        err = np.mean(np.abs(np.log(pred / S[None, :])), axis=1)
        j = int(np.argmin(err))
        if err[j] < best[0]:
            best = (float(err[j]), float(eps), int(MHALF_GRID[j]))
    return float(best[1]), best[2], math.exp(best[0])


def fit_part(data: Path, catalog: Path, sku: str, chip: str) -> dict:
    peak, bw_peak = chip_facts(catalog, chip)
    bw = bw_peak * hbm_derate(chip)
    rows = load(data, sku, FIT_KERNEL, FIT_DTYPE)
    train = [r for r in rows if r["moe_tp_size"] == 1 and r["moe_ep_size"] == 1]
    geoms = {(r["hidden_size"], r["inter_size"], r["topk"], r["num_experts"]) for r in train}
    if len(geoms) != 1 or len(train) < 20:
        raise SystemExit(f"{chip}: expected one geometry and >= 20 rows, got {geoms}, "
                         f"{len(train)}")
    T, F, H, S = terms(train, FIT_DTYPE, bw)
    eps, mh, err = fit(T, F, H, S, peak)
    _, signed = errors(T, F, H, S, peak, eps, mh)

    hold = [r for r in load(data, sku, HOLDOUT_KERNEL, HOLDOUT_DTYPE)
            if r["moe_tp_size"] == 1 and r["moe_ep_size"] == 1]
    ho = errors(*terms(hold, HOLDOUT_DTYPE, bw), peak, eps, mh) if hold else None
    (geom,) = geoms
    return {"sku": sku, "chip": chip, "eps": eps, "mh": mh, "err": err, "signed": signed,
            "n": len(train), "geom": geom, "peak_tf": peak / 1e12,
            "holdout": ho, "holdout_n": len(hold),
            }


def entry_lines(name: str, f: dict) -> list[str]:
    h, i, k, e = f["geom"]
    cite = (f"NVIDIA AISimulate systems/data/{f['sku']}/moe/{COLLECTION}/moe_perf.parquet, "
            f"{FIT_KERNEL}, moe_dtype {FIT_DTYPE}, balanced, moe_tp 1 / moe_ep 1, "
            f"{f['n']} token counts (hidden {h}, inter {i}, top-{k} of {e})")
    ho = (f"held out on vLLM's int4 Marlin MoE (hidden 7168, inter 2048, top-8 of 384; "
          f"{f['holdout_n']} token counts) at {f['holdout'][0]:.3f}x, signed "
          f"{f['holdout'][1]:.3f}" if f["holdout"] else "no int4 Marlin holdout on this part")
    if name.startswith("gemm_eps_max"):
        value, units = f"{f['eps']:.3f}", "dimensionless"
        rat = (f"Asymptotic fraction of this part's dense BF16 peak ({f['peak_tf']:g} "
               f"TFLOPS) that vLLM's Marlin MXFP4 W4A16 grouped GEMM reaches. Marlin "
               f"dequantizes the four-bit weights in registers and multiplies against BF16 "
               f"activations with mma.sync, so its rate is a fraction of the BF16 peak, not "
               f"of an FP8 or FP4 one. Fitted with gemm_m_half_w4a16_mxfp4_grouped under "
               f"the kernel's routed-expert composition -- the SM half summed with the "
               f"distinct experts' stored weight bytes (0.53125 B/weight, E8M0 scales "
               f"included) at BwPeak x hbm_derate -- so the ramp is what remains once the "
               f"kernel's own memory half is charged. Geometric error {f['err']:.3f}x "
               f"(signed {f['signed']:.3f}); {ho}. Applies to the routed-expert term only; "
               f"a dense MXFP4 linear runs a different Marlin kernel and is not measured. "
               f"Reproduce with scripts/fit_marlin_grouped_ramp.py.")
    else:
        value, units = str(f["mh"]), "tokens"
        rat = (f"The step token count at which the Marlin MXFP4 W4A16 grouped ramp reaches "
               f"half its asymptote, fitted jointly with gemm_eps_max_w4a16_mxfp4_grouped "
               f"over {f['n']} token counts; the kernel evaluates it at the step's token "
               f"count, as it does every ramp. Reproduce with "
               f"scripts/fit_marlin_grouped_ramp.py.")
    return [
        f"  - {name}:",
        f"      value: {value}",
        f"      units: {units}",
        "      method: measured",
        "      fitted: true",
        f"      scope: {{hardware: [{f['chip']}]}}",
        "      sources:",
        f'        - {{kind: model, cite: "{cite}", role: primary}}',
        "      rationale: >",
        f"        {rat}",
    ]


def render(text: str, fits: dict[str, dict]) -> str:
    """Replace each owned (name, chip) entry in place; insert a missing one after the
    chip's last gemm_* entry, so the family sits with the part's other ramps."""
    blocks = cb.split(text)
    for chip, f in fits.items():
        scope = f"{{hardware: [{chip}]}}"
        for name in NAMES:
            cb.upsert(blocks, entry_lines(name, f),
                      after=lambda b: (cb.name_of(b) or "").startswith("gemm_")
                      and cb.scope_of(b) == scope)
    return cb.join(blocks)


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default=DEFAULT_DATA)
    ap.add_argument("--catalog", default=DEFAULT_CATALOG)
    ap.add_argument("--check", action="store_true",
                    help="exit 1 if the file would change; write nothing")
    args = ap.parse_args(argv[1:])

    fits = {}
    for sku, chip in PARTS:
        f = fit_part(Path(args.data), Path(args.catalog), sku, chip)
        fits[chip] = f
        ho = (f"holdout int4 {f['holdout'][0]:.3f}x signed {f['holdout'][1]:.3f} "
              f"(n={f['holdout_n']})" if f["holdout"] else "no holdout")
        print(f"{chip:8} eps_max={f['eps']:.3f} m_half={f['mh']:4d}  "
              f"train {f['err']:.3f}x signed {f['signed']:.3f} (n={f['n']})  {ho}")

    before = SET_PATH.read_text(encoding="utf-8")
    after = render(before, fits)
    if args.check:
        if after != before:
            print(f"{SET_PATH.name} would change", file=sys.stderr)
            return 1
        print(f"{SET_PATH.name} already matches the fit")
        return 0
    SET_PATH.write_text(after, encoding="utf-8")
    print(f"{SET_PATH.name} written")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
