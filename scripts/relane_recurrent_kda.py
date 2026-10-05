#!/usr/bin/env python3
"""Refit the KDA recurrent pair on vLLM's own NVIDIA decode path.

THREE DEFECTS IN THE COMMITTED PAIR, all found by reading vLLM rather than the sweep.

1. WRONG KERNEL. The committed values were fitted on `kda_fused_decode`, described in the
   entry as "the best of the three KDA kernel paths measured". That kernel is AMD-ONLY:
   it lives at `vllm/models/kimi_k3/amd/ops/kda_decode.py` and
   `is_fused_kda_decode_supported` returns False unless `on_gfx950() or on_gfx942()`.
   No CUDA deployment ever runs it.

2. WRONG COMPOSITION. On NVIDIA, `vllm/models/kimi_k3/nvidia/kda.py:980-993` calls
   `causal_conv1d_update` and THEN `fused_recurrent_kda_packed_decode` for the same
   decode step. A layer therefore costs the SUM of the two kernels. The committed entry
   argued against summing -- "the fused figure is also the cheaper and therefore the less
   conservative of the two readings" -- but that reasoning was about the AMD kernel, which
   genuinely fuses the two. The NVIDIA path does not.

3. WRONG GEOMETRY. The fit was taken at `num_k_heads=12`. Kimi-K3's graph in blis-catalog
   declares `n_heads: 96` on its KDA nodes, and the kernel's coefficient carries no head
   term -- `kernel.go` prices a recurrent layer as `floor + tokens/rate` keyed on the kind
   alone -- so the fitted geometry must be the model's. Both lanes measure all four of
   12/24/48/96, so 96 was available and unused.

THE FORM. Two kernels run in series, so their times add. A single `floor + tokens/rate`
form carries that exactly: the floors add, and the reciprocal rates add (1/r = 1/r1 + 1/r2),
which is what this script computes. No new coefficient or kernel change is needed.

SGLANG IS NOT WRONG, IT IS A DIFFERENT ENGINE. SGLang's sweep carries the same two kernels
and fits comparably well at 96 heads (1.121x and 1.180x against vLLM's 1.054x and 1.179x).
The vLLM lane is chosen because BLIS prices vLLM, and the dev-build pin is recorded in the
citation rather than hidden.

Usage:
    python scripts/relane_recurrent_kda.py
    python scripts/relane_recurrent_kda.py --check
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
SET_PATH = HERE.parent / "coefficients" / "cost-model-recurrent.yaml"
DEFAULT_DATA = os.environ.get(
    "AISIMULATE_DATA",
    "/tmp/aisim/python/aisimulate/src/aisimulate_core/systems/data")

SWEEP = "h100_sxm/kda/vllm/0.1.dev19262/kda_perf.parquet"
# vllm/models/kimi_k3/nvidia/kda.py:980-993 -- the conv runs, then the recurrent scan.
CHAIN = ["causal_conv1d_update", "fused_recurrent_kda_packed_decode"]
# blis-catalog models/kimi-k3/graph.yaml: RecurrentUpdate n_heads: 96.
HEADS = 96


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, HERE / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


fr = _load("fit_recurrent")


def fit_chain(path: Path) -> dict:
    rows = fr.read(path)
    gen = [r for r in rows
           if r.get("phase") == "generation" and r["latency"] > 0
           and r.get("num_k_heads") == HEADS]
    if not gen:
        raise SystemExit(f"{path}: no generation rows at num_k_heads={HEADS}")
    parts = {}
    for ks in CHAIN:
        pts = [(r["num_tokens"], r["latency"] * 1000) for r in gen
               if r["kernel_source"] == ks]
        if len(pts) < 8:
            raise SystemExit(f"{path}: only {len(pts)} points for {ks} at {HEADS} heads")
        floor, rate, err = fr.fit(pts)
        parts[ks] = {"floor": floor, "rate": rate, "err": round(err, 3),
                     "n": len(pts)}
    floor = sum(p["floor"] for p in parts.values())
    # Serial kernels: times add, so the reciprocal rates add.
    rate = 1.0 / sum(1.0 / p["rate"] for p in parts.values())
    return {"parts": parts, "floor": round(floor, 1), "rate": round(rate, 3)}


def rewrite(text: str, fit: dict) -> tuple[str, int]:
    lines = text.split("\n")
    out: list[str] = []
    i = 0
    changed = 0
    conv = fit["parts"]["causal_conv1d_update"]
    scan = fit["parts"]["fused_recurrent_kda_packed_decode"]
    cite = (f"NVIDIA AISimulate systems/data/{SWEEP}, kernels "
            f"causal_conv1d_update + fused_recurrent_kda_packed_decode summed, "
            f"generation phase, num_k_heads={HEADS}, model moonshotai/Kimi-K3, "
            f"{conv['n']} and {scan['n']} points")
    while i < len(lines):
        m = re.match(r"  - (recurrent_decode_(floor|rate)_kda):$", lines[i])
        if not m:
            out.append(lines[i])
            i += 1
            continue
        j = i + 1
        while j < len(lines) and not re.match(r"  - [a-z_0-9]+:$", lines[j]):
            j += 1
        kind = m.group(2)
        value = fit["floor"] if kind == "floor" else fit["rate"]
        if kind == "floor":
            rat = (f"The minimum cost of one KDA layer's decode-phase state update. vLLM's "
                   f"NVIDIA path runs causal_conv1d_update and then "
                   f"fused_recurrent_kda_packed_decode for the same step "
                   f"(vllm/models/kimi_k3/nvidia/kda.py:980-993), so a layer costs the SUM: "
                   f"{conv['floor']}us + {scan['floor']}us. Fitted at num_k_heads={HEADS}, "
                   f"which is what Kimi-K3's graph declares; the kernel's coefficient "
                   f"carries no head term, so the fitted geometry must be the model's. "
                   f"Per-kernel geometric error {conv['err']}x and {scan['err']}x. The "
                   f"fused kernel this entry previously used is AMD-only "
                   f"(amd/ops/kda_decode.py gates on gfx942/gfx950) and runs on no CUDA "
                   f"deployment.")
        else:
            rat = (f"Tokens per microsecond past the floor. The two kernels run in series, "
                   f"so their times add and the reciprocal rates add: "
                   f"1/{conv['rate']} + 1/{scan['rate']} tok/us. The recurrent scan "
                   f"dominates, being sequential in the state dimension -- it cannot "
                   f"spread one sequence's work across the machine the way a matmul "
                   f"spreads a batch's. Fitted on the vLLM lane at num_k_heads={HEADS}.")
        out += [
            f"  - {m.group(1)}:",
            f"      value: {value}",
            # `tokens` is the schema's unit for this quantity -- semantically tokens
            # per microsecond, recorded under the in-schema member the mamba2 pair
            # already uses. `tokens_per_us` is not in UNITS and is rejected.
            "      units: " + ("us_per_transfer" if kind == "floor" else "tokens"),
            "      method: measured",
            "      fitted: true",
            "      scope: {hardware: [h100]}",
            "      sources:",
            f'        - {{kind: model, cite: "{cite}", role: primary}}',
            "      rationale: >",
            f"        {rat}",
        ]
        changed += 1
        i = j
    return "\n".join(out), changed


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default=DEFAULT_DATA)
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args(argv[1:])

    fit = fit_chain(Path(args.data) / SWEEP)
    for ks, p in fit["parts"].items():
        print(f"  {ks:36} n={p['n']:3} floor={p['floor']:4.1f}us "
              f"rate={p['rate']:5.2f} tok/us geo-err={p['err']}x")
    print(f"  {'SUM (serial)':36}     floor={fit['floor']:4.1f}us "
          f"rate={fit['rate']:.3f} tok/us")

    before = SET_PATH.read_text(encoding="utf-8")
    after, changed = rewrite(before, fit)
    if args.check:
        if before != after:
            print(f"\n{SET_PATH.name} would change ({changed} entries)", file=sys.stderr)
            return 1
        print(f"\n{SET_PATH.name} already matches the vLLM-lane chain fit")
        return 0
    SET_PATH.write_text(after, encoding="utf-8")
    print(f"\n{SET_PATH.name}: {changed} entries rewritten")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
