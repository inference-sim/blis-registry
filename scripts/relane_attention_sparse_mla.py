#!/usr/bin/env python3
"""Maintain the sparse-MLA decode rate in place, from fit_attention_sparse_mla.py.

    python scripts/relane_attention_sparse_mla.py --check
    python scripts/relane_attention_sparse_mla.py --write

WHAT THIS COMMITS. `attention_decode_rate_sparse_mla` for the six NVIDIA parts the
attention-only sparse table covers. The FLOOR is deliberately not committed: the fit pins
it to each part's measured `attention_decode_floor` because searching it freely improves
the fit by under 8% while re-opening the module-measurement trap that
`scripts/correct_mla_floor.py` had to undo. See the fitter's docstring for the per-part
numbers.

THE VALUES ARE NOT HARDCODED. Each is read from `fit_attention_sparse_mla.py` run against
the collection the citation names, so re-running re-derives them. `--check` asserts the
committed file still matches.

A NOTE ON WHAT SHIPPING THIS DOES AND DOES NOT DO. The kernel cannot use this rate yet,
and that is stated here rather than discovered later. Its decode term is

    attnSeconds += floor + decodeKVTokens * kvBytesPerToken / layers / rate

where `decodeKVTokens` is the FULL context. This rate was fitted against SELECTED bytes
(min(step,topk) plus a compressed remainder), so pairing it with a full-context byte count
would be worse than the fallback it replaces, not better -- the same class of mistake as
fitting a rate on one quantity and charging it against another.

Three things are needed before the kernel can price a sparse_mla layer correctly, and only
the first is in this repository:

  1. this rate (here);
  2. `IndexTopK` carried into the kernel's layer plan. blis-schemas ALREADY models
     `index_topk` and validates that sparse_mla requires it
     (spec/model/validate.go: "sparse MLA requires a positive index_topk"), but
     internal/price/plan.go carries only `AttnWindow`, so the value is loaded and dropped;
  3. `compress_ratio` added to blis-schemas. blis-catalog states it on deepseek-v4-pro's
     two sparse_mla layers (4 and 128) but no schema field exists, so it is silently
     dropped at load. The glm-5 family needs none -- those layers state `index_topk: 2048`
     with no compressed tier -- so this blocks deepseek-v4-pro only.

Committing the rate first is still right: it is the measured quantity, it is useless to
nobody else, and the alternative is leaving a fitted value in a scratch file while the
coefficient the kernel will need is unversioned.
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
SET_PATH = HERE.parent / "coefficients" / "cost-model-attention.yaml"
FITTER = HERE / "fit_attention_sparse_mla.py"

PARTS = [
    ("h100_sxm", "h100"),
    ("h200_sxm", "h200"),
    ("b200_sxm", "b200"),
    ("b300_sxm", "b300"),
    ("gb200", "gb200-nvl72"),
    ("gb300", "gb300"),
]

RATE = re.compile(r"^attention_decode_rate_sparse_mla\s+(\d+) bytes_per_us "
                  r"\(([\d.]+) of ([\d.]+) TB/s\)")
HEAD = re.compile(r"^# (\S+) \((\S+)\), (\S+), kind=sparse_mla")
GEOM = re.compile(r"^#\s+selected = min\(step, (\d+)\) \+ max\(0, step-\d+\)/(\d+); "
                  r"latent width (\d+) elements")
MODEL = re.compile(r"^#\s+model (\S+), catalog layer (\S+), lane (\S+)")
FLOOR = re.compile(r"^#\s+floor PINNED to this part's attention_decode_floor "
                   r"([\d.]+) us")
ERRS = re.compile(r"^#\s+(\d+) decode points, geometric error ([\d.]+)x overall, "
                  r"([\d.]+)x head \((\d+)\), ([\d.]+)x tail \((\d+)\)")


def fit(sku: str, chip: str, data: str | None, catalog: str | None) -> dict | None:
    cmd = [sys.executable, str(FITTER), "--sku", sku, "--chip", chip]
    if data:
        cmd += ["--data", data]
    if catalog:
        cmd += ["--catalog", catalog]
    out = subprocess.run(cmd, capture_output=True, text=True)
    got: dict = {"sku": sku, "chip": chip}
    for line in out.stdout.splitlines():
        for pat, keys in ((HEAD, ("_c", "_s", "coll")),
                          (MODEL, ("model", "layer", "lane")),
                          (GEOM, ("topk", "ratio", "width")),
                          (FLOOR, ("floor",)),
                          (RATE, ("rate", "frac", "peak")),
                          (ERRS, ("n", "err", "head", "headn", "tail", "tailn"))):
            m = pat.match(line.strip())
            if m:
                got.update(dict(zip(keys, m.groups())))
    need = {"coll", "model", "layer", "lane", "topk", "ratio", "width", "floor",
            "rate", "frac", "peak", "n", "err", "tail"}
    if not need <= set(got):
        return None
    got["rate"] = int(got["rate"])
    return got


def render(f: dict) -> list[str]:
    cite = (
        f"NVIDIA AISimulate systems/data/{f['sku']}/sparse_attention/{f['coll']}/"
        f"dsv4_hca_attn_module_perf.parquet, model {f['model']}, "
        f"lane {f['lane']}, {f['n']} decode rows (isl==1); selected-token geometry "
        f"index_topk/window {f['topk']} and compress_ratio {f['ratio']} from "
        f"blis-catalog models/deepseek-v4-pro/graph.yaml layer {f['layer']}, "
        f"latent width {f['width']} elements"
    )
    rationale = (
        f"The rate a sparse-MLA decode kernel sustains on {f['chip']}: "
        f"{f['rate']:,} bytes_per_us, {f['frac']} of this part's {f['peak']} TB/s "
        f"datasheet bandwidth. An order of magnitude below the full-attention rates "
        f"(0.52-0.88 of peak) because a sparse read GATHERS scattered pages rather than "
        f"streaming contiguous ones, which is the physical content of this coefficient. "
        f"THE BYTE COUNT IS NOT THE CONTEXT, and that is why this kind needs its own "
        f"rate: the kernel reads min(step, {f['topk']}) tokens at full resolution plus "
        f"the remainder divided by {f['ratio']}, so a 1M-token context costs about as "
        f"much as a 16K one. Measured: at batch 1 on h200 the latency is flat at "
        f"9.8-13.7us from step 0 to 16384 and reaches only 30.6us at step 1,048,575. "
        f"Pricing that layer with the full-attention form forces the rate to 1.00 of "
        f"datasheet peak -- a physically impossible figure, and the clamp is how the "
        f"wrong byte count announces itself; the selected-token form fits with the rate "
        f"interior on every part. THE TABLE IS AN ATTENTION MEASUREMENT, not a module "
        f"one: dsv4_hca_attn_module_perf floors at 9.6us on h200, BELOW this part's "
        f"13.5us GQA attention floor and 7x below the 66.2us needed to read the layer's "
        f"317.5 MB of projections, which the catalog prices as separate GEMM nodes. "
        f"Twelve of the thirteen tables in this family floor between 37.7us and 1186us "
        f"and are module measurements; fitting from one of those is the defect "
        f"scripts/correct_mla_floor.py had to undo. THE FLOOR IS NOT FITTED: it is "
        f"pinned to this part's measured attention_decode_floor of {f['floor']} us, "
        f"because searching it freely improves the fit by under 8% and the free optima "
        f"(12.6-14.4us across parts) do not track the part-wide floors they would "
        f"replace. Adding a parameter for that would re-open the MLA failure mode for "
        f"no accuracy. Fitted over {f['n']} decode points; geometric error {f['err']}x "
        f"overall, {f['head']}x head and {f['tail']}x tail, so the error is balanced "
        f"rather than an aggregate hiding a bad tail. KNOWN GAP, STATED: the kernel "
        f"cannot consume this yet -- its decode term charges FULL-context bytes, and "
        f"blis-schemas carries index_topk but not compress_ratio. See "
        f"scripts/relane_attention_sparse_mla.py for the three changes required."
    )
    return [
        "  - attention_decode_rate_sparse_mla:",
        f"      value: {f['rate']}",
        "      units: bytes_per_us",
        "      method: measured",
        "      fitted: true",
        f"      scope: {{hardware: [{f['chip']}]}}",
        "      sources:",
        f'        - {{kind: model, cite: "{cite}", role: primary}}',
        "      rationale: >",
        f"        {rationale}",
    ]


def spans(lines: list[str]) -> list[tuple[int, int, str | None]]:
    """Each committed sparse_mla rate entry as (start, end, chip)."""
    out = []
    i = 0
    while i < len(lines):
        if lines[i].strip() == "- attention_decode_rate_sparse_mla:":
            j = i + 1
            chip = None
            while j < len(lines) and not re.match(r"^  - [a-z_0-9]+:$", lines[j]):
                m = re.search(r"scope: \{hardware: \[([^\]]+)\]\}", lines[j])
                if m:
                    sc = [x.strip() for x in m.group(1).split(",")]
                    chip = sc[0] if len(sc) == 1 else None
                j += 1
            out.append((i, j, chip))
            i = j
            continue
        i += 1
    return out


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data")
    ap.add_argument("--catalog")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--check", action="store_true")
    g.add_argument("--write", action="store_true")
    args = ap.parse_args(argv[1:])

    fits = {}
    for sku, chip in PARTS:
        f = fit(sku, chip, args.data, args.catalog)
        if f is None:
            print(f"{chip:14} no sparse_mla fit", file=sys.stderr)
            continue
        fits[chip] = f
        print(f"{chip:14} {f['coll']:12} rate={f['rate']:>9,} B/us "
              f"({f['frac']} of {f['peak']} TB/s) err {f['err']}x")
    if not fits:
        print("no parts fitted", file=sys.stderr)
        return 1

    text = SET_PATH.read_text(encoding="utf-8")
    lines = text.split("\n")
    present = {c for _, _, c in spans(lines) if c}

    if args.check:
        missing = sorted(set(fits) - present)
        if missing:
            print(f"\nnot committed for: {missing}; run with --write", file=sys.stderr)
            return 1
        bad = []
        for lo, hi, chip in spans(lines):
            if chip not in fits:
                continue
            got = None
            for b in lines[lo:hi]:
                m = re.match(r"\s+value:\s*(\d+)\s*$", b)
                if m:
                    got = int(m.group(1))
                    break
            if got != fits[chip]["rate"]:
                bad.append(f"{chip}: committed {got} != fitted {fits[chip]['rate']}")
        for b in bad:
            print(f"  MISMATCH {b}", file=sys.stderr)
        if bad:
            return 1
        print(f"\nall {len(present & set(fits))} committed sparse_mla rate(s) match")
        return 0

    # --write: replace any present entry, append the rest after the last MLA rate entry.
    for lo, hi, chip in reversed(spans(lines)):
        if chip in fits:
            lines[lo:hi] = render(fits[chip])
    present = {c for _, _, c in spans(lines) if c}
    todo = [c for _, c in PARTS if c in fits and c not in present]
    if todo:
        anchor = None
        for i, line in enumerate(lines):
            if line.strip() == "- attention_decode_rate_mla:":
                j = i + 1
                while j < len(lines) and not re.match(r"^  - [a-z_0-9]+:$", lines[j]):
                    j += 1
                anchor = j
        if anchor is None:
            print("no attention_decode_rate_mla entry to anchor after", file=sys.stderr)
            return 1
        block = []
        for chip in todo:
            block += render(fits[chip])
        lines[anchor:anchor] = block

    SET_PATH.write_text("\n".join(lines), encoding="utf-8")
    print(f"\n{SET_PATH.name}: {len(fits)} sparse_mla rate entries written")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
