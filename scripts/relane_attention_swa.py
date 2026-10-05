#!/usr/bin/env python3
"""Relane the sliding-window decode pair onto the vLLM collection, in place.

WHY THIS SCRIPT EXISTS. The twelve `attention_decode_*_swa` entries were the last
attention coefficients fitted on TRT-LLM while every GQA and prefill entry had moved to
the vLLM lane. A model with mixed attention kinds — gpt-oss-120b alternates `swa`
(window 128) and `gqa` across 18 repeats, minimax-m3 across 57 — then has ONE predicted
step time summing vLLM-lane GQA layers and TRT-LLM-lane SWA layers. That is an engine
mixture inside a single number, and no score justifies it.

It rewrites rather than regenerates because `cost-model-attention.yaml` is hand-authored
(unlike the collectives set): only the ten entries named on the command line change, and
the file's other thirty are passed through byte-for-byte.

THE VALUES ARE NOT HARDCODED. Each is read from `fit_attention_by_kind.py` run against
the collection given, so re-running this script re-derives them. A fit that moves moves
the file; a fit that does not leaves it byte-identical.

L40S IS DEFERRED, NOT FORGOTTEN. Its vLLM-lane fit is WORSE than the committed TRT-LLM
entry (2.110x against it), so relaning it would trade provenance for accuracy. Pass it
explicitly to override that; the default part list omits it.

Usage:
    python scripts/relane_attention_swa.py --collection vllm/0.25.0
    python scripts/relane_attention_swa.py --collection vllm/0.25.0 --check
    python scripts/relane_attention_swa.py --collection vllm/0.25.0 h200_sxm:h200
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
SET_PATH = HERE.parent / "coefficients" / "cost-model-attention.yaml"

# The parts with a vLLM sliding-window sweep whose fit beats the committed TRT-LLM one.
# l40s is absent deliberately — see the module docstring.
DEFAULT_PARTS = [
    ("h200_sxm", "h200"),
    ("h100_sxm", "h100"),
    ("b200_sxm", "b200"),
    ("b300_sxm", "b300"),
    ("gb200", "gb200-nvl72"),
]

DEVICE = {
    "h200": "NVIDIA H200",
    "h100": "NVIDIA H100 80GB HBM3",
    "b200": "NVIDIA B200",
    "b300": "NVIDIA B300",
    "gb200-nvl72": "NVIDIA GB200",
    "l40s": "NVIDIA L40S",
}

# `  swa   floor  14.0us  rate   864,000 B/us (0.18 of 4.800 TB/s)  geo-err 1.390x
#  (naive 121.9x)  n=31,890 lane=...`
SWA_LINE = re.compile(
    r"^\s*swa\s+floor\s+([0-9.]+)us\s+rate\s+([0-9,]+) B/us\s+"
    r"\(([0-9.]+) of ([0-9.]+) TB/s\)\s+geo-err ([0-9.]+)x.*?n=([0-9,]+)"
)


def fit(sku: str, chip: str, collection: str) -> dict | None:
    """Run the committed per-kind fitter and parse its sliding-window row."""
    out = subprocess.run(
        [sys.executable, str(HERE / "fit_attention_by_kind.py"),
         "--sku", sku, "--chip", chip, "--collection", collection],
        capture_output=True, text=True, check=False,
    )
    if out.returncode != 0:
        raise SystemExit(f"{sku}:{chip}: fitter failed\n{out.stderr}")
    for line in out.stdout.splitlines():
        m = SWA_LINE.match(line)
        if m:
            return {
                "floor": float(m.group(1)),
                # The fitter prints the rate in B/us, which is the unit the registry
                # stores (`units: bytes_per_us`). Scaling here once produced a
                # 1000x error written to all ten entries.
                "rate": float(m.group(2).replace(",", "")),
                "frac": m.group(3),
                "peak": m.group(4),
                "err": m.group(5),
                "n": int(m.group(6).replace(",", "")),
            }
    return None


def rationale(kind: str, chip: str, collection: str, f: dict) -> str:
    if kind == "floor":
        return (
            f"The minimum cost of a windowed decode kernel, independent of context. Fitted on "
            f"the vLLM lane ({collection}) — the engine BLIS prices, and the lane every GQA and "
            f"prefill entry in this set already uses, so a model with both kinds no longer sums "
            f"two engines in one step. Paired with attention_decode_rate_swa over {f['n']:,} "
            f"windowed points; geometric error {f['err']}x."
        )
    return (
        f"{f['rate'] / 1e6:.2f} TB/s, {f['frac']} of this part's {f['peak']} TB/s datasheet "
        f"bandwidth. A windowed kernel reads a bounded slice of the context rather than all of "
        f"it and sustains materially less effective bandwidth than a full-attention kernel on "
        f"the same silicon, so pricing it with the full-attention rate makes every windowed "
        f"layer too cheap. Fitted on the vLLM lane ({collection}) over {f['n']:,} points."
    )


def rewrite(text: str, fits: dict[str, dict], collection: str) -> tuple[str, int]:
    """Replace the floor/rate entries for every chip in `fits`; pass everything else through."""
    lines = text.split("\n")
    out: list[str] = []
    i = 0
    changed = 0
    while i < len(lines):
        m = re.match(r"  - (attention_decode_(floor|rate)_swa):$", lines[i])
        if not m:
            out.append(lines[i])
            i += 1
            continue
        j = i + 1
        while j < len(lines) and not re.match(r"  - [a-z_0-9]+:$", lines[j]):
            j += 1
        block = lines[i:j]
        chip = None
        for b in block:
            sm = re.search(r"scope: \{hardware: \[([^\]]+)\]\}", b)
            if sm:
                scoped = [x.strip() for x in sm.group(1).split(",")]
                # A single-part scope is what the SWA entries carry; a shared scope would
                # make "which part was fitted" ambiguous, so leave it to a human.
                chip = scoped[0] if len(scoped) == 1 else None
        if chip is None or chip not in fits:
            out.extend(block)
            i = j
            continue
        f = fits[chip]
        kind = m.group(2)
        value = f["floor"] if kind == "floor" else f["rate"]
        cite = (
            f"NVIDIA AISimulate systems/data/{f['sku']}/attention/{collection}/"
            f"generation_attention_perf.parquet ({DEVICE[chip]}), "
            f"{f['n']:,} sliding-window decode points"
        )
        out += [
            f"  - {m.group(1)}:",
            f"      value: {value}",
            f"      units: {'us_per_transfer' if kind == 'floor' else 'bytes_per_us'}",
            "      method: measured",
            "      fitted: true",
            f"      scope: {{hardware: [{chip}]}}",
            "      sources:",
            f'        - {{kind: model, cite: "{cite}", role: primary}}',
            "      rationale: >",
            f"        {rationale(kind, chip, collection, f)}",
        ]
        changed += 1
        i = j
    return "\n".join(out), changed


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--collection", default="vllm/0.25.0",
                    help="lane/version under <sku>/attention (default vllm/0.25.0)")
    ap.add_argument("--check", action="store_true",
                    help="exit 1 if the file would change; write nothing")
    ap.add_argument("parts", nargs="*", metavar="sku:chip",
                    help=f"default: {' '.join(f'{s}:{c}' for s, c in DEFAULT_PARTS)}")
    args = ap.parse_args(argv[1:])

    parts = ([tuple(p.split(":", 1)) for p in args.parts] if args.parts
             else DEFAULT_PARTS)
    fits: dict[str, dict] = {}
    for sku, chip in parts:
        f = fit(sku, chip, args.collection)
        if f is None:
            print(f"{chip:14} no sliding-window rows at {args.collection}", file=sys.stderr)
            continue
        f["sku"] = sku
        # Unit guard. The rate is a fraction of this part's datasheet bandwidth, so
        # frac * peak(TB/s) * 1e6 is the same number in B/us. A parse or scaling slip
        # shows up here rather than in the registry: a 1000x error once reached all ten
        # entries because nothing cross-checked the magnitude.
        implied = float(f["frac"]) * float(f["peak"]) * 1e6
        if not 0.5 * implied <= f["rate"] <= 2.0 * implied:
            raise SystemExit(
                f"{chip}: rate {f['rate']:,.0f} B/us contradicts the fitter's own "
                f"{f['frac']} of {f['peak']} TB/s (= {implied:,.0f} B/us); "
                "refusing to write a value whose unit cannot be confirmed"
            )
        fits[chip] = f
        print(f"{chip:14} floor={f['floor']:5.1f}us  rate={f['rate']:>12,.0f} B/us  "
              f"geo-err {f['err']}x  n={f['n']:,}")
    if not fits:
        print("no parts fitted; nothing to do", file=sys.stderr)
        return 1

    before = SET_PATH.read_text(encoding="utf-8")
    after, changed = rewrite(before, fits, args.collection)
    if args.check:
        if before != after:
            print(f"\n{SET_PATH.name} would change ({changed} entries)", file=sys.stderr)
            return 1
        print(f"\n{SET_PATH.name} already matches the {args.collection} fit")
        return 0
    SET_PATH.write_text(after, encoding="utf-8")
    print(f"\n{SET_PATH.name}: {changed} entries rewritten on {args.collection}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
