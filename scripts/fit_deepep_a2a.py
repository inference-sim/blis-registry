#!/usr/bin/env python3
"""Fit the DeepEP MoE dispatch + combine price per backend from AISimulate moe_a2a_perf.

WHAT THIS CLOSES. blis-latency-kernel priced every routed all-to-all backend -- DeepEP
low-latency, high-throughput and v2 alike -- with NCCL's all-to-all floor and rate
(blis-latency-kernel#57). DeepEP's kernels are not NCCL's: HT synchronises with the host
to size its receive buffer, LL sends one message per (token, expert), v2 runs either an
expanding eager path or a capturable one. This writer fits each from its own measurement.

THE DATA. AISimulate `h100_sxm/comm/vllm/0.24.0/moe_a2a_perf.parquet` (aisimulate
95b7f839, Apache-2.0): vLLM v0.24.0's own DeepEPHT/LL/V2PrepareAndFinalize prepare() and
finalize() timed with CUDA events, BF16 activations (quant_config None, use_fp8_dispatch
False), EP8 on one H100 node, 10 runs after 3 warm-ups. `num_tokens` is PER-RANK source
tokens; `latency` is MICROSECONDS (the collector stores ms x 1000). HT runs 20 SMs, the
v0.31.0 default (DeepEPAll2AllManagerBase.num_sms, all2all.py:177). v2 has two rows
families: `deepep_v2_context` (use_cudagraph False: do_expand, CPU sync) and
`deepep_v2_generation` (use_cudagraph True); v0.31.0 picks between them by
`not enforce_eager` (all2all_utils.py:286-298), not per step.

WHY v0.24.0 / DeepEP 73b6ea4a MEASURES v0.31.0 / DeepEP d4f41e4e93.
  * vLLM. `git diff v0.24.0 v0.31.0` over prepare_finalize/deepep_ht.py adds only a
    ROCm-only DBO synchronize (`sync_dbo_comm = current_platform.is_rocm()`); deepep_ll.py
    changes docstrings only. deepep_v2.py DOES change: the prefill-mode topk_idx rebuild
    becomes one repeat_interleave instead of a fill_ per local expert, and finalize_async
    issues the combine asynchronously to overlap the shared expert. So v2's eager rows
    overstate v0.31.0 by up to one small launch per local expert, and the graph rows'
    combine is the synchronous form. Stated on the v2 entries.
  * DeepEP. v0.31.0 pins DEEPEP_COMMIT_HASH=d4f41e4e93 (docker/Dockerfile:516); the
    collection ran 73b6ea4a (HT/LL) and b306af06 (v2). 73b6ea4a..d4f41e4e93 moves the
    legacy sources under csrc/legacy; the host bodies of intranode_dispatch,
    intranode_combine, low_latency_dispatch, low_latency_combine and get_dispatch_layout
    are identical modulo the LEGACY_ constant prefix, intranode.cu changes only that prefix
    plus a barrier used by destroy(), and internode_ll.cu adds one
    fence.proxy.async.shared::cta (#642) and changes const-reference locals to values.
    b306af06..d4f41e4e93 is six commits; the one on this path is #630, "fix V2
    initialization for pure single-node intra EP". The residual is unmeasured.

THE FORM, per backend and phase:

    latency_us = floor + bytes / rate,   bytes = tokens x hidden x 2 x top_k

the basis the kernel already prices a routed all-to-all on (RoutedByTopK). A third
(peak-rate) parameter cut no phase's fit error by more than 0.008 in mean |log ratio|, so
it is not fitted. The kernel charges dispatch and combine as ONE collective per MoE layer,
so the committed pair is their sum: floor_d + floor_c, and 1 / (1/rate_d + 1/rate_c).

THE SPLIT, declared before any held-out value was read:
  * held-out A, token count: HT and v2 context {32, 128, 512, 2048, 8192, 32768}; LL and
    v2 generation {2, 8, 16, 32, 64, 128}; every other token count is fitted.
  * held-out B, hidden size: the whole (hidden 6144, top-8 of 256) configuration, which is
    what interpolating DeepSeek-V4.1-Flash's hidden 5120 by bytes rests on.
  * band: median |relative error| <= 15% and max <= 35%, per backend, on the sum.

THE GATE, AS DECIDED. No backend meets the max criterion: it sits below the data's own
scatter between two shapes with identical bytes (printed as the noise floor). HT, LL and
v2-eager meet the median and land as a recorded deviation; v2-graph misses the median too
and lands marked LOW CONFIDENCE in its rationale (the schema has no evidence-class field).

Usage:
    python scripts/fit_deepep_a2a.py [--data <aisimulate systems/data>] [--table out.md]
    python scripts/fit_deepep_a2a.py --check
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import sys
from math import comb
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import yaml

REPO = Path(__file__).resolve().parent.parent
SET_PATH = REPO / "coefficients" / "cost-model-collectives.yaml"
DEFAULT_DATA = os.environ.get(
    "AISIMULATE_DATA", "/tmp/aisim/python/aisimulate/src/aisimulate_core/systems/data")
COLLECTION = "h100_sxm/comm/vllm/0.24.0/moe_a2a_perf.parquet"
SHA256 = "c077f3e4a57d37ccd0695e5146aef090648830844318f681e917be2920a799a6"
CHIP, RANKS = "h100", 8

_spec = importlib.util.spec_from_file_location(
    "coefficient_blocks", Path(__file__).parent / "coefficient_blocks.py")
cb = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(cb)

# The kernel's name for each variant, and the AISimulate backend that measures it.
VARIANTS = {
    "deepep_high_throughput": "deepep_ht",
    "deepep_low_latency": "deepep_ll",
    "deepep_v2_eager": "deepep_v2_context",
    "deepep_v2_graph": "deepep_v2_generation",
}
HELD_TOKENS = {
    "deepep_ht": {32, 128, 512, 2048, 8192, 32768},
    "deepep_v2_context": {32, 128, 512, 2048, 8192, 32768},
    "deepep_ll": {2, 8, 16, 32, 64, 128},
    "deepep_v2_generation": {2, 8, 16, 32, 64, 128},
}
HELD_SHAPE = (6144, 8, 256)
BAND_MEDIAN, BAND_MAX = 0.15, 0.35

# DeepEP README v1 (deepseek-ai/DeepEP 567632dd, H800, ~160 GB/s NVLink): intranode EP8
# normal kernels at 4096 tokens, hidden 7168, top-8, BF16 combine, 158 GB/s logical.
README_COMBINE_GBPS = 158.0
H100_NVLINK_UNIDIR_GBPS = 450.0


def load(data: Path):
    path = data / COLLECTION
    cols = ["comm_backend", "phase", "ep_size", "node_num", "hidden_size", "topk",
            "num_experts", "num_tokens", "latency"]
    t = pq.read_table(path, columns=cols).to_pandas()
    t = t[(t.ep_size == RANKS) & (t.node_num == 1)]
    keys = ["comm_backend", "hidden_size", "topk", "num_experts", "num_tokens"]
    wide = t.pivot_table(index=keys, columns="phase", values="latency").reset_index()
    wide["bytes"] = wide.num_tokens * wide.hidden_size * 2.0 * wide.topk
    shape = list(zip(wide.hidden_size, wide.topk, wide.num_experts))
    wide["held_shape"] = [s == HELD_SHAPE for s in shape]
    wide["held_tokens"] = [n in HELD_TOKENS[b] for b, n in
                           zip(wide.comm_backend, wide.num_tokens)]
    wide["fit"] = ~(wide.held_shape | wide.held_tokens)
    return wide


def fit_phase(b: np.ndarray, y: np.ndarray) -> tuple[float, float, float]:
    """floor (us) and rate (B/us) minimising mean |log(pred / y)|, by a deterministic
    log-grid search refined three times around its best point."""
    lo_f, hi_f = np.log(max(y.min() * 0.2, 1e-3)), np.log(y.min() * 1.2)
    lo_r, hi_r = np.log(1e3), np.log(1e8)
    best = None
    for _ in range(4):
        fl = np.exp(np.linspace(lo_f, hi_f, 241))
        rt = np.exp(np.linspace(lo_r, hi_r, 241))
        pred = fl[:, None, None] + b[None, None, :] / rt[None, :, None]
        loss = np.mean(np.abs(np.log(pred / y[None, None, :])), axis=2)
        i, j = np.unravel_index(np.argmin(loss), loss.shape)
        best = (float(fl[i]), float(rt[j]), float(loss[i, j]))
        sf, sr = (hi_f - lo_f) / 24, (hi_r - lo_r) / 24
        lo_f, hi_f = np.log(fl[i]) - sf, np.log(fl[i]) + sf
        lo_r, hi_r = np.log(rt[j]) - sr, np.log(rt[j]) + sr
    return best


def fit_all(wide):
    out = {}
    for variant, backend in VARIANTS.items():
        g = wide[wide.comm_backend == backend]
        f = g[g.fit]
        ph = {p: fit_phase(f.bytes.values, f[p].values) for p in ("dispatch", "combine")}
        floor = round(ph["dispatch"][0] + ph["combine"][0], 1)
        rate = float(f"{1.0 / (1.0 / ph['dispatch'][1] + 1.0 / ph['combine'][1]):.4g}")
        out[variant] = {"backend": backend, "phases": ph, "floor": floor, "rate": rate,
                        "rows": g, "n_fit": len(f)}
    return out


def errors(fit, rows):
    pred = fit["floor"] + rows.bytes.values / fit["rate"]
    y = (rows["dispatch"] + rows["combine"]).values
    return pred / y - 1.0


def generic_before(text):
    """The price the kernel gave every routed backend before this fit: NCCL's 8-rank fp16
    all-to-all on this part, max(floor + bytes / transition, bytes / peak)."""
    want = {f"collective_{k}_alltoall_fp16_{RANKS}rank_{CHIP}": k
            for k in ("floor", "peak_rate", "transition_rate")}
    got = {}
    for e in yaml.safe_load(text)["coefficients"]:
        (name, body), = e.items()
        if name in want and body["scope"]["hardware"] == [CHIP]:
            got[want[name]] = float(body["value"])
    return lambda b: np.maximum(got["floor"] + b / got["transition_rate"],
                                b / got["peak_rate"])


def noise_floor(rows):
    """Two measured shapes share the byte basis exactly -- hidden 7168, top-8 of 256 and of
    384 -- so no bytes-only form can split them; their disagreement is a floor under any
    such form's error."""
    a = rows[(rows.hidden_size == 7168) & (rows.topk == 8) & (rows.num_experts == 256)]
    b = rows[(rows.hidden_size == 7168) & (rows.topk == 8) & (rows.num_experts == 384)]
    m = a.merge(b, on="num_tokens")
    r = np.abs((m.dispatch_x + m.combine_x) / (m.dispatch_y + m.combine_y) - 1.0)
    return float(np.median(r)), float(np.max(r))


def summarise(fits, before):
    lines = ["| variant | set | n | median abs err | max abs err | signed median | "
             "in band | before: median abs / signed |",
             "|---|---|---:|---:|---:|---:|---|---|"]
    verdict = {}
    for v, f in fits.items():
        g = f["rows"]
        verdict[v] = True
        for name, sel in (("fit", g.fit), ("held-out A (tokens)", g.held_tokens & ~g.held_shape),
                          ("held-out B (hidden 6144)", g.held_shape)):
            e = errors(f, g[sel])
            y = (g[sel]["dispatch"] + g[sel]["combine"]).values
            eb = before(g[sel].bytes.values) / y - 1.0
            med, mx = float(np.median(np.abs(e))), float(np.max(np.abs(e)))
            ok = "" if name == "fit" else ("yes" if med <= BAND_MEDIAN and mx <= BAND_MAX
                                           else "NO")
            if ok == "NO":
                verdict[v] = False
            lines.append(f"| {v} | {name} | {len(e)} | {med:.1%} | {mx:.1%} | "
                         f"{float(np.median(e)):+.1%} | {ok} | "
                         f"{float(np.median(np.abs(eb))):.1%} / {float(np.median(eb)):+.1%} |")
        nm, nx = noise_floor(g)
        lines.append(f"| {v} | noise floor (same bytes, E 256 vs 384) | | {nm:.1%} | "
                     f"{nx:.1%} | | | |")
    return lines, verdict


def readme_check(fits):
    """Held-out C: the DeepEP README's H800 intranode HT combine bandwidth. Different
    hardware, so it bounds rather than scores: H100's logical bandwidth must sit between
    H800's NVLink-capped figure and H100's own NVLink rate."""
    t, h, k, e, r = 4096, 7168, 8, 256, RANKS
    per = e // r
    distinct = r * (1 - comb(e - per, k) / comb(e, k))
    fl, rt, _ = fits["deepep_high_throughput"]["phases"]["combine"]
    us = fl + t * h * 2 * k / rt
    gbps = t * distinct * h * 2 / us / 1e3
    ok = README_COMBINE_GBPS <= gbps <= H100_NVLINK_UNIDIR_GBPS
    return (f"held-out C: HT combine at 4096 tokens, hidden 7168, top-8 of 256 prices "
            f"{us:.0f} us = {gbps:.0f} GB/s logical (DeepEP's recv_x bytes, "
            f"{distinct:.2f} ranks per token); README H800 {README_COMBINE_GBPS:.0f} GB/s, "
            f"H100 NVLink {H100_NVLINK_UNIDIR_GBPS:.0f} GB/s: "
            f"{'within' if ok else 'OUTSIDE'} the bound"), ok


RATIONALE = {
    "deepep_high_throughput": (
        "DeepEP high-throughput (intranode, 20 SMs). Its floor is mostly the host "
        "synchronisation dispatch makes to size the receive buffer (notify_dispatch, then "
        "a CPU spin on moe_recv_counter), which vLLM v0.31.0 always runs eager at dp > 1 "
        "(vllm/config/compilation.py:1237-1253), so the measured eager floor is the cost "
        "it pays."),
    "deepep_low_latency": (
        "DeepEP low-latency (NVLink intranode, allow_nvlink_for_low_latency_mode=True as "
        "v0.31.0 builds it, all2all.py:316-327). Measured EAGER: the floor includes vLLM's "
        "host issue time, which a captured graph does not pay, so under CUDA graphs it is "
        "an upper bound. Measured to 256 tokens per rank; above that the rate is "
        "extrapolated."),
    "deepep_v2_eager": (
        "DeepEP v2 ElasticBuffer with use_cudagraph False (do_expand, CPU sync): what "
        "v0.31.0 runs under enforce_eager (all2all_utils.py:286-298). v0.24.0's eager "
        "prepare rebuilt topk_idx with one fill_ per local expert where v0.31.0 runs one "
        "repeat_interleave, so this floor overstates v0.31.0 by up to that many small "
        "launches."),
    "deepep_v2_graph": (
        "LOW CONFIDENCE: the one variant outside the declared median band (held out ~25%), "
        "on rows whose own same-bytes scatter is 61% median; landed as a large improvement "
        "over the generic price, not as validated. "
        "DeepEP v2 ElasticBuffer with use_cudagraph True (no expand, no CPU sync): what "
        "v0.31.0 runs at every step size whenever enforce_eager is off "
        "(all2all_utils.py:286-298). Measured eager and to 256 tokens per rank only, so "
        "the floor is an upper bound under graphs and the rate is extrapolated above 256 "
        "tokens; v0.31.0 also overlaps this combine with the shared expert "
        "(deepep_v2.py finalize_async), which the synchronous v0.24.0 rows do not."),
}


def entry_lines(name, value, units, variant, f, table_ref):
    ph = f["phases"]
    cite = (f"NVIDIA AISimulate systems/data/{COLLECTION} (aisimulate 95b7f839, sha256 "
            f"{SHA256[:12]}), backend {f['backend']}, EP8 one node, BF16")
    what = (f"dispatch {ph['dispatch'][0]:.1f} us + {ph['dispatch'][1] / 1e3:.1f} GB/s and "
            f"combine {ph['combine'][0]:.1f} us + {ph['combine'][1] / 1e3:.1f} GB/s per "
            f"phase (mean |log| {ph['dispatch'][2]:.3f} / {ph['combine'][2]:.3f})")
    rat = (f"{RATIONALE[variant]} One MoE layer's dispatch plus combine as "
           f"floor + tokens x hidden x 2 x top_k / rate, per-rank tokens: {what}, "
           f"summed. Fitted on {f['n_fit']} (shape, token) rows; {table_ref} Reproduce "
           f"with scripts/fit_deepep_a2a.py.")
    return [
        f"  - {name}:",
        f"      value: {value}",
        f"      units: {units}",
        "      method: measured",
        "      fitted: true",
        f"      scope: {{hardware: [{CHIP}]}}",
        "      sources:",
        f'        - {{kind: model, cite: "{cite}", role: primary}}',
        "      rationale: >",
        f"        {rat}",
    ]


def held_out_note(f):
    g = f["rows"]
    a = errors(f, g[g.held_tokens & ~g.held_shape])
    b = errors(f, g[g.held_shape])
    return (f"Held out by token count: median {np.median(np.abs(a)):.1%}, max "
            f"{np.max(np.abs(a)):.1%}; by hidden size (6144): median "
            f"{np.median(np.abs(b)):.1%}, max {np.max(np.abs(b)):.1%}.")


def render(text, fits):
    blocks = cb.split(text)
    scope = f"{{hardware: [{CHIP}]}}"
    for variant, f in fits.items():
        note = held_out_note(f)
        for name, value, units in (
                (f"moe_a2a_floor_{variant}_{RANKS}rank_{CHIP}", f["floor"], "us_per_transfer"),
                (f"moe_a2a_rate_{variant}_{RANKS}rank_{CHIP}", f["rate"], "bytes_per_us")):
            cb.upsert(blocks, entry_lines(name, value, units, variant, f, note),
                      after=lambda b: (cb.name_of(b) or "").startswith(
                          ("collective_transition_rate_alltoall_fp16_8rank_h100",
                           "moe_a2a_"))
                      and cb.scope_key(cb.scope_of(b) or "{}") == cb.scope_key(scope))
    return cb.join(blocks)


def main(argv):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default=DEFAULT_DATA)
    ap.add_argument("--table", help="write the fit / held-out table (markdown) here")
    ap.add_argument("--check", action="store_true",
                    help="exit 1 if the file would change; write nothing")
    args = ap.parse_args(argv[1:])

    fits = fit_all(load(Path(args.data)))
    before_text = SET_PATH.read_text(encoding="utf-8")
    lines, verdict = summarise(fits, generic_before(before_text))
    readme, readme_ok = readme_check(fits)
    for v, f in fits.items():
        print(f"{v:24} floor {f['floor']:7.1f} us  rate {f['rate'] / 1e3:7.1f} GB/s")
    report = "\n".join(lines + ["", readme, "",
                                "verdict: " + ", ".join(f"{v} {'pass' if ok else 'FAIL'}"
                                                        for v, ok in verdict.items())])
    print(report)
    if args.table:
        Path(args.table).write_text(report + "\n", encoding="utf-8")

    before = before_text
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
