#!/usr/bin/env python3
"""DeepSeek-V4.1-Flash on H200: the attention, indexer, weight-only GEMM and host-gather
coefficients vLLM v0.31.0's kernels need, each measured where a kernel-level measurement
of the right kernel exists and declared `assumed` where none does.

    python scripts/fit_deepseek_v41_h200.py            # print the fits, rewrite both sets
    python scripts/fit_deepseek_v41_h200.py --check    # exit 1 if either set would change

Owns six entries, two fitted and four assumed (inference-sim/blis-registry#37, #38, #39):

  cost-model-attention.yaml, scoped {hardware: [h200], model: [deepseek-v4.1-flash]}
    attention_decode_rate_sparse_mla           measured  FlashMLA sparse decode
    attention_decode_floor_block_index_scores  measured  indexer MQA-logits scan floor
    attention_decode_rate_block_index_scores   assumed   indexer scan rate (HBM bound)
  cost-model-primitives.yaml, scoped {hardware: [h200]}
    gemm_eps_max_w8a16_mxfp8, gemm_m_half_w8a16_mxfp8   assumed  Marlin MXFP8 W8A16 dense
    host_gather_rate_uva                                assumed  engram rows over UVA

WHAT vLLM v0.31.0 RUNS ON SM90 FOR THIS MODEL (read from the tag, db9527a4):
  * Sparse attention: with no explicit backend, SM90 falls through to
    `DeepseekV4FlashMLAAttention` (`_select_dsv4_attn_cls`, nvidia/model.py: FlashInfer is
    SM12x or an explicit FLASHINFER_* backend, mega attention SM100 only), whose
    `_forward_decode` (nvidia/flashmla.py) makes ONE `flash_mla_with_kvcache` call per
    layer: the sliding window (`indices=swa_indices`) plus the indexer's top-k as an extra
    cache (`extra_indices_in_kvcache`), FP8 KV, query heads padded to 64
    (`get_padded_num_q_heads`: 64 for any count up to 64, so V4.1's 64 heads hit the h64
    kernel at EVERY tensor-parallel width). The KV record is the V4 one on every arch but
    SM100 -- 448 FP8 NoPE + 64 BF16 RoPE + 8 scale/pad bytes, 584 B
    (`_use_v41_mxfp8_kv_record`, attention.py).
  * Indexer scan: `SparseAttnIndexer` -> DeepGEMM `fp8_fp4_paged_mqa_logits` over the
    compressed index keys, FP8 with an FP32 scale, 132 B per key
    (`_indexer_k_cache_head_dim`). MXFP4 keys and the sparse-logits path both require
    SM100 (`dsa_indexer_uses_fp4`, `AttentionConfig.indexer_sparse_logits`), so on H200
    the scan is dense over ceil(context / compress_ratio) keys per request.
  * Dense linears: `MarlinMxfp8LinearKernel`, "MXFP8 W8A16 GEMM via Marlin"
    (model_executor/kernels/linear/mxfp8/marlin.py); the FlashInfer MXFP8 kernels ahead of
    it require SM100.
  * Engram: tables in pinned host memory by default (`EngramConfig.cpu_offload = True`),
    gathered by `_engram_lookup_kernel` (models/deepseek_v41/common/engram.py) through UVA.

WHY THE vLLM 0.25.0 SWEEPS MEASURE THE v0.31.0 KERNELS. AISimulate has no vLLM collection
newer than 0.25.0 for h200_sxm. For the two attention kernels the device code was compared
across the pins the two tags build:
  * FlashMLA (vllm-project/FlashMLA a6ec2ba -> 0eee43b): `csrc/sm90/decode/sparse_fp8/*`
    became `csrc/kernels/sm90/decode/sparse/*` with only namespace, include-path and
    MODEL1 -> V4 renames; the 584 B `BYTES_PER_TOKEN` assert is unchanged.
  * DeepGEMM (deepseek-ai/DeepGEMM a6b593d -> vllm-project/DeepGEMM e1f418c; v0.31.0
    builds from vLLM's fork, deepgemm.cmake): `sm90_fp8_paged_mqa_logits.cuh` and its
    scheduler differ by renames and by bounds handling for zero-length contexts and
    block-table rows past a request's pages; the 64-token compute loop is unchanged.

THE TABLES, AND WHY EACH IS (OR IS NOT) A KERNEL MEASUREMENT.
  * `dsv4_hca_attn_module_perf` (vllm/0.25.0, FLASHMLA_SPARSE_DSV4): times `forward_mqa`,
    i.e. the FlashMLA call with no projections, floor ~10 us -- below this part's 13.5 us
    attention floor, so it is the kernel and not the block (methodology section 10.1). Fitted
    on the DeepSeek-V4-Flash-FP8 rows: 64 query heads, the same h64 instantiation and 584 B
    record V4.1 runs. Its rows are compress_ratio 128, whose compressed tier is read in
    full; the kernel walks SWA blocks then extra blocks in one 64-token-block loop
    (`num_orig_kv_blocks`, splitkv_mla.cuh), so the cost of a request is its BLOCK count
    whatever tier the blocks come from. V4.1 reads at most 2 + 8 = 10 blocks (window 128,
    top-512) per query, inside the 1-130 blocks the table covers; both counts are derived
    from the configs, and the writer refuses if V4.1's would leave the measured range.
  * `dsv4_paged_mqa_logits_module_perf` (vllm/0.25.0): times `fp8_fp4_paged_mqa_logits`
    itself under CUDA graphs. Its FLOOR is the kernel's. Its SLOPE is not an HBM rate: the
    collector gives every query row the same block table, so the unique key footprint is at
    most 34.6 MB and stays resident in H200's 50 MB L2, and the fitted throughput is 1.12x
    the HBM datasheet peak -- impossible from HBM. In serving each request's keys are its
    own and HBM-resident, so the scan's rate is bounded by HBM, which is what is committed.
  * `dsv41/sglang/dev-1aa0e962.../dsv41_module_perf` is NOT used: it is SGLang, eager (no
    CUDA graph), a module timing (`MQALayer.forward`, projections included), batch <= 3 and
    context <= 2,049, flat at 0.8-2.0 ms -- launch-bound, not a kernel rate.

UNITS. `latency` is MILLISECONDS in every AISimulate parquet.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

import pyarrow.parquet as pq
import yaml

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import coefficient_blocks as cb  # noqa: E402

ATTN_PATH = HERE.parent / "coefficients" / "cost-model-attention.yaml"
PRIM_PATH = HERE.parent / "coefficients" / "cost-model-primitives.yaml"
DEFAULT_DATA = os.environ.get(
    "AISIMULATE_DATA",
    "/tmp/aisim/python/aisimulate/src/aisimulate_core/systems/data")
# The vendored chip descriptors (validator/testdata/catalog), so the writer runs in CI;
# BLIS_CATALOG overrides it with a live checkout.
DEFAULT_CATALOG = (os.environ.get("BLIS_CATALOG")
                   or str(HERE.parent / "validator" / "testdata" / "catalog"))

SKU, CHIP, MODEL = "h200_sxm", "h200", "deepseek-v4.1-flash"
COLLECTION = "vllm/0.25.0"
MODEL_SCOPE = f"{{hardware: [{CHIP}], model: [{MODEL}]}}"
PART_SCOPE = f"{{hardware: [{CHIP}]}}"

FLASHMLA_TABLE = "dsv4_hca_attn_module_perf.parquet"
FLASHMLA_LANE = "FLASHMLA_SPARSE_DSV4"
FLASHMLA_FIT_MODEL = "sgl-project/DeepSeek-V4-Flash-FP8"
INDEXER_TABLE = "dsv4_paged_mqa_logits_module_perf.parquet"
INDEXER_LANE = "vllm.utils.deep_gemm.fp8_fp4_paged_mqa_logits"

# FlashMLA's TOPK_BLOCK_SIZE, its unit of work. Also the collector's indexer page size
# (`block_kv = 64`, collect_dsv4_attn.py), which is why one constant serves both.
BLOCK = 64
INDEXER_RATIO = 4     # the measured indexer rows are DeepSeek-V4's compress_ratio 4
H200_L2_BYTES = 50e6  # the H100/H200 SXM product's L2 (the full GH100 die has 60 MB)

# Engram rows: 256 FP8 values in one table, 8 UE8M0 scales in a separate one
# (ParallelEngramEmbedding._allocate_weights). The value row is 256 B at a 256 B stride,
# two whole 128 B lines; the 8 scale bytes cost at least one 32 B sector -- the minimum
# a global load moves (CUDA C++ Best Practices Guide, "Coalesced Access to Global Memory").
ENGRAM_ROW_BYTES = 256 + 8
ENGRAM_FETCHED_BYTES = 256 + 32

RATE_FRACTIONS = [x / 1000 for x in range(5, 1001)]
INDEXER_MODELS = {"sgl-project/DeepSeek-V4-Flash-FP8", "sgl-project/DeepSeek-V4-Pro-FP8"}


def interior(value, grid, what: str) -> None:
    """A fit pinned to its grid's edge is the grid's answer, not the data's."""
    if not math.isfinite(value) or value <= grid[0] or value >= grid[-1]:
        raise SystemExit(f"{what} {value} is at the edge of its grid [{grid[0]}, {grid[-1]}]")


def model_config(data: Path, name: str) -> dict:
    path = data.parent.parent / "model_configs" / f"{name}_config.json"
    cfg = json.loads(path.read_text())
    return cfg.get("text_config", cfg)


def committed(path: Path, name: str, chip: str) -> float:
    doc = yaml.safe_load(path.read_text())
    for e in doc["coefficients"]:
        ((n, body),) = e.items()
        if n == name and body["scope"].get("hardware") == [chip] and \
                "model" not in body["scope"]:
            return float(body["value"])
    raise SystemExit(f"{chip}: no committed part-wide {name}")


def catalog_facts(catalog: Path) -> tuple[float, float]:
    """(HBM datasheet bandwidth in bytes per microsecond, dense FP8 peak in TFLOPS)."""
    facts = yaml.safe_load((catalog / "hardware" / f"{CHIP}.yaml").read_text())
    return float(facts["BwPeakTBs"]) * 1e6, float(facts["TFlopsFP8"])


def read(data: Path, table: str) -> dict:
    path = data / SKU / "sparse_attention" / COLLECTION / table
    if not path.is_file():
        raise SystemExit(f"{path} is absent; the pinned collection is required")
    return pq.read_table(path).to_pydict()


def geo(points, floor, rate) -> tuple[float, float]:
    """(geometric mean |error|, geometric mean signed ratio) of floor + work/rate."""
    logs = [math.log((floor + w / rate) / y) for w, y in points]
    return (math.exp(sum(abs(x) for x in logs) / len(logs)),
            math.exp(sum(logs) / len(logs)))


def check_geometry(data: Path) -> dict:
    """Refuse unless V4.1 and the fitted checkpoints share the kernel geometry."""
    v41 = model_config(data, "deepseek-ai--DeepSeek-V4.1-Flash")
    flash = model_config(data, "sgl-project--DeepSeek-V4-Flash-FP8")
    pro = model_config(data, "sgl-project--DeepSeek-V4-Pro-FP8")

    def padded(h):  # DeepseekV4FlashMLAAttention.get_padded_num_q_heads
        return 64 if h <= 64 else 128

    if not padded(v41["num_attention_heads"]) == padded(flash["num_attention_heads"]) == 64:
        raise SystemExit("V4.1 and the fitted V4-Flash rows do not share the h64 kernel")
    for k in ("head_dim", "qk_rope_head_dim", "sliding_window"):
        if v41[k] != flash[k]:
            raise SystemExit(f"{k} differs between V4.1 and V4-Flash; the record differs")
    if set(v41["compress_ratios"]) - {0, 1, 2}:
        raise SystemExit(f"V4.1 compress_ratios {set(v41['compress_ratios'])}")
    for c in (flash, pro):
        if c["index_head_dim"] != v41["index_head_dim"]:
            raise SystemExit("indexer head_dim differs; the 132 B key does not hold")
    if flash["index_n_heads"] != pro["index_n_heads"]:
        raise SystemExit("the two measured checkpoints differ in index heads; pooling them "
                         "would pool two kernel configurations")
    nope = v41["head_dim"] - v41["qk_rope_head_dim"]
    window, topk = v41["sliding_window"], v41["index_topk"]
    return {
        # FlashMLA SM90 V4 BYTES_PER_TOKEN = HEAD_DIM_NOPE + 2*HEAD_DIM_ROPE + 8
        "record": nope + 2 * v41["qk_rope_head_dim"] + 8,
        "window": window, "topk": topk,
        # The most a V4.1 sparse layer reads per query, in FlashMLA blocks and tokens.
        "v41_blocks": math.ceil(window / BLOCK) + math.ceil(topk / BLOCK),
        "v41_tokens": window + topk,
        "key": v41["index_head_dim"] + v41["index_head_dim"] // 128 * 4,
        "v41_index_heads": v41["index_n_heads"],
        "fit_index_heads": flash["index_n_heads"],
    }


def fit_flashmla(data: Path, g: dict, peak: float, floor: float) -> dict:
    t = read(data, FLASHMLA_TABLE)
    pts, blocks_of = [], []
    for i in range(len(t["latency"])):
        if t["isl"][i] != 1 or t["model"][i] != FLASHMLA_FIT_MODEL:
            continue
        # Every fitted row must be the configuration V4.1 runs; anything else refuses
        # rather than pooling silently.
        if (t["kernel_source"][i] != FLASHMLA_LANE or t["compress_ratio"][i] != 128
                or t["kv_cache_dtype"][i] != "fp8" or t["local_num_heads"][i] != 64
                or not (math.isfinite(t["latency"][i]) and t["latency"][i] > 0)):
            raise SystemExit(f"unexpected FlashMLA row {i}: {t['kernel_source'][i]}, ratio "
                             f"{t['compress_ratio'][i]}, kv {t['kv_cache_dtype'][i]}, heads "
                             f"{t['local_num_heads'][i]}, latency {t['latency'][i]}")
        ctx = t["step"][i] + 1
        blocks = (math.ceil(min(ctx, g["window"]) / BLOCK)
                  + math.ceil((ctx // 128) / BLOCK))
        pts.append((t["batch_size"][i] * blocks * BLOCK * g["record"],
                    t["latency"][i] * 1000.0))
        blocks_of.append(blocks)
    if len(pts) < 150:
        raise SystemExit(f"only {len(pts)} FlashMLA decode rows")
    best = min((geo(pts, floor, peak * f)[0], f) for f in RATE_FRACTIONS)
    frac = best[1]
    interior(frac, RATE_FRACTIONS, "FlashMLA rate fraction")
    rate = int(round(peak * frac))
    err, signed = geo(pts, floor, rate)
    # Free-floor fit, reported to show what pinning the floor costs.
    free = min((geo(pts, fl / 10, peak * f)[0], fl / 10, f)
               for fl in range(50, 200) for f in RATE_FRACTIONS[::5])
    # Hold out the long-extra regime: train on <= 3 blocks per request, test on >= 5.
    tr = [p for p, b in zip(pts, blocks_of) if b <= 3]
    te = [p for p, b in zip(pts, blocks_of) if b >= 5]
    trf = min((geo(tr, floor, peak * f)[0], f) for f in RATE_FRACTIONS)[1]
    v41 = [p for p, b in zip(pts, blocks_of) if b == g["v41_blocks"]]
    if g["v41_blocks"] > max(blocks_of):
        raise SystemExit(f"V4.1 reads {g['v41_blocks']} blocks per query, beyond the "
                         f"{max(blocks_of)} measured; the rate would be an extrapolation")
    return {"n": len(pts), "rate": rate, "frac": frac, "err": err, "signed": signed,
            "free_err": free[0], "free_floor": free[1],
            "hold_n": len(te), "hold": geo(te, floor, peak * trf),
            "v41_n": len(v41), "v41": geo(v41, floor, rate), "floor": floor,
            "max_blocks": max(blocks_of)}


def fit_indexer(data: Path, g: dict) -> dict:
    t = read(data, INDEXER_TABLE)
    pts, uniq, models = [], [], []
    for i in range(len(t["latency"])):
        if t["isl"][i] != 1:
            continue
        if (t["kernel_source"][i] != INDEXER_LANE or t["compress_ratio"][i] != INDEXER_RATIO
                or t["kv_cache_dtype"][i] != "fp8" or t["mla_dtype"][i] != "fp8_e4m3"
                or t["model"][i] not in INDEXER_MODELS
                or not (math.isfinite(t["latency"][i]) and t["latency"][i] > 0)):
            raise SystemExit(f"unexpected indexer row {i}: {t['kernel_source'][i]}, "
                             f"{t['model'][i]}, keys {t['mla_dtype'][i]}, latency "
                             f"{t['latency'][i]}")
        b, s = t["batch_size"][i], t["step"][i]
        # The collector's context_lens: (past_kv + row + 1) // 4, at least 1, per row.
        row_keys = sum(max((s + r) // INDEXER_RATIO, 1) for r in range(1, b + 1))
        pts.append((row_keys, t["latency"][i] * 1000.0))
        models.append(t["model"][i])
        # Every row shares one block table, so the unique keys are one request's.
        uniq.append(math.ceil(max((s + b) // INDEXER_RATIO, 1) / BLOCK) * BLOCK * g["key"])
    if len(pts) < 150 or set(models) != INDEXER_MODELS:
        raise SystemExit(f"only {len(pts)} indexer decode rows over {sorted(set(models))}")
    floors = [x / 100 for x in range(200, 800)]
    rates = [10 ** (3 + k / 500) for k in range(1501)]  # 1e3 .. 1e6 row-keys per us
    best = min((geo(pts, fl, r)[0], fl, r) for fl in floors[::10] for r in rates[::10])
    _, fl0, r0 = best
    best = min((geo(pts, fl, r)[0], fl, r)
               for fl in floors if abs(fl - fl0) <= 0.2
               for r in rates if 0.8 * r0 <= r <= 1.25 * r0)
    err, floor, rate = best
    interior(floor, floors, "indexer floor")
    interior(rate, rates, "indexer work rate")
    floor = round(floor, 1)
    implied = rate * g["key"]  # bytes per microsecond, if each row-key were a key read
    small = [p for p in pts if p[0] / rate < 0.05 * floor]
    # Each checkpoint on its own, to show the floor is the kernel's rather than one model's.
    per_model = {}
    for m in sorted(set(models)):
        mp = [p for p, x in zip(pts, models) if x == m]
        per_model[m] = min((geo(mp, fl, r)[0], fl, r)
                           for fl in floors if abs(fl - floor) <= 0.3
                           for r in rates[::5] if 0.8 * rate <= r <= 1.25 * rate)[1]
    return {"n": len(pts), "floor": floor, "rate_rowkeys": rate, "err": err,
            "per_model": per_model,
            "signed": geo(pts, floor, rate)[1], "implied": implied,
            "max_unique": max(uniq), "small_n": len(small),
            "small": geo(small, floor, rate)}


def lines(name, value, units, method, fitted, scope, sources, rationale):
    out = [f"  - {name}:", f"      value: {value}", f"      units: {units}",
           f"      method: {method}", f"      fitted: {'true' if fitted else 'false'}",
           f"      scope: {scope}", "      sources:"]
    out += [f'        - {{kind: {k}, cite: "{c}", role: {r}}}' for k, c, r in sources]
    return out + ["      rationale: >", f"        {rationale}"]


def entries(g: dict, fm: dict, ix: dict, peak: float, fp8_tflops: float, derate: float,
            bf16_eps: float, bf16_mh: int, grouped_eps: float, grouped_mh: int,
            host_bw: float) -> tuple[list, list]:
    tb = peak / 1e6
    ridge = fp8_tflops * 1e6 / peak                       # FLOP per byte at the FP8 peak
    fit_heads = g["fit_index_heads"]
    measured_tflops = ix["rate_rowkeys"] * fit_heads * 256 / 1e6
    # Speculative decoding scores k+1 rows per key read; at the measured compute rate,
    # the key rate that sustains, as a fraction of the committed HBM bound, for k = 5.
    spec_keys = measured_tflops * 1e6 / (6 * g["v41_index_heads"] * 256)
    spec_frac = spec_keys * g["key"] / (peak * derate)
    floors = ", ".join(f"{v:.2f} us" for v in ix["per_model"].values())
    aisim = "NVIDIA AISimulate systems/data/h200_sxm/sparse_attention/vllm/0.25.0/"
    hbm_rate = int(round(peak * derate))
    gather = int(round(host_bw * ENGRAM_ROW_BYTES / ENGRAM_FETCHED_BYTES))
    attn = [
        lines("attention_decode_rate_sparse_mla", f"{fm['rate']}.0", "bytes_per_us",
              "measured", True, MODEL_SCOPE,
              [("model", f"{aisim}{FLASHMLA_TABLE} (NVIDIA H200), {FLASHMLA_LANE}, "
                f"{FLASHMLA_FIT_MODEL} (64 query heads, the h64 kernel), "
                f"{fm['n']} decode points", "primary"),
               ("model", "vLLM v0.31.0 models/deepseek_v41/nvidia/flashmla.py "
                "DeepseekV4FlashMLAAttention._forward_decode", "supporting")],
              f"{fm['rate'] / 1e6:.2f} TB/s, {fm['frac']:.3f} of this part's {tb:g} TB/s "
              f"datasheet bandwidth, for vLLM v0.31.0's FlashMLA sparse decode on SM90 -- "
              f"one flash_mla_with_kvcache call per layer over the sliding window plus the "
              f"indexer's top-k, query heads padded to 64 at every TP width, the "
              f"{g['record']} B V4 KV record. BYTES ARE COUNTED IN FLASHMLA'S 64-TOKEN "
              f"BLOCKS: per request and per query token, {g['record']} x 64 x (ceil(window "
              f"tokens / 64) + ceil(selected tokens / 64)), which once the context fills the "
              f"window and the top-{g['topk']} is {g['v41_blocks']} blocks, "
              f"{g['v41_blocks'] * BLOCK * g['record']:,} B. The floor is this part's "
              f"part-wide attention_decode_floor ({fm['floor']:g} us), which "
              f"blis-latency-kernel inherits for a kind with a rate and no floor; fitting "
              f"it freely lands at {fm['free_floor']:.1f} us for "
              f"{fm['free_err']:.3f}x against {fm['err']:.3f}x pinned, so the data does not "
              f"ask for a separate one. Geometric error {fm['err']:.3f}x (signed "
              f"{fm['signed']:.3f}) over {fm['n']} points spanning 1-{fm['max_blocks']} "
              f"blocks per request; trained on requests of <= 3 blocks and held out on "
              f">= 5 it predicts {fm['hold'][0]:.3f}x (signed {fm['hold'][1]:.3f}, "
              f"n={fm['hold_n']}); at exactly V4.1's {g['v41_blocks']} blocks "
              f"{fm['v41'][0]:.3f}x (signed {fm['v41'][1]:.3f}, n={fm['v41_n']}, batch <= 8 "
              f"-- larger batches at {g['v41_blocks']} blocks are outside the sweep's batch x "
              f"context <= 1M grid). WHY THIS SHIPS WHEN DEEPSEEK-V4-PRO'S WAS WITHDRAWN "
              f"(methodology section 10.3): that rate was fitted on Pro's 128-head rows (the "
              f"h128 instance, which Pro runs only below TP 2) and was to be charged against "
              f"the kernel's selection rule for Pro's csa4 layers -- up to 262,912 tokens "
              f"at a 1M context, far outside any measured row. Every V4.1 sparse layer reads "
              f"at most {g['v41_tokens']} tokens per query ({g['window']} window + top-"
              f"{g['topk']}) on the h64 instance at every TP width, inside the measured "
              f"range; the kernel charges that read once blis-latency-kernel#32 drops the "
              f"compressed tier and keeps the window. SCOPED TO THIS "
              f"MODEL so no other sparse_mla deployment changes. The FlashMLA SM90 decode "
              f"device code is unchanged between the FlashMLA pins of vLLM 0.25.0 and "
              f"0.31.0 (a6ec2ba, 0eee43b: renames only). END-TO-END CHECK PENDING: FPM "
              f"carries DeepSeek-V4.1-Flash on h200, but blis-latency-kernel cannot yet "
              f"price this model -- it refuses the fp8_ds_mla cache, and its KV bytes, "
              f"sparse selection rule and FlashMLA head padding are wrong for it "
              f"(blis-latency-kernel#36, #30, #32, #39); "
              f"see methodology section 10.5. Reproduce with "
              f"scripts/fit_deepseek_v41_h200.py."),
        lines("attention_decode_floor_block_index_scores", ix["floor"], "us_per_transfer",
              "measured", True, MODEL_SCOPE,
              [("model", f"{aisim}{INDEXER_TABLE} (NVIDIA H200), {INDEXER_LANE}, "
                f"DeepSeek-V4-Flash-FP8 and V4-Pro-FP8 (64 index heads, head_dim 128, "
                f"FP8 keys), {ix['n']} decode points", "primary"),
               ("model", "vLLM v0.31.0 model_executor/layers/sparse_attn_indexer.py, "
                "fp8_fp4_paged_mqa_logits decode call", "supporting")],
              f"The launch-and-schedule floor of DeepGEMM's fp8_fp4_paged_mqa_logits, the "
              f"lightning indexer's scoring scan that vLLM v0.31.0 runs on SM90 for every "
              f"index-source layer of DeepSeek-V4.1 (FP8 keys and dense logits: MXFP4 keys "
              f"and the sparse-logits path need SM100). Timed under CUDA graphs on the "
              f"kernel itself, not a module. Fitted jointly with a per-row-key work rate "
              f"over {ix['n']} points (geometric error {ix['err']:.3f}x); the floor is "
              f"identified by the {ix['small_n']} points whose work term is under 5% of it "
              f"({ix['small'][0]:.3f}x, signed {ix['small'][1]:.3f}). The work rate is NOT "
              f"committed -- see attention_decode_rate_block_index_scores. The measured "
              f"checkpoints carry {fit_heads} index heads where V4.1 carries "
              f"{g['v41_index_heads']}; a floor is a launch cost, and fitted on each "
              f"checkpoint alone it is {floors}. The kernel and scheduler are unchanged across the "
              f"DeepGEMM pins of vLLM 0.25.0 and 0.31.0 except for zero-length bounds "
              f"handling. Covers the logits kernel only: the top-k and V4.1's candidate "
              f"select/mask kernels are separate launches this does not include. Consumed "
              f"by no reader yet: the kernel prices a role: block_index_scores node with "
              f"its layer kind's pair and needs to look this name up "
              f"(attention_decode_<floor|rate>_<role>). Reproduce with "
              f"scripts/fit_deepseek_v41_h200.py."),
        lines("attention_decode_rate_block_index_scores", f"{hbm_rate}.0", "bytes_per_us",
              "assumed", False, MODEL_SCOPE,
              [("model", f"{aisim}{INDEXER_TABLE} (NVIDIA H200): fitted throughput "
                f"{ix['implied'] / 1e6:.2f} TB/s-equivalent at {g['key']} B/key, above the "
                f"{tb:g} TB/s HBM peak", "upper_bound"),
               ("model", "NVIDIA AISimulate systems/h200_sxm.yaml "
                "mem_bw_empirical_scaling_factor (hbm_derate)", "primary")],
              f"ASSUMED: the HBM bound, BwPeak x hbm_derate = {tb:g} TB/s x {derate:g}. "
              f"Bytes are {g['key']} per compressed index key per request ({g['key'] - 4} "
              f"FP8 + one FP32 scale, _indexer_k_cache_head_dim), over ceil(context / "
              f"compress_ratio) keys -- V4.1's compress ratios are 1 and 2, so the scan "
              f"grows with context while the attention it feeds does not. WHY NOT THE "
              f"MEASURED SLOPE: the only kernel-level sweep gives every query row the same "
              f"block table, so its unique key footprint is at most "
              f"{ix['max_unique'] / 1e6:.1f} MB and stays in H200's 50 MB L2; its fitted "
              f"throughput, {ix['implied'] / 1e6:.2f} TB/s-equivalent, exceeds the HBM "
              f"datasheet peak, which an HBM read cannot. In serving each request's keys are "
              f"its own and HBM-resident. Compute does not bind at one query token per "
              f"request: {g['v41_index_heads']} heads x 128 x 2 = "
              f"{g['v41_index_heads'] * 256:,} FLOP per {g['key']} B key is "
              f"{g['v41_index_heads'] * 256 / g['key']:.0f} FLOP/B against an FP8 ridge of "
              f"{ridge:.0f}. DIRECTION: an upper bound on the rate, so a lower bound on scan "
              f"time. Under speculative decoding each request scores k+1 query rows against "
              f"each key, so k = 5 brings it to {6 * g['v41_index_heads'] * 256 / g['key']:.0f} "
              f"FLOP/B -- still under the ridge at datasheet peak, but at the "
              f"{measured_tflops:.0f} TFLOPS this kernel was measured sustaining (a lower "
              f"bound on what it can) the scan could become compute-bound at as little as "
              f"{spec_frac:.2f} of this rate. TO REPLACE IT: the same "
              f"sweep with a distinct block table per query row, so the footprint exceeds "
              f"L2. Scoped to this model; consumed by no reader yet (see the floor)."),
    ]
    prim = [
        lines("gemm_eps_max_w8a16_mxfp8", bf16_eps, "dimensionless", "assumed", False,
              PART_SCOPE,
              [("model", "NVIDIA AISimulate systems/data/h200_sxm/gemm/vllm/0.25.0/"
                "gemm_perf.parquet via gemm_eps_max_bf16 (cuBLAS BF16 envelope)",
                "upper_bound"),
               ("model", "vLLM v0.31.0 model_executor/kernels/linear/mxfp8/marlin.py "
                "MarlinMxfp8LinearKernel", "supporting")],
              f"ASSUMED: this part's measured BF16 GEMM asymptote, as an UPPER BOUND. On "
              f"SM90 vLLM v0.31.0 runs every MXFP8 dense linear through "
              f"MarlinMxfp8LinearKernel (W8A16; the FlashInfer MXFP8 kernels ahead of it "
              f"require SM100): FP8 weights dequantized in registers, BF16 activations, "
              f"mma.sync. Its compute rate is therefore a fraction of the BF16 peak, and "
              f"the measured cuBLAS BF16 envelope ({bf16_eps:g} of peak) bounds it above. "
              f"No sweep measures dense Marlin on any part, so the fraction it reaches is "
              f"unknown; the same Marlin MMA core in its grouped (MoE) form reaches "
              f"{grouped_eps:g} (gemm_eps_max_w4a16_mxfp4_grouped); if dense Marlin reaches "
              f"what the grouped form does, the bound is loose by about "
              f"{bf16_eps / grouped_eps:.1f}x at prefill-sized M. At decode-sized M "
              f"the kernel's HBM term -- the FP8 "
              f"weight read -- dominates, and this term barely enters. DIRECTION: optimistic "
              f"for compute-bound GEMMs. TO REPLACE IT: an h200 GEMM sweep with "
              f"MarlinMxfp8LinearKernel forced (AISimulate's vLLM GEMM collector has no "
              f"Marlin path). Read by no consumer until blis-latency-kernel selects a ramp "
              f"by weight format and chip (blis-latency-kernel#34)."),
        lines("gemm_m_half_w8a16_mxfp8", bf16_mh, "tokens", "assumed", False, PART_SCOPE,
              [("model", "NVIDIA AISimulate systems/data/h200_sxm/gemm/vllm/0.25.0/"
                "gemm_perf.parquet via gemm_m_half_bf16", "primary")],
              f"ASSUMED: this part's measured cuBLAS BF16 knee (gemm_m_half_bf16), paired "
              f"with gemm_eps_max_w8a16_mxfp8 so the two describe one ramp rather than an "
              f"asymptote from one kernel and a knee from another. The knee is where a GEMM "
              f"stops being operand-starved; Marlin's grouped form measures {grouped_mh} "
              f"tokens on this part (gemm_m_half_w4a16_mxfp4_grouped), against {bf16_mh} "
              f"here, so the dense Marlin knee is unmeasured within that range. DIRECTION: "
              f"NOT a bound. A knee above Marlin's own lowers the ramp at mid M, so this "
              f"half is likely pessimistic there while the asymptote is optimistic at large "
              f"M; the two partly offset and neither direction is established. At decode "
              f"M the kernel's HBM term -- the FP8 weight read -- dominates either way. TO "
              f"REPLACE IT: the same MarlinMxfp8LinearKernel sweep as the asymptote; both "
              f"come from one fit."),
        lines("host_gather_rate_uva", f"{gather}.0", "bytes_per_us", "assumed", False,
              PART_SCOPE,
              [("model", "NVIDIA AISimulate systems/h200_sxm.yaml pcie_bw "
                "(host_link_bandwidth)", "primary"),
               ("model", "vLLM v0.31.0 models/deepseek_v41/common/engram.py "
                "_engram_lookup_kernel, ParallelEngramEmbedding", "supporting")],
              f"ASSUMED: an UPPER BOUND on the useful-byte rate of DeepSeek-V4.1's engram "
              f"lookup when the tables are in pinned host memory (EngramConfig.cpu_offload, "
              f"the v0.31.0 default) and read through UVA. Derived as host_link_bandwidth "
              f"({host_bw / 1e3:g} GB/s, PCIe 5.0 x16 per direction) x {ENGRAM_ROW_BYTES}/"
              f"{ENGRAM_FETCHED_BYTES}: a row is {ENGRAM_ROW_BYTES} useful bytes -- 256 FP8 "
              f"values in one table at a 256 B stride (two whole 128 B lines) and 8 UE8M0 "
              f"scales in a separate table, which cost at least one 32 B sector -- so at "
              f"least {ENGRAM_FETCHED_BYTES} B cross the link per row. Per token and engram "
              f"layer a rank reads cdiv(24, shards) rows. WHAT THE BOUND IGNORES, each "
              f"making the real rate LOWER: PCIe transaction-layer overhead; GPU address "
              f"translation for random rows over a table of 384 million rows, ~101 GB per layer "
              f"before TP sharding (vLLM's own comment: 'The "
              f"table dwarfs TLB reach'); the latency-bound regime at decode batch sizes, "
              f"where 24 rows are a few kilobytes and a per-launch round trip dominates -- "
              f"no floor is stated because none is measured; and host-side contention when "
              f"every GPU of a node gathers from one host's memory. The lookup runs on a "
              f"side stream at half the SMs, so how much of it a step exposes is a "
              f"composition question, not this rate's. TO REPLACE IT: time "
              f"ParallelEngramEmbedding.lookup(background=True) with cpu_offload on an H200 "
              f"host over 1-8,192 tokens at TP 1 and 8, and fit floor + bytes / rate. Read "
              f"by no consumer until the schema can state a host-resident gather "
              f"(blis-schemas#48, blis-latency-kernel#38)."),
    ]
    return attn, prim


def render(attn_text: str, prim_text: str, attn: list, prim: list) -> tuple[str, str]:
    a = cb.split(attn_text)
    for e in attn:
        # After the last attention entry, so the family sits at the end of the set.
        cb.upsert(a, e, after=lambda b: cb.name_of(b) is not None)
    p = cb.split(prim_text)
    for e in prim:
        stem = "gemm_" if cb.name_of(e).startswith("gemm_") else "host_link_bandwidth"
        cb.upsert(p, e, after=lambda b, s=stem: (cb.name_of(b) or "").startswith(s)
                  and cb.scope_of(b) == PART_SCOPE)
    return cb.join(a), cb.join(p)


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default=DEFAULT_DATA)
    ap.add_argument("--catalog", default=DEFAULT_CATALOG)
    ap.add_argument("--check", action="store_true",
                    help="exit 1 if either set would change; write nothing")
    args = ap.parse_args(argv[1:])
    data, catalog = Path(args.data), Path(args.catalog)

    g = check_geometry(data)
    peak, fp8_tflops = catalog_facts(catalog)
    floor = committed(ATTN_PATH, "attention_decode_floor", CHIP)
    fm = fit_flashmla(data, g, peak, floor)
    ix = fit_indexer(data, g)
    if ix["max_unique"] >= H200_L2_BYTES or ix["implied"] <= peak:
        raise SystemExit(
            f"the indexer sweep no longer looks L2-resident (footprint "
            f"{ix['max_unique'] / 1e6:.1f} MB, throughput {ix['implied'] / 1e6:.2f} TB/s "
            f"against {peak / 1e6:g} TB/s): fit its rate instead of assuming the HBM bound")
    derate = committed(PRIM_PATH, "hbm_derate", CHIP)
    bf16_eps = committed(PRIM_PATH, "gemm_eps_max_bf16", CHIP)
    bf16_mh = int(committed(PRIM_PATH, "gemm_m_half_bf16", CHIP))
    host_bw = committed(PRIM_PATH, "host_link_bandwidth", CHIP)
    grouped_eps = committed(PRIM_PATH, "gemm_eps_max_w4a16_mxfp4_grouped", CHIP)
    grouped_mh = int(committed(PRIM_PATH, "gemm_m_half_w4a16_mxfp4_grouped", CHIP))

    print(f"flashmla sparse decode  rate={fm['rate']:,} B/us ({fm['frac']:.3f} of peak) "
          f"floor pinned {floor:g} us  {fm['err']:.3f}x signed {fm['signed']:.3f} "
          f"(n={fm['n']}); free floor {fm['free_floor']:.1f} us {fm['free_err']:.3f}x; "
          f"holdout >=5 blocks {fm['hold'][0]:.3f}x (n={fm['hold_n']}); "
          f"10 blocks {fm['v41'][0]:.3f}x (n={fm['v41_n']})")
    print(f"indexer scan            floor={ix['floor']} us  {ix['err']:.3f}x (n={ix['n']}) "
          f"per checkpoint {ix['per_model']}; "
          f"throughput {ix['rate_rowkeys']:,.0f} row-keys/us = "
          f"{ix['implied'] / 1e6:.2f} TB/s-equivalent, footprint <= "
          f"{ix['max_unique'] / 1e6:.1f} MB (L2-resident): rate assumed at HBM bound "
          f"{peak * derate:,.0f} B/us")
    print(f"w8a16 mxfp8 ramp        assumed = bf16 ({bf16_eps:g}, {bf16_mh})")
    print(f"host gather (UVA)       assumed = {host_bw:,.0f} x "
          f"{ENGRAM_ROW_BYTES}/{ENGRAM_FETCHED_BYTES}")

    attn, prim = entries(g, fm, ix, peak, fp8_tflops, derate, bf16_eps, bf16_mh,
                         grouped_eps, grouped_mh, host_bw)
    a0, p0 = ATTN_PATH.read_text(encoding="utf-8"), PRIM_PATH.read_text(encoding="utf-8")
    a1, p1 = render(a0, p0, attn, prim)
    if args.check:
        stale = [p.name for p, x, y in ((ATTN_PATH, a0, a1), (PRIM_PATH, p0, p1)) if x != y]
        if stale:
            print(f"would change: {', '.join(stale)}", file=sys.stderr)
            return 1
        print("both sets already match")
        return 0
    ATTN_PATH.write_text(a1, encoding="utf-8")
    PRIM_PATH.write_text(p1, encoding="utf-8")
    print("both sets written")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
