# How a coefficient in this registry is defined, fitted and checked

This document states the rules. `band-selection.md` records one model-selection decision
made under them; `vllm/docs/perf-model/hypothesis-log.md` records the measurements, the
negative results and the retractions.

## 1. What a coefficient is

A coefficient is a **measured constant of a law the kernel evaluates** — never a fudge
factor and never a per-model correction. Each entry carries:

| field | meaning |
|---|---|
| `value` | the number |
| `units` | stated explicitly, because the single commonest error in this project is a unit slip |
| `method` | `measured`, `vendor_spec`, or `assumed` |
| `fitted` | whether a script produced it |
| `scope` | the deployment dimensions it applies to: hardware, model, TP, EP, nodes |
| `sources` | the exact file the value came from, with row counts |
| `rationale` | why this value, what it replaced, and what rejecting it would cost |

Two rules follow from `method`:

* An `assumed` coefficient is a **declared gap**, not a measurement. Six of them exist
  (`host_admission_per_token`, `host_output_token`, `host_completion`,
  `host_launch_eager_per_layer`, `host_replay_graph_per_step`, `host_launch_per_kernel`).
  Any claim resting on one has to say so.
* A `vendor_spec` coefficient is NVIDIA's observation, not this project's. `hbm_derate`
  at 0.8 is the example.

**A value and its citation move together.** Editing one without the other is how a
registry comes to describe a fit nobody ran, and the validator re-derives values from the
sweep its citation names, so a stale citation fails rather than lies.

**A coefficient with no reader is not live.** `dtypeFit` in the kernel can return only
three dtype suffixes — `bf16`, `fp8`, `nvfp4` — so the 12 `gemm_*_fp8_block` entries are
unreachable. That was discovered by grepping the kernel for the key, not the registry.
Check both.

## 2. The lane rule: fit the engine you predict

AISimulate ships five framework lanes (`vllm` 332 files, `sglang` 313, `trtllm` 217,
`nccl` 23, `oneccl` 2). The lane is part of the path, so choosing it is explicit — and
the lanes are not interchangeable:

| operator | finding |
|---|---|
| decode attention | vLLM/TRT-LLM differ 1.30-1.35x on Hopper and 0.83-0.85x on Blackwell. The SIGN flips, because `vllm/v1/attention/backends/fa_utils.py:99-107` dispatches on compute capability — FA3 at major 9, FA4 at major 10. |
| prefill attention | vLLM runs at 0.79-0.81x of TRT-LLM's on all five parts, concentrated at short prompts: 0.505x at `isl=1` against a ~0.87x plateau above 128. |
| MoE | 1.38-1.41x Hopper, 0.86-0.90x Blackwell, with entirely disjoint `kernel_source` values. |
| GEMM | the lanes AGREE to 0.4-3.6% over 100,668-130,156 shared shapes — same cuBLAS/CUTLASS kernels — so GEMM stays on the larger TRT-LLM sweep. |
| all_gather, reduce_scatter, broadcast | NCCL **is** the vLLM path. `CustomAllreduce` implements all-reduce only; `should_custom_all_gather` and `should_custom_reduce_scatter` gate and fall through. AISimulate has no vLLM data for them because there is nothing to measure. |

So "use the vLLM lane" is a conclusion to be tested per operator, not a blanket rule.
Of 654 lane-citing coefficients, 572 are not on the vLLM lane and most of those are
RIGHT:

| citation | n | verdict |
|---|---|---|
| nccl, all_gather / reduce_scatter / all_to_all | 414 | correct — not implemented by `CustomAllreduce` |
| nccl, all_reduce | 84 | the vLLM lane exists; applied where it helps (see §5) |
| trtllm, gemm | 42 | correct — lanes agree to 0.4-3.6% |
| trtllm, attention | 12 | sliding-window and L40S/A100 gaps; see the absences in §5 |
| trtllm, moe | 12 | the vLLM lane differs by architecture; untried |
| sglang / trtllm, recurrent | 4 | vLLM `kda` and `linear_attention` data exists; untried |
| sglang, gemm / moe | 4 | A100's only lane for those families |
| **vllm** | **82** | decode and prefill attention, all five served parts, plus A100 |

**One collection per part, per family.** Globbing collections mixes engine versions and
silently changes the row set — it once produced 66,148 rows where the committed fit used
40,367. A difference between two parts should be the silicon, not a software release.

## 3. The three-level data separation

| source | role | why that role |
|---|---|---|
| AISimulate per-operator sweeps | **fitting** | per-kernel, lane-labelled, and the only data that isolates one primitive |
| HF FPM whole-forward (Apache-2.0) | **validation and model selection** | one synchronized forward pass at a known batch and KV composition — the quantity the kernel composes, with no scheduler |
| InferenceX measured rows | **evaluation only** | end-to-end with a scheduler and a client; never enters a fit or a selection |
| InferenceX remaining rows | reporting only | their engine settings are not stated per run |

`scripts/select_overlap_band.py` enforces the last two by refusing any path under an
`inferencex` or `semianalysis` directory, and a new fitter should import that guard rather
than restate it.

FPM is entirely vLLM — all 85,484 rows carry `backend: vllm` — and splits as 12,522 pure
decode, 5,226 pure prefill, 67,736 mixed prefill+decode.

**The constraint that governs everything else: a better fit to a component benchmark can
be a worse model.** Five independently verified corrections each improved a per-primitive
or per-step measurement and made the end-to-end score worse, because the kernel's accuracy
rests on partially cancelling errors. So **no per-primitive fit ships without an
end-to-end check**, and a correction whose offsetting term is unidentified waits.

## 4. Train, validate, evaluate

Fitting splits are taken along a dimension generalization actually has to cross, not at
random — a random row split puts near-duplicate shapes on both sides and reports a
generalization that was never tested.

* **GEMM shape ramp**: whole `k` VALUES held out, so a test shape has a reduction depth
  the fit never saw. Train and test agree to within 0.04x on every part and dtype.
* **FPM topology**: MiniMax-M2.7 on h200_sxm spans six topologies (`dep/tp1/ep2`,
  `dep/tp1/ep4`, `pure_tp/tp2/ep1`, `pure_tp/tp4/ep1`, `tep/tp2/ep2`, `tep/tp4/ep4`), so a
  parallelism-dependent term can be validated with model and chip fixed.
* **FPM regime**: fit on pure-prefill and pure-decode rows, validate on the 67,736 mixed
  rows, whose batch composition varies independently of concurrency.

**A confound to state rather than hide**: in FPM, model and system are perfectly
collinear — DeepSeek-V4-Pro only on gb300, GLM-5.2 only on b200_sxm, MiniMax-M2.7 only on
h200_sxm. A model holdout is also a chip holdout, so a change across it cannot be
attributed to either.

Errors are reported for train, validation AND evaluation. A fit quoted only on its
training data is not a result.

## 5. Per-part provenance

Coefficient counts by citation lane, as committed:

| chip | total | nccl | vllm | trtllm | sglang | other |
|---|---|---|---|---|---|---|
| h200 | 100 | 63 @ 2.29.2 | 13 @ 0.24.0/0.25.0 | 10 @ 1.3.0rc20 | — | 14 |
| h100 | 104 | 63 @ 2.29.2 | 13 @ 0.24.0/0.25.0 | 12 @ 1.3.0rc20 | 2 @ 0.5.16 | 14 |
| b200 | 102 | 63 @ 2.29.2 | 13 @ 0.24.0/0.25.0 | 12 @ 1.3.0rc20 | — | 14 |
| b300 | 102 | 63 @ 2.29.2 | 13 @ 0.24.0/0.25.0 | 12 @ 1.3.0rc20 | — | 14 |
| gb200-nvl72 | 78 | 48 @ 2.29.2 | 4 @ 0.25.0 | 12 @ 1.3.0rc20 | — | 14 |
| l40s | 100 | 72 @ 2.27.3 | 4 @ 0.24.0 | 10 @ 1.3.0rc20 | — | 14 |
| a100-sxm | 94 | 63 @ 2.27.3 | 13 @ 0.14.0 | — | 4 @ 0.5.10 | 14 |
| a100-80 | 174 | 126 @ 2.27.3 | 22 @ 0.14.0 | — | 4 @ 0.5.10 | 22 |

`other` is the non-AISimulate set: host overheads, HBM derate, communicator sizes.

### Version pins, stated rather than averaged away

A100's vLLM collection is **0.14.0** and L40S's is **0.24.0**, against 0.25.0 on the
Hopper and Blackwell parts. Those are the newest vLLM collections AISimulate ships for
those SKUs. Pinning one collection per part keeps a difference between parts attributable
to silicon; it also means A100's coefficients describe an older engine, and that is a
property of the data rather than a choice.

### A100-80 is an alias

`a100-80` and `a100-sxm` are the same silicon — identical `TFlopsPeak` 312,
`BwPeakTBs` 2.039, `MemoryGiB` 80, `SMCount` 108 — and the catalog's own chip file says
"Alias for A100-SXM". Both names appear in `hardware_config.json`, so a BLIS run can
request either.

Aliasing is done two ways, deliberately:

* Where the coefficient NAME has no chip suffix, `a100-80` is added to `a100-sxm`'s
  **scope**, so one measurement serves both names and a refit cannot update one and miss
  the other. The kernel matches scope by membership (`resolve.Scope.Admits`).
* The collective names DO carry a chip suffix, because the kernel builds the key as
  `collective_<kind>_<op>_<dtype>_<rank>rank_<chip>` with `-` replaced by `_`. A
  membership scope cannot satisfy that, so 72 `*_a100_80` entries carry A100-SXM's values
  with the alias stated in each rationale. **Re-fit both together.**

### Deliberate absences

Absent is not the same as missing:

* `gemm_*_fp8` and `gemm_*_fp8_block` on A100 — the part has `TFlopsFP8: 0`, so
  `dtypeFit` returns the `bf16` suffix. A fraction of a rate the silicon does not have
  would be worse than nothing.
* `attention_decode_*_swa` on A100 — vLLM 0.14.0's schema has no `window_size` column;
  the sliding-window sweep postdates it. The kernel falls back to the unsuffixed pair.
* int8 all-reduce on A100 stays on NCCL — `allreduce_dtype` in the custom_allreduce sweep
  is bfloat16 across all 138 rows, so there is no vLLM int8 measurement. Unmeasured is not
  unchanged.
* L40S sliding-window decode stays on TRT-LLM — the vLLM-lane fit is WORSE (2.110x
  against the committed entry's), so it is reported and not applied.
* L40S all-reduce stays on NCCL — the vLLM-graph floors agree with NCCL to within 13%,
  because L40S has no NVLink (PCIe Gen4, 32 GB/s) and `custom_allreduce` needs peer-to-peer
  NVLink to help. A100-SXM has NVLink 3 at 300 GB/s and its floors drop 2.1-3.8x, which is
  the same mechanism that moved the Hopper and Blackwell parts.

## 6. Reproducing any value

Every fitted coefficient names its script in the rationale. The fitters are pure: given
the same collection they print the same numbers, and the validator re-runs them.

    scripts/fit_attention.py <data>/<sku>/attention/<lane>          --chip <chip>
    scripts/fit_attention_prefill.py --collection <lane> <sku>:<chip>
    scripts/fit_attention_by_kind.py --collection <lane> --sku <sku> --chip <chip>
    scripts/fit_collectives_vllm.py <data>/<sku>/comm/vllm/<version>
    scripts/fit_collectives.py <data>/<sku>/comm/nccl/<version>
    scripts/fit_gemm_envelope.py <data>/<sku>/gemm/<lane>
    scripts/fit_gemm_shape_ramp.py [--emit] <sku>:<chip>
    scripts/probe_cudagraph_dispatch.py --settings S --corpus C

Two traps these scripts exist to avoid, both of which produced wrong answers by hand:

* **`latency` is MILLISECONDS** in every AISimulate parquet. Reading it as seconds gives
  2,070 TB/s on a 4.8 TB/s part, or a 143,314 GB/s collective peak. The error announces
  itself if you check the number against the datasheet, and not otherwise.
* **FPM's `total_kv_read_tokens` is summed OVER THE BATCH**, while the kernel takes a
  per-sequence context. Joining them directly is a 256x KV error at batch 256; the join is
  on `(batch, batch*ctx == kv_total)`.

## 7. Validation gates

`pytest validator/` — 188 passing, 8 skipped. The ones that matter:

* **independent re-derivation** — re-runs the fitter and compares to the registry, so a
  value edited without its citation fails.
* **single-sourced citations** — every source in the AISimulate sets must cite AISimulate.
* **lane pinning** — the all-reduce test pins the `vllm_graph` lane specifically, because
  eager is SLOWER than NCCL and accepting either would accept a 3.4x mispricing.
* **skips are explicit** — a gate that cannot run says why. The three-factor GEMM ramp's
  gate skips while the registry carries the one-factor form, and activates the moment the
  coefficients land.
