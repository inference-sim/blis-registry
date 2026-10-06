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

### 6.0 What is reproducible, and what is not

Three kinds of value live in this registry, and only the first is reproducible
from a dataset. Counted from the coefficient files themselves:

| `method` | entries | reproducible by | 
|---|---|---|
| `measured` | 654 | a named fitter in `scripts/`, re-run by the validator |
| `vendor_spec` | 64 | a datasheet citation |
| `assumed` | 7 | nothing — each states its reasoning and what would replace it |

All seven `assumed` entries are host overheads in
`cost-model-host-overheads.yaml`: `host_admission_per_request`,
`host_admission_per_token`, `host_output_token`, `host_completion`,
`host_launch_eager_per_layer`, `host_replay_graph_per_step` and
`host_launch_per_kernel`. No dataset in this project measures client-observed
host time: AISimulate times kernels, FPM times one synchronized forward pass
(all 42 of its columns were enumerated — there is no `ttft`, `e2e`, `client`,
`queue` or `host` column), and InferenceX is evaluation-only.

So a reader can re-derive 718 of 725 entries and must take 7 on the reasoning
written into them. Each of those seven carries the measurement that would
replace it; `host_admission_per_request` names the exact experiment, a
client-observed TTFT measurement at 1-token and 1,024-token prompts against a
running vLLM server at concurrency 1 with a warm cache, whose intercept is the
coefficient with no evaluation data involved.

This boundary is stated rather than smoothed over because the largest single
accuracy movement in this work — TTFT mape 52.94% to 31.47% on the measured
tier — comes from one of the seven.

### 6.1 Reproducing the end-to-end evaluation tables

Reproducing a coefficient needs only a fitter and its collection. Reproducing an
accuracy FIGURE needs four things pinned, and each of them has silently produced a
wrong number during this work:

| what | why it matters |
|---|---|
| the scorer checkout | `cmd/metricscore` is on inference-sim's `kernel-exclusive` branch only — not `main`, not `modeling`. A checkout of either fails with `stat cmd/metricscore: directory not found` |
| `-registry` | defaults to the sibling `blis-registry` checkout, so scoring a branch without passing it explicitly scores a different registry |
| `-catalog` | the figures move with the catalog revision; a comparison across catalog revisions once produced an apparent 0.03pp "effect" that was two catalog commits |
| `-config-tier`, `-framework`, `-length-range-ratio` | each changes every figure. On one kernel revision the same command with and without `-length-range-ratio 1.0` gives 11.62/12.05/15.23/52.94 against 11.22/12.19/16.22/57.13 |

`scripts/compare_registries.py` pins all four by construction and prints the
revisions it used. Beyond comparing two registries it records and checks absolute
figures:

    # write the current figures, with revisions and flags, to docs/evaluation-baseline.json
    python scripts/compare_registries.py --record --scorer /path/to/kernel-exclusive/checkout

    # fail if any figure, or any n, has moved since
    python scripts/compare_registries.py --check  --scorer /path/to/kernel-exclusive/checkout

`docs/evaluation-baseline.json` is the machine-checkable record. It holds the four
headline figures and their point counts per tier, plus the framework, the
length-range-ratio and the scorer and catalog revisions, because a figure is a
claim about that whole triple rather than about the registry alone. `--check`
compares the revisions too and says when they have moved, so a drift is
attributable rather than merely visible.

A figure quoted without its flags cannot be checked, and this is the reason every
table in the companion documents carries its invocation.

### 6.1.1 The scorer's kernel is pinned, and the pin is not this registry's concern

One asymmetry in the reproduce recipe is worth stating because it looks like an
oversight and is not. `inference-sim`'s scoring branch clones the catalog and the
registry with `-b modeling`, because both are read at RUN TIME by path and the
scorer takes `-catalog` and `-registry` flags. It clones `blis-latency-kernel`
WITHOUT a branch flag, because the kernel is a COMPILED dependency pinned by
module version in `go.mod`, and its default branch is `main`.

The consequence for a reader: changing this registry changes the figures
immediately, while changing the kernel does not until its pin is advanced. At the
time of writing the scoring branch pins kernel `0bbcb97`, which is an ancestor of
the kernel's `modeling` branch by 17 commits. A figure recorded by
`compare_registries.py --record` therefore states the scorer revision alongside
the registry, and `--check` reports when it has moved, because a figure is a claim
about the pair.

A local `replace` directive pointing at on-disk kernel checkouts is how the
scoring branch's own plan describes linking the two, and it is explicitly
worktree-only: committing one would break the build for anyone following the
clone recipe.

### 6.2 Which tier to weigh

`inferencex_engine_settings.json` records **68 sweeps with a captured engine-args
log and 136 without**, each of the latter listed in its `incomplete` array with
`reason: "no engine-args log"`. For those 136 the engine configuration is resolved
from vLLM's defaults rather than observed, so an error there is as likely to be a
wrong assumed configuration as a wrong model.

The measured tier is the only one where the simulated deployment is ground truth.
It is therefore the tier on which a coefficient change can be attributed to the
model, and the tier to weigh when judging one. The resolved tier answers a weaker
and still useful question — how well a predictor does when it must guess the
configuration too — and mixing the two in one average mixes two kinds of
evidence, which is why the scorer takes `-config-tier` and the baseline records
them separately.

### 6.3 Validation gates

`pytest validator/` — 147 passing, 8 skipped (verified by running it). The ones that matter:

* **independent re-derivation** — re-runs the fitter and compares to the registry, so a
  value edited without its citation fails.
* **single-sourced citations** — every source in the AISimulate sets must cite AISimulate.
* **lane pinning** — the all-reduce test pins the `vllm_graph` lane specifically, because
  eager is SLOWER than NCCL and accepting either would accept a 3.4x mispricing.
* **skips are explicit** — a gate that cannot run says why. The three-factor GEMM ramp's
  gate skips while the registry carries the one-factor form, and activates the moment the
  coefficients land.

## 7. Held-out validation of the lane decisions

§4 states the splitting principle. This section records the folds actually run, the
numbers they produced, and the decisions they support or refuse. Every figure here is
reproduced by the script named beside it; none is a residual on the rows that were fitted.

### 7.1 Sliding-window attention: hold out one window width

`scripts/holdout_attention_swa.py`. The window is the dimension that defines the kind —
`kv_bytes` reads `min(context, window)` positions — so a held-out width is a byte count the
fit never saw. Reported as the median over folds, because the widths are heavily unbalanced
(window 128 carries 82% of h200's rows, so that fold trains on an eighth of the sweep while
the others hold out 3-6%); a mean over folds this uneven would describe none of them.

| part | vLLM collection | vLLM median held-out | TRT-LLM median held-out | shared folds won by vLLM | decision |
|---|---|---|---|---|---|
| h200 | vllm/0.25.0 | **1.457x** | 1.754x | 3/3 | relane |
| h100 | vllm/0.25.0 | **1.433x** | 1.787x | 2/2 | relane |
| b200 | vllm/0.25.0 | **1.344x** | 1.475x | 2/3 | relane |
| b300 | vllm/0.25.0 | **1.317x** | 1.498x | 2/2 | relane |
| gb200-nvl72 | vllm/0.25.0 | **1.276x** | 1.358x | 3/3 | relane |
| l40s | vllm/0.24.0 | 2.073x | **1.681x** | 0/2 | **keep TRT-LLM** |

Twelve of thirteen shared folds favour vLLM on the five relaned parts; the single exception
is b200 at window 2048 (1.259x against 1.244x). The vLLM sweep also covers five widths
where TRT-LLM covers two or three, so the TRT-LLM fit was never tested at 512 or 1024 on
any part.

L40S is the documented exception and now has held-out evidence rather than a residual: its
vLLM fit is worse on BOTH folds. Note that L40S has no `vllm/0.25.0` attention collection
at all — `vllm/0.24.0` is its newest — so the comparison uses that, and an earlier reading
of "no vLLM data" for this part was wrong.

### 7.2 KDA: hold out the head-count geometry

`scripts/fit_recurrent.py --group-by kernel+heads`. vLLM's KDA sweep varies `num_k_heads`
over 12/24/48/96 where SGLang's holds it at 12. Pooling those four geometries into one fit
gives geo-err **1.733x**; fitted per geometry the same rows give 1.169x to 1.196x. The
pooled figure is an artefact of the pooling, not a property of the lane:

| lane | num_k_heads | n | floor | rate | geo-err |
|---|---|---|---|---|---|
| sglang/0.5.16 (committed) | 12 | 11 | 5.1 us | 1.50 tok/us | **1.160x** |
| vllm/0.1.dev19262 | 12 | 11 | 5.4 us | 1.75 tok/us | **1.169x** |
| vllm/0.1.dev19262 | 24 | 11 | 5.3 us | 1.00 tok/us | 1.196x |
| vllm/0.1.dev19262 | 48 | 11 | 3.6 us | 0.50 tok/us | 1.191x |
| vllm/0.1.dev19262 | 96 | 11 | 3.6 us | 0.25 tok/us | 1.188x |

At the matching geometry — `num_k_heads: 12`, which is what SGLang measures — the two lanes
are **equivalent** (1.160x against 1.169x, a difference of 0.009). An earlier claim that the
vLLM lane fits 1.49x worse was wrong, and wrong because of the pooling above.

This does not by itself relane KDA. The vLLM collection is a dev build
(`0.1.dev19262+gb6bbf29dd`), Kimi-K3 appears in no InferenceX tier and in no FPM artifact
with a KDA node, so the change is unevaluable end to end and the decision rests on fit
quality alone, where the two lanes tie.

### 7.3 GEMM ramp: hold out whole K values, and the largest M values

`scripts/holdout_gemm_ramp.py`. Two folds, and the second one found a defect in the FORM
rather than in the lane.

**The ramp is the wrong shape above M ~ 4096.** `eff(M) = eps_max * M / (M + M_half)` is
monotone and saturating, but the measured envelope PEAKS mid-sweep and then declines — on
h200 bfloat16, 0.956 at M=3329 falling to 0.875 at M=32768 on the vLLM lane, and 0.870 at
M=4097 falling to 0.839 on TRT-LLM. Both lanes show it, so it is a property of the kernel
and not of either engine. Holding out the three largest M values and extrapolating:

| held-out M | predicted | measured | error |
|---|---|---|---|
| 8192 | 0.987 | 0.930 | +0.057 |
| 16384 | 0.993 | 0.911 | +0.082 |
| 32768 | 0.997 | 0.875 | **+0.121** |

So `eps_max` is fitted to an asymptote the kernel never reaches, and a fit that has not seen
large M over-predicts there by up to 12 points of efficiency.

**Why this is recorded and not fixed.** A decode step's M is the token count in the batch,
and the corpus runs M = 1 to 2048 with 97.2% of points (1,036 of 1,066) at M <= 128. The decline begins above
M ~ 4096, outside the range any evaluated deployment reaches. Scoring the same fit inside
and outside the operating range:

| lane | range | n | rms | mean signed |
|---|---|---|---|---|
| vllm/0.25.0 | M <= 128 (97% of corpus steps) | 30 | 0.0585 | +0.0509 |
| vllm/0.25.0 | M <= 2048 (all corpus steps) | 52 | 0.0745 | +0.0009 |
| trtllm/1.3.0rc20 | M <= 128 | 30 | **0.0474** | **+0.0393** |
| trtllm/1.3.0rc20 | M <= 2048 | 52 | **0.0538** | +0.0040 |

**This refuses a vLLM-lane GEMM relane.** In the range that matters TRT-LLM fits better on
both measures, which is consistent with §2's finding that the lanes agree to 0.4-3.6% on
shared shapes — the same cuBLAS/CUTLASS kernels underneath — so the larger TRT-LLM sweep
wins on determination rather than on engine. The lane rule is "fit the engine you predict"
only where the engine changes the kernel, and for GEMM it does not.

Both lanes over-predict efficiency at small M (+0.039 and +0.051), which UNDER-predicts
decode latency on the shapes that dominate the corpus. That is a real, quantified direction
for future work, and it is a form problem: the fix is a non-monotone ramp or a small-M
correction term, not a different collection.

### 7.4 What FPM could and could not adjudicate

FPM is the validation source in §3, but its reach is narrower than the coefficient set:

* **SWA**: unusable. No FPM model carries a sliding-window node. Four of the six models in
  the FPM catalog have a graph in blis-catalog — MiniMax-M2.7 (`gqa`), GLM-5.2
  (`gqa`, `sparse_mla`), DeepSeek-V4-Pro (`gqa`, `mla`, `sparse_mla`) and Kimi-K3
  (`kda`, `mla`) — and none declares `kind: swa`; the other two
  (DeepSeek-V4.1-Flash, DeepSeek-V4-Flash-0731) have no catalog graph, so the kernel cannot
  price them either way. The relane in §7.1 therefore rests on held-out AISimulate folds
  plus provenance, with no FPM arbiter, and that is weaker evidence than a GEMM change
  would carry.
* **KDA**: unusable for the same reason; Kimi-K3's FPM artifacts carry no KDA-node forward
  pass in the local snapshot.
* **GEMM**: usable in principle — each of the four graphed FPM models carries dense `GEMM`
  nodes (3 on MiniMax-M2.7, 9 on GLM-5.2, 18 on DeepSeek-V4-Pro, 17 on Kimi-K3) — but the
  AISimulate K and M folds above already refuse the relane on the shapes the corpus runs,
  so no FPM run was needed to decide it. An FPM fold remains the right arbiter for a change
  to the ramp's FORM, which §7.3 identifies and does not attempt.

## 8. Lane provenance: every coefficient family, and every remaining borrow

The lane rule in §2 is applied per operator. This section is the resulting census, so a
reader can tell at a glance which engine's measurements each coefficient rests on, and
which four are borrowed because no alternative exists.

### 8.1 The census

| lane | n | share | what it is |
|---|---|---|---|
| `nccl` | 498 | 68.8% | all_gather, reduce_scatter, all_to_all, and the int8 all-reduce |
| `vllm` | 152 | 21.0% | attention, GEMM, MoE imbalance, KDA, and the fp16 all-reduce |
| descriptor | 64 | 8.8% | NVIDIA's own `systems/<sku>.yaml` — HBM derate, PCIe, p2p, NCCL buffers |
| none | 6 | 0.8% | host overheads, all `method: assumed` |
| `trtllm` | 2 | 0.3% | mamba2 — **borrow** |
| `sglang` | 2 | 0.3% | A100 MoE imbalance — **borrow** |

**NCCL is not a borrow.** It is vLLM's own path for every collective except all-reduce:
`CustomAllreduce` implements all-reduce only, and `should_custom_all_gather` /
`should_custom_reduce_scatter` gate and fall through. AISimulate ships no vLLM data for
those operators because there is nothing to measure. Of the 654 data-fitted coefficients,
**650 (99.4%) rest on vLLM's actual execution path** — 152 vLLM-lane plus 498 NCCL. The
remaining four are the borrows in §8.3.

### 8.2 Per-family detail

| set | family | n | lanes |
|---|---|---|---|
| attention | `attention_decode_{floor,rate}` | 14 | vllm 0.25.0 ×10, 0.24.0 ×2, 0.14.0 ×2 |
| attention | `attention_decode_*_swa` | 12 | vllm 0.25.0 ×10, 0.24.0 ×2 |
| attention | `attention_prefill_{floor,work_scale}` | 14 | vllm 0.25.0 ×10, 0.24.0 ×2, 0.14.0 ×2 |
| collectives | all-reduce fp16 (floor, peak, transition) | 69 | **vllm** 0.24.0 ×36, 0.14.0 ×18, nccl ×15 |
| collectives | every other collective | 429 | nccl 2.29.2 / 2.27.3 |
| primitives | `gemm_{eps_max,m_half}` | 44 | vllm 0.27.1 ×24, 0.25.0 ×12, 0.24.0 ×6, 0.14.0 ×2 |
| primitives | `moe_routing_imbalance_{median,p90}` | 14 | vllm ×12, **sglang 0.5.10 ×2 (A100)** |
| primitives | descriptors | 64 | AISimulate `systems/<sku>.yaml` |
| recurrent | `recurrent_decode_*_kda` | 2 | vllm 0.1.dev19262 |
| recurrent | `recurrent_decode_*_mamba2` | 2 | **trtllm 1.3.0rc20** |
| host-overheads | all | 6 | none — `method: assumed` |

Where a part has no collection at the newest version, it uses the newest it has: L40S's
attention and GEMM come from `vllm/0.24.0` because 0.25.0 does not cover it, and A100's
from `vllm/0.14.0`. The citation on each entry names the collection it was fitted on.

### 8.3 The four borrows, and why each is unavoidable

**mamba2 (2 entries, trtllm/1.3.0rc20).** `mamba2_perf.parquet` exists on the TRT-LLM lane
and nowhere else — checked across all eight SKUs in the tree. The vLLM `linear_attention`
collections carry `gdn_perf.parquet` only. A strict vLLM-only registry could not price
Nemotron-3 hybrids at all. The entries remain a documented LOWER BOUND regardless of lane:
the sweep has only `causal_conv1d_fn` and `causal_conv1d_update`, so the selective-scan
kernel that does the rest of a Mamba2 layer is in no collection.

**A100 MoE imbalance (2 entries, sglang/0.5.10).** A100's only vLLM MoE collection
(`vllm/0.14.0`) contains `power_law_1.01` and `power_law_1.2` rows and **no `balanced`
rows at all**. The coefficient is a ratio of skewed to balanced latency at identical
shape, so with no balanced measurement there is no ratio to take. SGLang's sweep carries
all three distributions. A100's GEMM entries DID move to the vLLM lane — that sweep is
usable — so this is the narrowest possible borrow.

### 8.4 What validates what

The three-level separation in §3 assigns FPM the validation role, but FPM's reach is
narrower than the coefficient set and it is worth being exact about where it applies.

FPM's 59 records are 41 vLLM and 18 SGLang, across five GPU families: B200 (17),
GB300 (21), H200 (13), GB200 (4), B300_SXM (4). So:

* **GEMM and MoE**: validatable. Every FPM model carries dense `GEMM` nodes, and the MoE
  models carry `GroupedGEMM`, on chips the catalog prices.
* **Attention, full and windowed**: GQA and MLA are validatable; **sliding window is not**
  — no FPM model declares a `kind: swa` node. The SWA relane therefore rests on held-out
  AISimulate folds (§7.1) plus lane correctness, with no FPM arbiter.
* **KDA**: the data EXISTS but the catalog cannot reach it. Four Kimi-K3 artifacts are
  all vLLM and all on **GB300**, two of them on `vllm/0.1.dev19262` — the same collection
  the kernel fit used, which would be the cleanest possible pairing. The catalog has no
  `gb300` chip, so no scenario can be built. Tracked as blis-catalog#17 and
  blis-registry#25; until then KDA is the one family shipping on fit quality and vLLM
  source reading alone.
* **mamba2**: no FPM artifact carries a Mamba2 forward pass, and no InferenceX tier
  carries Nemotron-3. Unvalidatable from either direction.
* **A100 and L40S**: absent from FPM and from InferenceX entirely. Their coefficients are
  fitted and shipped but carry no end-to-end check, which is stated rather than implied.

## 9. The FPM mixed rows, and what they found

§8.4 noted that FPM's reach is narrower than the coefficient set. It was also being
under-used: 79.2% of FPM is MIXED prefill+decode rows (67,736 of 85,484), and none had
ever been scored. `scripts/score_fpm_mixed.py` closes that, and the result is the largest
single finding in this work.

### 9.1 What the coordinates mean, verified in AISimulate's own source

FPM's coordinate system is `iteration_totals_balanced_v1`
(`crates/core/src/perfmodel/perf_database/fpm_forward.rs:59`), and the totals are
**per-rank iteration totals**, not global ones. AISimulate builds a query as

```rust
// crates/core/src/perfmodel/operators/fpm_forward.rs:203-218
FpmPhase::Prefill => vec![b, b * s as f64, b * prefix as f64],
FpmPhase::Decode   => vec![b, b / w * s as f64],
```

so `total_prefill_tokens = batch × per_request_prefill` and
`total_kv_read_tokens = batch × per_request_context`. Dividing by `batch_size` to recover
the per-request shape the kernel takes is therefore the correct inverse, and is what
`select_overlap_band.py` already documents. The source warns that the round trip adds
integer-division rounding, which is why the scorer medians per grid point rather than
per row.

For a data-parallel cell the recorded latency is the **maximum across DP ranks**
(`collector/fpm_forward/native_artifact.py:704`,
`expected_wall_time = max(value for _, value in wall_times)`), not a mean — so a DP cell's
measurement is gated by its slowest rank.

### 9.2 Six cells scored, and the error is strongly topology-dependent

NoOverlap edge, all mixed grid points, against the committed registry:

| cell | parallelism | n | mean abs | signed | over |
|---|---|---|---|---|---|
| m27 pure_tp4 | tp4, ep1 | 5,986 | 15.63% | **−13.63%** | 13.0% |
| m27 tep2 | tp2, ep2 | 5,782 | 20.13% | **−14.86%** | 20.8% |
| m27 tep4 | tp4, ep4 | 5,986 | 25.58% | **−25.24%** | 3.1% |
| m27 dep2 | tp1, dp2, ep2 | 5,383 | 34.58% | **−33.96%** | 2.6% |
| m27 dep4 | tp1, dp4, ep4 | 5,918 | 47.14% | **−46.67%** | 1.6% |
| glm-5.2 tep8 | tp8, ep8 | 5,910 | 49.79% | **−45.04%** | 10.5% |

Every cell under-predicts, and the error spans 13% to 47% — a 3.4× range across topologies
of the same model on the same chip. That is not scatter; it is a missing term that scales
with parallelism.

The error is also **flat across the prefill share of the batch** on the tep4 cell
(−25.05% below 25% prefill, −22.57% at 25–75%, −28.40% above 75%), so it is not a
prefill-specific or a decode-specific defect. The whole forward pass is under-priced.

### 9.3 The mechanism: attention-DP funnels every rank's tokens into the experts

With attention data parallelism, all DP ranks' tokens are concatenated before expert
routing, so the MoE grouped GEMM sees `dp × tokens` rather than one rank's share. Three
independent sources agree:

* **vLLM**: `fused_moe/routed_experts_capturer.py:115-117` — "``n == total`` (naive
  dispatch): all DP ranks' tokens are **concatenated before routing**". `naive` and
  `allgather_reducescatter` are that path, and `allgather_reducescatter` is vLLM's default
  (`config/parallel.py:195`).
* **AISimulate**: `crates/core/src/perfmodel/operators/moe.rs:287` —
  `let num_tokens = num_tokens.saturating_mul(self.attention_dp_size.max(1));` with the
  comment "Attention-dp scales up the total input tokens (all dp ranks all-gather into one
  shared expert pool)".
* **blis-latency-kernel**: `kernel.go:402` —
  `routedPerRank := tokensF * float64(l.TopK) * k.localExpertShare`. **No DP factor.**
  `DP` reaches exactly one decision, `SequenceParallelMoE`
  (`internal/resolve/layout.go:106`), and that rule requires `tp > 1 && dp > 1`
  (`blis-schemas rules/rules.go:196`) — so on the two `dep` cells, which carry `tp=1`,
  DP influences no pricing term at all.

The two `dep` cells are the two worst MiniMax results and they order by `dp` (−33.96% at
dp=2, −46.67% at dp=4), which is the direction and roughly the magnitude a missing `×dp`
on the routed term predicts.

**What this does NOT explain.** `glm-5.2 tep8` is dp=1 and still −45.04%, so a missing DP
factor cannot be the whole account. GLM-5.2 differs from MiniMax-M2.7 in more than
topology — it carries `sparse_mla` attention and a separate indexer GEMM — so that cell
needs its own diagnosis rather than being folded into this one. Stating the limit is the
point: one mechanism explains the MiniMax ordering, and a second, unidentified term is
still live on GLM.

### 9.4 A committed figure this corrected

`docs/band-selection.md` reports NoOverlap's signed mean as −3.44% over 219 points and
calls it "nearly unbiased". Scoring every pure-decode grid point in the m27 tep4 artifact
rather than the 45 the old band sweep reached gives **−22.93%** (1,500 points), or
**−19.27%** restricted to batches within the scenario's declared `max_num_seqs` of 256
(1,048 points). FPM carries 1,557 distinct decode grid points for that cell, so the
committed figure rested on under 3% of the evidence available for it, and the deficit
grows with batch (−12.8% at batch 8 to −31.3% at 512) in a way a 45-point slice can miss.

The band CHOICE is unaffected — NoOverlap beats Overlap on every slice measured here, by
about 7pp on the mixed rows. The claim of near-unbiasedness is withdrawn.
