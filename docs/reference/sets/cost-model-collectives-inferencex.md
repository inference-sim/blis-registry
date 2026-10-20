---
hide:
  - navigation
  - toc
---

# cost-model-collectives-inferencex

DeepEP low-latency dispatch plus combine, priced as one MoE layer's all-to-all under
CUDA-graph replay. vLLM v0.31.0 captures the low-latency kernels in a decode step's graph,
so a captured step takes this pair:

> latency = `floor` + tokens · hidden · 2 · top_k / `rate`

with per-rank tokens. The eager pair (`moe_a2a_*_deepep_low_latency_8rank_<part>` in
[cost-model-collectives](cost-model-collectives.md)) prices a step with no graph.

**Where the values come from.** InferenceX CollectiveX, backend deepep-v2 in low-latency
mode (DeepEP's legacy Buffer kernels), bf16, EP8 on one node, the roundtrip p50 per
tokens-per-rank point. The fit uses 1/4/16/64/256 tokens and holds out 2/8/32/128. Each
entry states its runs and held-out error.

**Why a separate set.** cost-model-collectives comes from a single source, NVIDIA
AISimulate. A set that mixed the two methodologies would confound comparisons across
parts. A scenario lists this set next to cost-model-collectives to take these prices.
Without it, the kernel prices as before.

<!-- registry:set cost-model-collectives-inferencex -->

**Regenerate:** `fit_deepep_ll_graph.py`; `--check` is the drift gate.
