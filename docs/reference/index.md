# Coefficient sets

Everything on the pages in this section is generated from the committed files when the
site is built, at revision <!-- registry:ref -->. Nothing is typed in by hand, so a count
here cannot disagree with the data.

## The sets

<!-- registry:census -->

*Entries* counts every (name, scope) pair. *Coefficients* counts distinct quantities,
after folding the part out of names that carry it, such as
`collective_floor_all_reduce_fp16_8rank_h200`. *Parts* is the number of GPU parts with
at least one entry.

## What each set rests on

The share of each set by method, with the number of entries at the right:

<!-- registry:chart -->

## Which sets the kernel reads

A scenario names the sets it loads. The five `cost-model-*` sets for step time are the
ones every committed scenario in blis-latency-kernel lists. Within them, a few families
are committed but not looked up by the kernel. As of blis-latency-kernel
[`d20a227`](https://github.com/inference-sim/blis-latency-kernel/commit/d20a227):

| Set | Not read |
|---|---|
| `cost-model-primitives` | `hbm_constant_latency`, `p2p_latency`, `moe_routing_imbalance_p90`, and every `gemm_*_fp8_block` entry: the kernel's dtype selection resolves to `bf16`, `fp8` or `nvfp4` only |
| `cost-model-memory` | the whole set. It is committed ahead of its reader, [blis-latency-kernel#22](https://github.com/inference-sim/blis-latency-kernel/issues/22) |

An unread entry still passes every check, so this table is the place to look before
assuming a change to one will move an estimate.

## Reading a set page

Each set page shows the set as a matrix: one row per coefficient, one column per part.
A glyph before each value gives its method, and its shape alone identifies it:

| Glyph | Method |
|---|---|
| <span class="reg-m reg-m-measured">■</span> | `measured` |
| <span class="reg-m reg-m-vendor_spec">◆</span> | `vendor_spec` |
| <span class="reg-m reg-m-assumed">○</span> | `assumed` |
| <span class="reg-m reg-m-not_charged">–</span> | `not_charged` |

Every value is a link to its entry in the file, where the sources, rationale and scope
are recorded. A dot (·) marks a part the set has no entry for. Whether that is an
omission or a deliberate absence is stated in the set's header and on its page.
