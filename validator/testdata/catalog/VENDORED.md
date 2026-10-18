# Vendored chip descriptors

The nine `hardware/*.yaml` files the committed coefficients scope to, copied from
[blis-catalog] at release `0.2.2`, `3ed02ff75e6bf505f64cf7a54aa5e3bd626f85a5` (2026-10-13).
blis-latency-kernel vendors the same descriptors and should be bumped to this release too,
so the two repositories check against one revision. The previous pin was `747a2213`
(2026-10-07); `0.2.2` changes one value the registry reads, b300 `BwPeakTBs` 7.75 → 7.7,
and the three b300 rates fitted as a fraction of it (`attention_decode_rate`, `_swa`, `_mla`)
were re-fitted with the bump.

[blis-catalog]: https://github.com/inference-sim/blis-catalog

```
a100-80  a100-sxm  b200  b300  gb200-nvl72  gb300  h100  h200  l40s
```

Copied verbatim, with three model graphs beside them:

```
models/deepseek-v3/graph.yaml   models/kimi-k3/graph.yaml   models/deepseek-v4-pro/graph.yaml
```

and every model's `models/<name>/model.yaml` (34 of them), which name the model and the
Hugging Face repository it was derived from. `scripts/emit_acceptance_profiles.py` reads
them to resolve an InferenceX curve's model to a catalog name, and refuses a curve whose
model is none of them, so the copy must be the catalog's whole model list, not a subset.

The hardware descriptors are what `validator/test_coefficient_properties.py` opens -- it
reads `BwPeakTBs` and the memory figures to check that a fitted rate is physically
possible -- and what every fitter reads its peaks from. The graphs are the ones the fitters
resolve a latent or sparse geometry from (`fit_attention_mla.py`: deepseek-v3, kimi-k3;
`fit_attention_sparse_mla.py`: deepseek-v4-pro). With them here, every writer and fitter in
`scripts/` defaults to this directory, so `--check` gives the same answer on every machine
whatever branch a local catalog checkout is on; `BLIS_CATALOG` still overrides it.
Two exceptions: `compare_registries.py`, the end-to-end scorer, needs a whole catalog; and
`emit_primitives.py` takes its catalog as a required argument when it renders (its
`--check`, the gate, reads no catalog).

## Why vendored

`CATALOG` defaulted to `/Users/sri/Documents/Projects/blis-catalog`, a path that exists on
one machine. In CI every `chip_facts()` call returned `None`, so:

| test | behaviour in CI before |
|---|---|
| `test_no_bandwidth_rate_exceeds_its_parts_datasheet_peak` | examined **0** rate/chip pairs |
| `test_every_bandwidth_rate_is_a_plausible_fraction_of_peak` | examined **0** pairs |
| `test_hardware_independent_sets_cover_every_catalog_part` | skipped outright |

The first of those is what caught it: its own `assert checked >= 10` turned a vacuous pass
into a visible failure, which is the only reason this was found rather than shipped. The
other two had the same blind spot and said nothing — a physical bound that does not run is
worse than no bound, because it reports the invariant as protected.

That makes this the same defect class as the home-directory paths removed from
`inference-sim` (`sim/kernelmodel/roots.go`) and from `blis-latency-kernel`
(`testdata/VENDORED.md`): a default nobody else can resolve, failing quietly.

## Overrides

`BLIS_CATALOG` still redirects, so a working copy can be checked against a live catalog:

```
BLIS_CATALOG=../blis-catalog python -m pytest validator/ -q
```

An **empty** `BLIS_CATALOG` falls back to this copy rather than being honoured as a root —
`BLIS_CATALOG= pytest …` is the natural way to ask for the default, and treating the empty
string as a path resolves every chip relative to nothing and lands back in the
vacuous-pass state. All three forms (unset, empty, live catalog) are expected to give the
same 14 passing tests.

## Updating

Re-copy the nine files from an upstream checkout and update the commit above. A chip
rename or a changed `BwPeakTBs` is *supposed* to break the tests that name it — that
coupling is the point, and it is why the copy is pinned to a commit rather than tracking a
branch.

## Kept in step with CI

`.github/workflows/rederive.yml` checks out blis-catalog at this same commit and fails if
its `hardware/`, any of the three vendored model graphs, or the set of `model.yaml` files
differs from this copy, so the property tests and the re-derivation gates always read one
revision.

`devices/storage.yaml` is copied verbatim from blis-catalog `deec066` (the merge of
`5f6ecab`, the CollectiveX-fitted `cpu_dram` tier), so the validator can assert that
`host_link_bandwidth` on h100 is the same fit as the tier it caps.
