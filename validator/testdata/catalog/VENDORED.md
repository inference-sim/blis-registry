# Vendored chip descriptors

The nine `hardware/*.yaml` files the committed coefficients scope to, copied from
[blis-catalog] at `747a2213e13030ae675f9872c3e2a76b6b770d50` (2026-10-07) — the same commit
blis-latency-kernel vendors, so the two repositories check against one revision.

[blis-catalog]: https://github.com/inference-sim/blis-catalog

```
a100-80  a100-sxm  b200  b300  gb200-nvl72  gb300  h100  h200  l40s
```

Copied verbatim. 40 KB, against the 16 M the catalog occupies: only the `hardware/`
descriptors are here, because that is all `validator/test_coefficient_properties.py`
opens — it reads `BwPeakTBs` and the memory figures to check that a fitted rate is
physically possible.

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
