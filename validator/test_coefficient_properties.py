"""Properties the committed coefficients must satisfy whatever the sweeps contain.

WHY PROPERTIES RATHER THAN PINNED VALUES. The re-derivation gates
(test_generated_from_aisimulate.py, test_writers_are_idempotent.py) assert that each
committed number is what its fitter produces. That catches drift but says nothing about
whether the number is PHYSICALLY SENSIBLE: a fitter re-run after a unit slip reproduces
its own mistake exactly and both gates stay green.

These tests encode what the cost model's own physics implies, so they hold for any part,
any future sweep, and any refit. Each one below is a defect this project actually shipped
at least once, which is the bar for including it:

  * A rate stated as a fraction of datasheet bandwidth once landed at 2.00 of peak -- a
    physically impossible figure -- because a windowed kernel's byte count was computed
    from the context length rather than the window.
  * A 1000x unit error (ms read as s) produced a peak of 143,314 GB/s on h200 and a
    three-parameter error worse than the two-parameter one.
  * Pooling geometries in a sweep that varies them produced a recurrent rate 6.4x too
    fast (KDA at num_k_heads=12 for a 96-head model) and a GDN rate 1.27x-1.80x off.

They read only the committed files and the catalog, so they run in CI with no AISimulate
tree present. That is deliberate: a property that needs the sweep to check is a
re-derivation, and those live elsewhere.
"""

from __future__ import annotations

import collections
import glob
import os
import re
from pathlib import Path

import pytest
import yaml

REPO = Path(__file__).resolve().parent.parent

# The chip descriptors these properties check against, vendored under
# validator/testdata/catalog so they are present wherever the suite runs.
#
# This default used to be /Users/sri/Documents/Projects/blis-catalog, a path that exists on
# one machine. In CI `chip_facts` returned None for every chip, so
# `test_no_bandwidth_rate_exceeds_its_parts_datasheet_peak` examined ZERO rate/chip pairs
# and only its own `checked >= 10` guard turned that into a failure rather than a
# vacuous pass. The companion fraction-of-peak test had the same blind spot, and the
# hardware-coverage test below skipped outright. A physical bound that does not run is
# worse than no bound: it reports the invariant as protected.
#
# 40 KB for the nine chips the committed coefficients scope to, pinned to blis-catalog
# 747a2213e13030ae675f9872c3e2a76b6b770d50 (2026-10-07) -- the same commit
# blis-latency-kernel vendors. BLIS_CATALOG still overrides, so a working copy can be
# checked against a live catalog; a chip rename there is SUPPOSED to break these tests,
# which is why the copy tracks a commit rather than a branch.
#
# `or` rather than a two-argument `os.environ.get`: an EMPTY BLIS_CATALOG must fall back to
# the vendored copy, not be honoured as a root. `BLIS_CATALOG= pytest ...` is the natural
# way to ask for the default, and get() treats the empty string as a real value -- which
# resolves every chip path relative to nothing, returns None for all of them, and lands
# back in the vacuous-pass state this vendoring exists to prevent.
CATALOG = Path(os.environ.get("BLIS_CATALOG")
               or REPO / "validator" / "testdata" / "catalog")


def committed() -> list[tuple[str, str, dict]]:
    """Every committed entry as (set-name, coefficient-name, body)."""
    out = []
    for path in sorted(glob.glob(str(REPO / "coefficients" / "*.yaml"))):
        doc = yaml.safe_load(Path(path).read_text())
        for e in doc["coefficients"]:
            (name, body), = e.items()
            out.append((os.path.basename(path), name, body))
    return out


ENTRIES = committed()


def chip_facts(chip: str) -> dict | None:
    p = CATALOG / "hardware" / f"{chip}.yaml"
    if not p.is_file():
        return None
    return yaml.safe_load(p.read_text())


# --------------------------------------------------------------------------------------
# Physical bounds
# --------------------------------------------------------------------------------------

def test_no_bandwidth_rate_exceeds_its_parts_datasheet_peak():
    """A measured byte rate cannot beat the memory it reads from.

    `attention_decode_rate*` and the MLA rate are bytes per microsecond, and no kernel
    sustains more than the datasheet HBM figure. An entry above it means the byte count
    the fit divided by was wrong -- which is exactly how an early windowed fit reached
    2.00 of peak.
    """
    checked = 0
    for setname, name, body in ENTRIES:
        if not re.match(r"attention_decode_rate(_|$)", name):
            continue
        for chip in (body.get("scope") or {}).get("hardware", []):
            facts = chip_facts(chip)
            if not facts or "BwPeakTBs" not in facts:
                continue
            peak_bytes_per_us = facts["BwPeakTBs"] * 1e6
            assert body["value"] <= peak_bytes_per_us, (
                f"{setname}: {name} on {chip} is {body['value']:,.0f} B/us, above the "
                f"part's {peak_bytes_per_us:,.0f} B/us datasheet peak. A kernel cannot "
                f"read faster than its memory; the byte count in the fit is wrong."
            )
            checked += 1
    assert checked >= 10, f"only {checked} rate/chip pairs checked; the filter is wrong"


def test_every_bandwidth_rate_is_a_plausible_fraction_of_peak():
    """And not implausibly far below it either.

    Below ~2% of datasheet peak a 'measured bandwidth' is more likely a unit slip than a
    slow kernel. The lowest committed figure is l40s SWA at ~0.25 of peak, so 0.02 is a
    wide floor that still catches a 1000x error.
    """
    for setname, name, body in ENTRIES:
        if not re.match(r"attention_decode_rate(_|$)", name):
            continue
        for chip in (body.get("scope") or {}).get("hardware", []):
            facts = chip_facts(chip)
            if not facts or "BwPeakTBs" not in facts:
                continue
            frac = body["value"] / (facts["BwPeakTBs"] * 1e6)
            assert frac >= 0.02, (
                f"{setname}: {name} on {chip} is {frac:.4f} of datasheet peak. That is "
                f"below any measured kernel in this registry and is the signature of a "
                f"unit error (ms read as s gives 1/1000)."
            )


def test_gemm_efficiency_asymptotes_are_fractions():
    """`gemm_eps_max_*` is a fraction of peak FLOPs: 0 < eps <= 1."""
    checked = 0
    for setname, name, body in ENTRIES:
        if not name.startswith("gemm_eps_max"):
            continue
        assert 0 < body["value"] <= 1.0, (
            f"{setname}: {name} = {body['value']} is not a fraction. The GEMM ramp's "
            f"asymptote is the share of this dtype's peak a large matmul reaches."
        )
        checked += 1
    assert checked >= 20, f"only {checked} eps_max entries checked"


def test_moe_imbalance_is_near_one_and_p90_is_not_below_median():
    """The imbalance multiplier is a ratio of skewed to balanced latency at one shape.

    It cannot be far from 1 -- it is the same work on the same kernel, differing only in
    routing distribution -- and the p90 of a distribution cannot sit below its median.
    The second is a metamorphic check: it relates two entries that must move together.
    """
    med = {}
    p90 = {}
    for _, name, body in ENTRIES:
        for chip in (body.get("scope") or {}).get("hardware", []):
            if name == "moe_routing_imbalance_median":
                med[chip] = body["value"]
            elif name == "moe_routing_imbalance_p90":
                p90[chip] = body["value"]
    assert med, "no imbalance medians found"
    for chip, v in sorted(med.items()):
        assert 0.5 <= v <= 2.0, (
            f"moe_routing_imbalance_median on {chip} is {v}, implausible for a ratio of "
            f"the same kernel at the same shape under two routing distributions."
        )
        if chip in p90:
            assert p90[chip] >= v, (
                f"{chip}: p90 {p90[chip]} is below the median {v}. These are order "
                f"statistics of one paired distribution, so p90 >= median by definition "
                f"-- a violation means the two were computed from different row sets."
            )


# --------------------------------------------------------------------------------------
# Monotonicity across rank width: the collective floors
# --------------------------------------------------------------------------------------

def _collectives():
    pat = re.compile(
        r"^collective_(floor|peak_rate|transition_rate)_"
        r"(all_reduce|all_gather|reduce_scatter|alltoall)_"
        r"(fp16|int8)_(\d+)rank_(.+)$")
    out = collections.defaultdict(dict)
    for _, name, body in ENTRIES:
        m = pat.match(name)
        if m:
            kind, op, dt, w, chip = m.groups()
            out[(chip, op, dt, kind)][int(w)] = body["value"]
    return out


COLLECTIVES = _collectives()


def test_collective_floors_do_not_fall_as_the_group_widens():
    """A wider collective cannot have a LOWER launch floor.

    More ranks means more synchronisation, and the measured 4->8 all-reduce ratio is
    1.53x-1.91x on every part swept at both widths. A floor that falls with width means
    two widths were fitted from different row sets or different collections.

    Stated as non-decreasing rather than strictly increasing because an alltoall's floor
    is nearly flat in width (median 4->8 ratio 1.090) and an equal reading is credible.
    """
    checked = 0
    for (chip, op, dt, kind), byw in sorted(COLLECTIVES.items()):
        if kind != "floor" or len(byw) < 2:
            continue
        widths = sorted(byw)
        for a, b in zip(widths, widths[1:]):
            assert byw[b] >= byw[a], (
                f"{op} {dt} on {chip}: the {b}-rank floor ({byw[b]}us) is BELOW the "
                f"{a}-rank floor ({byw[a]}us). A wider group synchronises more, so its "
                f"floor cannot be lower."
            )
            checked += 1
    assert checked >= 20, f"only {checked} width pairs checked"


def test_collective_rates_do_not_rise_steeply_as_the_group_widens():
    """Per-GPU throughput does not IMPROVE MATERIALLY with more ranks.

    A loose bound, deliberately, because the strict version is false and the measurements
    say why. Eight committed (part, op, dtype, kind) pairs do rise with width, and they
    are not defects:

      * h100/h200 fp16 all-reduce rises 1.13x from 4 to 8 ranks on the vLLM lane. That
        is vLLM's one-shot custom kernel, not a ring: every rank writes once and reads
        once, so a wider group amortises the launch over more payload instead of adding
        ring hops. The committed entries cite `custom_allreduce_perf.parquet` precisely
        because this kernel behaves differently from NCCL.
      * Several NCCL reduce_scatter and alltoall pairs rise 1.007x-1.041x, which is grid
        resolution on a near-flat curve rather than a measured improvement.

    So the invariant worth asserting is that nothing rises STEEPLY: a rate that nearly
    doubles with width would mean two widths were fitted from different collections,
    which is the defect this guards. 1.25x sits well above the 1.13x the one-shot kernel
    genuinely reaches and well below any plausible cross-collection mix.
    """
    for (chip, op, dt, kind), byw in sorted(COLLECTIVES.items()):
        if kind not in ("peak_rate", "transition_rate") or len(byw) < 2:
            continue
        widths = sorted(byw)
        for a, b in zip(widths, widths[1:]):
            assert byw[b] <= byw[a] * 1.25, (
                f"{op} {dt} {kind} on {chip}: the {b}-rank figure ({byw[b]}) is "
                f"{byw[b] / byw[a]:.2f}x the {a}-rank one ({byw[a]}). A collective does "
                f"not get materially faster per GPU as it widens; this is the signature "
                f"of two widths fitted from different collections."
            )


def test_transition_rate_never_exceeds_peak_rate():
    """The transition rate is the rate in the ramp; the peak is the ceiling.

    `liftCollectiveFloors` prices a collective as max(floor + size/transition,
    size/peak), so a transition above peak makes the second term unreachable and the
    peak coefficient dead.
    """
    checked = 0
    for (chip, op, dt, kind), byw in sorted(COLLECTIVES.items()):
        if kind != "transition_rate":
            continue
        peaks = COLLECTIVES.get((chip, op, dt, "peak_rate"), {})
        for w, v in byw.items():
            if w in peaks:
                assert v <= peaks[w], (
                    f"{op} {dt} on {chip} at {w} ranks: transition {v} exceeds peak "
                    f"{peaks[w]}. The transition rate sits inside the ramp, below the "
                    f"asymptote."
                )
                checked += 1
    assert checked >= 20, f"only {checked} transition/peak pairs checked"


# --------------------------------------------------------------------------------------
# Recurrent families
# --------------------------------------------------------------------------------------

def test_recurrent_rates_are_slower_than_the_convolution_alone():
    """A conv+scan chain is slower than the conv by itself.

    mamba2 is the convolution ALONE (its selective scan is in no collection) while KDA
    and GDN are conv + recurrent scan on the same part. Serial kernels mean reciprocal
    rates add, so the chained families must have the LOWER tokens/us. This is the
    metamorphic relation that would have caught the original KDA fit, whose 1.50 tok/us
    sat above mamba2's conv-only figure and was 6.4x too fast.
    """
    byfam = collections.defaultdict(dict)
    for _, name, body in ENTRIES:
        m = re.match(r"recurrent_decode_rate_(kda|gdn|mamba2)$", name)
        if not m:
            continue
        for chip in (body.get("scope") or {}).get("hardware", []):
            byfam[m.group(1)][chip] = body["value"]
    shared = set(byfam.get("mamba2", {}))
    assert shared, "no mamba2 rates committed; this relation cannot be checked"
    checked = 0
    for fam in ("kda", "gdn"):
        for chip in sorted(shared & set(byfam.get(fam, {}))):
            assert byfam[fam][chip] < byfam["mamba2"][chip], (
                f"{fam} rate on {chip} ({byfam[fam][chip]} tok/us) is NOT slower than "
                f"the conv-only mamba2 figure ({byfam['mamba2'][chip]}). {fam} is that "
                f"same convolution plus a sequential scan, and serial kernels add "
                f"reciprocal rates, so it must be slower."
            )
            checked += 1
    assert checked >= 6, f"only {checked} family/chip comparisons made"


def test_recurrent_floors_are_positive_and_bounded():
    """A per-layer state update is microseconds, not nanoseconds or milliseconds."""
    for setname, name, body in ENTRIES:
        if not re.match(r"recurrent_decode_floor_", name):
            continue
        assert 0.1 <= body["value"] <= 1000.0, (
            f"{setname}: {name} = {body['value']} us is outside the range a single "
            f"recurrent layer's launch can plausibly take."
        )


# --------------------------------------------------------------------------------------
# Scope and provenance hygiene
# --------------------------------------------------------------------------------------

def test_hardware_independent_sets_cover_every_catalog_part():
    """A set that claims to apply everywhere must name every part the catalog holds.

    The host overheads are CPU work in the engine's Python path, so none has a per-part
    value. They are scoped to an explicit list because the validator rejects an empty
    scope -- which means adding a chip to the catalog silently leaves it priced at ZERO
    until someone edits that list. That has now happened twice: once scoped to
    [h100, h200] (zero on all of Blackwell) and once omitting gb300.
    """
    cat = {os.path.basename(p)[:-5]
           for p in glob.glob(str(CATALOG / "hardware" / "*.yaml"))}
    if not cat:
        pytest.skip(f"catalog not present at {CATALOG}")
    for setname, name, body in ENTRIES:
        # `host_*` in cost-model-host-overheads only. `host_link_bandwidth` lives in
        # cost-model-primitives and is a genuinely PER-PART figure -- NVIDIA's descriptor
        # states a different PCIe or NVLink-C2C rate per platform -- so it is scoped to
        # one chip by design and must not be swept into this rule by its name prefix.
        if setname != "cost-model-host-overheads.yaml":
            continue
        scoped = set((body.get("scope") or {}).get("hardware", []))
        missing = sorted(cat - scoped)
        assert not missing, (
            f"{setname}: {name} does not scope {missing}, so it resolves to NOTHING "
            f"there and host cost is priced at zero. Every term in this set is CPU work "
            f"with no per-part value, so the scope must list every catalog part."
        )


def test_every_measured_entry_cites_a_source():
    """`measured` means someone measured it, and the citation is where."""
    for setname, name, body in ENTRIES:
        if body.get("method") != "measured":
            continue
        srcs = body.get("sources") or []
        assert srcs, (
            f"{setname}: {name} is method: measured with no sources. A measurement "
            f"whose provenance is absent cannot be re-derived or audited."
        )


def test_every_assumed_entry_explains_itself():
    """`assumed` is a declared gap, so the rationale carries the whole argument."""
    for setname, name, body in ENTRIES:
        if body.get("method") != "assumed":
            continue
        rat = body.get("rationale") or ""
        assert len(rat) > 200, (
            f"{setname}: {name} is method: assumed with a {len(rat)}-character "
            f"rationale. An assumed value is only as good as its stated reasoning; "
            f"this set's house style states the dimension, the magnitude's caveat, and "
            f"the experiment that would replace it."
        )
        assert not body.get("fitted"), (
            f"{setname}: {name} is assumed AND fitted: true. `fitted` is reserved for "
            f"a value fitted against its own term."
        )


def test_fitted_implies_measured():
    """The schema reserves `fitted: true` for `method: measured`."""
    for setname, name, body in ENTRIES:
        if body.get("fitted"):
            assert body.get("method") == "measured", (
                f"{setname}: {name} is fitted: true with method "
                f"{body.get('method')!r}."
            )


# --------------------------------------------------------------------------------------
# Kind-specific terms against their part-wide siblings
# --------------------------------------------------------------------------------------

# How far a kind-specific decode FLOOR may sit above the same part's part-wide floor.
#
# A floor is a launch-and-setup cost. Changing the attention kind changes what the setup
# reads -- an MLA kernel reads a latent cache, a windowed kernel a bounded block -- which
# moves the floor by a modest factor. It does not multiply it by six: a figure that large
# is the signature of a MODULE measurement, one that includes the projection GEMMs the
# catalog prices as separate nodes in the same layer.
#
# THE DEFECT THIS ENCODES. `attention_decode_floor_mla` shipped at 4.7x-6.2x its part's
# attention floor (51.5-89.5 us against 9.5-14.5 us), fitted cleanly from
# `mla_generation_module_perf.parquet` and labelled `method: measured, fitted: true`. The
# fit was sound and measured the wrong quantity: scripts/fit_attention_mla.py had already
# recorded that the module floor IS the projection weight read (293.6 MB at fp8, 61.2 us
# at H200's 4.80 TB/s, against a 44.0-63.8 us measured module floor), so the kernel charged
# it twice. End-to-end on the InferenceX corpus it cost 2.45 points of overall TPOT
# (14.98% -> 17.43%) and 8.2 points on kimi-k2.5 (6.38% -> 14.57%), flipping the kernel
# from beating AISimulate to losing to it. scripts/correct_mla_floor.py is the correction.
#
# 3.0 rather than something tighter: the measured windowed/full floor ratios span
# 0.77-1.00, so every legitimate kind-specific floor fitted so far sits at or BELOW its
# part-wide sibling. The bound only has to separate "a different kernel's setup" from "a
# different quantity entirely", and the rejected values were above 4.7.
MAX_KIND_FLOOR_RATIO = 3.0


def _by_hardware(prefix: str) -> dict[str, float]:
    """hardware name -> value, for every entry whose name is exactly `prefix`."""
    out: dict[str, float] = {}
    for _, name, body in ENTRIES:
        if name != prefix:
            continue
        for hw in (body.get("scope") or {}).get("hardware", []):
            out[hw] = body["value"]
    return out


def test_kind_specific_decode_floors_are_not_module_measurements():
    """A per-kind decode floor must stay near its part's attention-kernel floor.

    A floor many times larger is measuring a different quantity -- a module including its
    projections -- not a different kernel.
    """
    partwide = _by_hardware("attention_decode_floor")
    assert partwide, "no attention_decode_floor entries; this test would prove nothing"

    checked = 0
    for setname, name, body in ENTRIES:
        if not name.startswith("attention_decode_floor_"):
            continue
        for hw in (body.get("scope") or {}).get("hardware", []):
            base = partwide.get(hw)
            if base is None or base <= 0:
                continue
            ratio = body["value"] / base
            assert ratio <= MAX_KIND_FLOOR_RATIO, (
                f"{setname}: {name} on {hw} is {body['value']} us, {ratio:.2f}x this "
                f"part's attention_decode_floor of {base} us. A kind changes what the "
                f"setup reads, not its order of magnitude; a floor this large is a "
                f"MODULE measurement including the projection GEMMs the catalog prices "
                f"as separate nodes, so the kernel would charge them twice. See "
                f"scripts/correct_mla_floor.py."
            )
            checked += 1
    assert checked, "no kind-specific decode floor was checked against a part-wide floor"
