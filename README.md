# blis-registry

The learned numbers behind a [BLIS](https://github.com/inference-sim/inference-sim) latency
estimate: the coefficients a cost model fits from measurements or, where no measurement
exists, assumes. Each coefficient records how it was obtained and which hardware it
applies to, so an estimate can be traced back to the evidence it rests on.

**Documentation: <https://inference-sim.github.io/blis-registry/>**

## Where it fits in BLIS

BLIS keeps different kinds of number in different repositories:

- **Declared facts**, the properties a vendor states about models and hardware, are in
  [blis-catalog](https://github.com/inference-sim/blis-catalog).
- **Learned numbers**, the coefficients an estimate fits or assumes, are here.
- **Deployment choices**, the GPU, parallelism and engine settings of a run, are stated
  per run in a scenario and committed nowhere.

[blis-schemas](https://github.com/inference-sim/blis-schemas) defines and validates the
file format. [blis-latency-kernel](https://github.com/inference-sim/blis-latency-kernel)
reads the sets a scenario names and prices each step with them, and inference-sim calls
the kernel once per simulated step.

## What is here

| Path | Contents |
|---|---|
| `coefficients/` | The coefficient sets, one YAML file each. |
| `scripts/` | The fitters and writers that produce them from public data. |
| `validator/` | Property and re-derivation tests over the committed sets. |
| `docs/` | The documentation site, including the methodology. |

Six sets are committed. Five price a forward pass and are the ones blis-latency-kernel's
scenarios load: `cost-model-primitives`, `cost-model-collectives`,
`cost-model-attention`, `cost-model-recurrent` and `cost-model-host-overheads`. The
sixth, `cost-model-memory`, holds memory-occupancy magnitudes and is committed ahead of
its reader. The [reference pages](https://inference-sim.github.io/blis-registry/latest/reference/)
list every entry, generated from these files.

## A coefficient

```yaml
kind: CoefficientSet
name: cost-model-primitives
coefficients:
  - gemm_m_half_bf16:
      value: 94
      units: tokens
      method: measured
      fitted: true
      scope: {hardware: [a100-80, a100-sxm]}
      sources:
        - {kind: model, cite: "NVIDIA AISimulate systems/data/a100_sxm/gemm/vllm/0.14.0/gemm_perf.parquet ...", role: primary}
      rationale: >
        The token count at which the ramp reaches half its asymptote ...
```

Every entry states its `value`, `units`, `method` (`measured`, `vendor_spec`,
`assumed`, …), whether it was `fitted`, and its `scope`; most also carry `sources` and a
`rationale`. The [file format](https://inference-sim.github.io/blis-registry/latest/reference/format/)
gives the full rules.

## Checks

```sh
pip install -r requirements.txt
python -m pytest validator/ -q -rs
```

CI runs three jobs on every pull request:

- **schema-validate** runs blis-schemas' validator (`cmd/validate-registry`, pinned by
  tag) over every set, the same rules a consumer applies when it loads one.
- **derivation-tests** runs `validator/`. The property tests always run. The
  re-derivation tests need a local clone of AISimulate and skip in CI; run them locally,
  with `AISIMULATE_DATA` set, before changing a fitted value.
- **docs** builds the documentation in strict mode.

## Documentation

```sh
pip install -r requirements-docs.txt
mkdocs serve
```

The site is published from `main` as `dev` and from each release as its version, with
`latest` pointing at the newest release. See
[Releases and pinning](https://inference-sim.github.io/blis-registry/latest/using/releases/).

## Licence

Apache 2.0.
