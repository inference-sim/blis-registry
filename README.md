# blis-registry

The performance **registry** for [BLIS](https://github.com/inference-sim/inference-sim):
the *learned* numbers a latency estimate depends on — the coefficients a latency model
fits or assumes — each recording **where it came from** and **where it holds**. These are
numbers a measurement would revise.

## Where it fits in BLIS

BLIS separates the numbers behind an estimate by who owns them:

- **Vendor facts** — the declared properties of hardware and models — are held in the
  [blis-catalog](https://github.com/inference-sim/blis-catalog).
- **Learned numbers** — the coefficients an estimate fits or assumes — live **here**, in
  the registry.
- **Deployment choices** — the knobs a particular run selects — are stated at run time,
  not committed anywhere.

Keeping the learned numbers in their own repository is what lets each one carry its
provenance (how it was obtained) and its scope (the range it holds over), so an estimate
can say not just *what* it computed but *on what evidence*.

## What this repository holds

A **coefficient set** is a standalone, immutable collection of coefficients that feeds one
latency model. Each coefficient records its value alongside the provenance and scope that
make it auditable, so a reader can tell a measured number from an assumed one and know the
range it was established over. Each set is a self-identifying document, complete on its own
— there is no inheritance between sets.

The repository validates its own contents: the committed data is checked against the
schema so that missing provenance, unrecognized fields, or ill-formed values are rejected
— each failure pointing at the specific place it occurred — before the data is relied on.

## Authoring and validating a set

A coefficient set is one YAML file under `coefficients/`. It declares `kind:
CoefficientSet`, a unique `name` (the set's identity — e.g. `roofline-h100`), and a
`coefficients` list. Each list entry is a single-key map keyed by the coefficient name.
Sets are standalone: there is no `backend` field and no inheritance between sets.

Every coefficient records `value`, `units`, `method` (how the number was obtained),
`fitted`, and `scope` (the range it holds over), plus optional provenance
(`sources`, `rationale`, `ci95`, …). The strict rules — which fields are required, the
allowed enums, and the required-by-method provenance — are defined and enforced by the
validator in `validator/`, which is the authoritative, maintained specification. The
validator checks **shape only**; per-backend completeness (that a set carries every
coefficient a given backend consumes) is enforced by the simulator-side loader, which
knows what each backend reads.

A minimal set looks like:

```yaml
kind: CoefficientSet
name: roofline-l40s
coefficients:
  - mfu_prefill:
      value: 0.32
      units: dimensionless
      method: literature
      fitted: false
      sources:
        - {kind: discussion, cite: "inference-sim#589", role: primary}
      rationale: >
        L40S prefill MFU discount; see #589.
      scope: {hardware: [L40S]}
```

To validate the registry (the same check CI runs on every pull request):

```sh
pip install -r requirements.txt
python validator/validate.py            # validate every committed set
python -m pytest validator/ -q          # run the validator's own test suite
```

The validator prints one line per problem, each naming the file and the offending entry
or key, and exits non-zero if any set is rejected.

Real sets live in `coefficients/` — the roofline MFU sets (one per supported GPU), the
`trained-physics` correction set, the `lora-adapter-costs` set (Digital-Twin adapter cost
terms), and the `legacy-kv-transfer` set (the pre-#1590 CPU↔GPU transfer defaults) are
transcribed there, with more added by later tasks. These committed sets are what CI
validates on every run.

## Usage

This repository holds and validates the coefficient data; it does not run simulations.
For how BLIS *reads* and *uses* these coefficients as part of producing an estimate, see
[inference-sim](https://github.com/inference-sim/inference-sim).
