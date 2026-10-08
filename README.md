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
CoefficientSet`, a unique `name` (the set's identity — e.g. `cost-model-attention`), and a
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
name: example-set
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

To run the Python validator locally (the provenance-and-evidence half of what CI checks;
the Go schema is the other half — see [Two validators, by design](#two-validators-by-design)):

```sh
pip install -r requirements.txt
python validator/validate.py            # validate every committed set
python -m pytest validator/ -q          # run the validator's own test suite
```

The validator prints one line per problem, each naming the file and the offending entry
or key, and exits non-zero if any set is rejected.

### Two validators, by design

CI runs the committed sets through **two** validators, which own different rules:

- The **Python** validator in `validator/` owns **provenance and evidence policy** — the
  `sources`, `method`, `fitted`, scope-key vocabulary, and supersession checks above.
- **blis-schemas** (the Go schema every consumer loads these files through —
  `blis-latency-kernel` reads them via `blisschemas.LoadCoefficientSet`) owns **document
  shape and resolver-facing invariants** — the vocabulary, required fields, finiteness,
  and the duplicate-`(name, scope)` rule a resolver's last-wins stacking depends on.

The `schema-validate` CI job runs the Go gate (`blis-schemas` `cmd/validate-registry`,
pinned by SHA) so a set that would be rejected at the point of use is caught here, on the
PR that introduces it, rather than downstream in another repository. Stating the split
keeps a rule from being added to one validator and silently absent from the other.

Real sets live in `coefficients/`: the five `cost-model-*` sets that price a step —
primitives, collectives, attention, recurrent and host overheads. Those five are exactly
what `blis-latency-kernel` loads, and nothing in `coefficients/` is unread by it. These
committed sets are what CI validates on every run, and every value in them is re-derivable
from public data — see
[`docs/reproducing-coefficients.md`](docs/reproducing-coefficients.md).

## Usage

This repository holds and validates the coefficient data; it does not run simulations.
For how BLIS *reads* and *uses* these coefficients as part of producing an estimate, see
[inference-sim](https://github.com/inference-sim/inference-sim).
