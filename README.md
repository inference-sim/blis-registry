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

A **coefficient set** is an immutable collection of coefficients that feeds one latency
model. Each coefficient records its value alongside the provenance and scope that make it
auditable, so a reader can tell a measured number from an assumed one and know the range
it was established over. Sets are organized so that related numbers can be shared and
built upon rather than duplicated.

The repository validates its own contents: the committed data is checked against the
schema so that missing provenance, unrecognized fields, or ill-formed values are rejected
— each failure pointing at the specific place it occurred — before the data is relied on.

## Authoring and validating a set

A coefficient set is one YAML file under `operators/`, identified by its filename. It
declares `kind: CoefficientSet`, the `backend` it feeds, and a map of `coefficients`; it
may `extends` another set — one in the same scanned root (namespace), named by filename
stem — to inherit entries it does not override. Each backend under `backends/` declares the
coefficient names it consumes, and a set is validated against that list — an omitted
coefficient is refused by name rather than silently defaulted.

Every coefficient records `value`, `units`, `method` (how the number was obtained),
`fitted`, and `scope` (the range it holds over), plus optional provenance
(`sources`, `rationale`, `ci95`, …). The strict rules — which fields are required, the
allowed enums, and the required-by-method provenance — are defined and enforced by the
validator in `validator/`, which is the authoritative, maintained specification.

To validate the registry (the same check CI runs on every pull request):

```sh
pip install -r requirements.txt
python validator/validate.py            # validate every committed set
python -m pytest validator/ -q          # run the validator's own test suite
```

The validator prints one line per problem, each naming the file and the offending entry
or key, and exits non-zero if any set is rejected.

Real sets live in `operators/` — the roofline MFU sets and the trained-physics
correction/overhead sets (one of each per supported GPU) are transcribed there, with more
added by later tasks. A trained-physics set `extends` its GPU's roofline set to inherit the
MFU pair: the correction/overhead coefficients are one global vector, while MFU is per-GPU.
A committed, clearly-labelled **synthetic** set in `fixtures/` gives CI a committed
artifact to validate on every run; it is a schema fixture, never real data, and forms a
separate namespace — a set under one scanned root cannot `extends` or collide with a set
under another.

## Usage

This repository holds and validates the coefficient data; it does not run simulations.
For how BLIS *reads* and *uses* these coefficients as part of producing an estimate, see
[inference-sim](https://github.com/inference-sim/inference-sim).
