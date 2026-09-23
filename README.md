# blis-registry

The performance **registry** for [BLIS](https://github.com/inference-sim/inference-sim):
the *learned* numbers a latency estimate depends on — MFU discounts, regression
coefficients, communication costs — each recording **where it came from** and **where it
holds**. These are numbers a measurement would revise.

## Where it fits in BLIS

BLIS separates three kinds of number by who owns them:

- **Vendor facts** — a GPU's peak TFLOPs, a model's layer count — are declared in the
  [blis-catalog](https://github.com/inference-sim/blis-catalog).
- **Learned numbers** — the coefficients a latency model fits or assumes — live **here**,
  in the registry.
- **Deployment choices** — GPU type, tensor-parallel degree — are stated on the command
  line, not committed anywhere.

Keeping the learned numbers in their own repository is what lets each one carry its
provenance (how it was obtained) and its scope (the range it holds over), so an estimate
can say not just *what* it computed but *on what evidence*.

## What this repository contains

- **`operators/`** — the committed **`CoefficientSet`s**. A `CoefficientSet` is a named,
  immutable set of coefficients that feeds one backend and may `extend` another set,
  inheriting the entries it does not override. Every entry records its value, units,
  method (how the number was obtained), whether it was fitted, and the scope it holds
  over.
- **`backends/`** — one manifest per backend declaring the coefficient **names** that
  backend consumes, so a set that omits a required coefficient is refused by name rather
  than defaulted silently.
- **`validator/`** — a strict validator (Python 3 + PyYAML) that enforces the schema:
  unknown keys, missing provenance, and ill-formed values are rejected, each error naming
  the offending file and entry. It runs in CI over every committed set on every pull
  request and every push to `main` (see `.github/workflows/validate.yml`).

The schema and its rules are defined in `validator/schema.py`, and the committed sets
under `operators/` serve as worked references for the shape each field takes.

## Usage

This repository holds and validates the coefficient data; it does not run simulations.
For how BLIS *reads* and *uses* these coefficients — running an estimate, selecting a
backend, and the rest of the workflow — see the usage examples in
[inference-sim](https://github.com/inference-sim/inference-sim).
