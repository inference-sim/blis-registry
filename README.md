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

## Usage

This repository holds and validates the coefficient data; it does not run simulations.
For how BLIS *reads* and *uses* these coefficients as part of producing an estimate, see
[inference-sim](https://github.com/inference-sim/inference-sim).
