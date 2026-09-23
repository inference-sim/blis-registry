# blis-registry

The performance **registry** for [BLIS](https://github.com/inference-sim/inference-sim):
the *learned* numbers a latency estimate depends on — MFU discounts, regression
coefficients, communication costs — each stating **where it came from** and **where it
holds**. These are numbers a measurement would revise, as opposed to the vendor facts
(a GPU's peak TFLOPs, a model's layer count) that live in the
[blis-catalog](https://github.com/inference-sim/blis-catalog), and the deployment choices
(GPU type, tensor-parallel degree) stated on the command line.

This repository owns the **schema** for those numbers and a **validator** that runs in
CI. It bootstraps the registry (R2G1); the real coefficient values are transcribed by
later tasks (R2G2/G3/G3b/G4), all written and validated against the schema fixed here.

## Layout

```
blis-registry/
├── backends/                   # WHAT A BACKEND CONSUMES — one file per backend
│   ├── roofline.yaml           #   the coefficient NAMES the backend prices
│   └── trained-physics.yaml
├── operators/                  # THE COEFFICIENT SETS — one file per set
│   └── example.yaml            #   illustrative; real values arrive in R2G2/G3/G3b/G4
└── validator/                  # the strict validator + its tests (Python 3 + PyYAML)
    ├── validate.py             #   CLI: validate a file or a directory
    ├── schema.py               #   the CoefficientSet rules
    ├── loader.py               #   strict YAML load (duplicate + complex keys are errors)
    └── test_validate.py
```

## The CoefficientSet schema

A **`CoefficientSet`** is a named, immutable set of coefficients that feeds **one
backend** and may **extend** another set, inheriting entries it does not override. It is
**strictly parsed**: an unknown key is an error, never a silent default.

```yaml
name: example
backend: roofline          # must match a manifest in backends/
extends: some-base-set     # optional; inherits entries not overridden here
coefficients:
  mfu_prefill:
    value: 0.45            # the number (not a "correction" — some entries are rates)
    units: dimensionless   # dimensionless | us_per_layer | us_per_request | us_per_step | us_per_hop
    method: literature     # measured | literature | vendor_spec | copied | assumed | not_charged
    fitted: false          # true only for `measured` (a free parameter fit on its own term)
    sources: ["https://github.com/inference-sim/inference-sim/discussions/589"]
    rationale: "Prefill MFU discount; see #589."
    scope: {hardware: [H100]}   # the range the entry is known to hold over
```

Every entry requires `value`, `units`, `method`, `fitted`, and `scope`. Optional:
`ci95`, `sources`, `rationale`, `copied_from`, `supersedes`.

**Required by method:** `rationale` for every method except `measured`; a non-empty
`sources` list for `literature` and `vendor_spec`; `copied_from` for `copied`.

**`ABSENT` and `ZERO` are distinct.** *Absent* means the backend does not price that term
— the coefficient name is simply not in the backend's `consumes` list. *Zero* means the
backend prices it at no cost, and is written as `value: 0` with **`method: not_charged`**.
A zero value with any other method is rejected, so a silently-dropped coefficient can
never masquerade as a deliberate zero.

**Scope keys are open.** Known keys are `hardware`, `tp`, `ep`, `nodes_spanned`,
`model_class`. A key the validator does not know is an error (strict parse); adding a new
one is a one-line schema addition (`SCOPE_KEYS` in `validator/schema.py`), not a redesign.

## Backend manifests

A backend **declares the coefficient names it consumes** in `backends/<name>.yaml`:

```yaml
consumes: [mfu_prefill, mfu_decode]
```

The validator checks each set against its backend's list, so a coefficient the backend
prices but the set omits is **refused by name** rather than defaulted silently. A
coefficient present in a set but *not* consumed by its backend is allowed — that is the
ABSENT case, and it lets a set reuse a shared base (via `extends`) while a backend
drops or zeroes terms it does not price, with no schema change.

## Validating

```sh
pip install -r requirements.txt
python validator/validate.py            # validate every set in operators/
python validator/validate.py operators/example.yaml   # or a single file
python -m pytest validator/ -q          # run the validator's own tests
```

The validator prints one line per problem, each naming the file and the offending entry
or key, and exits non-zero if any set is rejected. CI (`.github/workflows/validate.yml`)
runs the unit tests and validates every committed set on every pull request and on
every push to `main`.

## Scope of this repository

The registry holds the numbers and validates them here. It does **not** wire them into
the simulator: the simulator learning to *read* these sets, and the cross-repo check that
loads the real registry through the real resolver, are separate follow-up work (the
N-track / R2H3 extension). This repository defines the schema and the registry-side
validation only.
