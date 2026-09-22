# R2G1 — CoefficientSet schema + strict validator + CI

**Goal:** Bootstrap the empty `blis-registry` repo with the `CoefficientSet` schema,
a strict validator that runs in CI, per-backend "consumed names" manifests, and at
least one committed set for CI to validate.
**Source:** inference-sim/blis-registry#1 (R2G1). Tracker: inference-sim/inference-sim#1817.
**Closes:** Fixes #1

## Source Document Audit (Step 1.5)

The issue has **no comments**, so the source document is the issue body alone —
no design refinements to reconcile (`REFINEMENT` thread is empty). One
interpretation choice was required:

- **CLARIFICATION — validator language.** The issue explicitly says "language is an
  implementation choice." Chosen: **Python 3 + PyYAML**. Rationale: the validator is
  a standalone CI tool with no simulator coupling; the sibling `blis-catalog` repo
  ships *no* validator by design ("validation is the simulator's"), so there is no Go
  toolchain convention to match here. Python gives a zero-build, self-contained script
  and a YAML loader we can subclass to enforce strict duplicate-key detection.
- **CLARIFICATION — what counts as "at least one committed set."** R2 is
  value-preserving and G2/G3/G3b/G4 *own* transcribing the real numbers. So this PR
  commits only a clearly-labeled **example** set (`operators/example.yaml`) exercising
  every schema feature, and ships the real **backend manifests** (a G1 deliverable).
  It does **not** transcribe real roofline/trained-physics values — that is G2/G3.

## The schema (from the issue)

A `CoefficientSet` is a named, immutable set of coefficients that feeds **one backend**
and may **extend** another set. Strictly parsed: unknown key ⇒ error.

Top-level keys (strict):
- `name` (str, required) — the set's identifier.
- `backend` (str, required) — which backend consumes this set; must match a manifest.
- `extends` (str, optional) — name of another set this one inherits from.
- `coefficients` (map, required) — entry-name → entry.

Each entry requires five fields: `value`, `units`, `method`, `fitted`, `scope`.
Optional: `ci95`, `sources`, `rationale`, `copied_from`, `supersedes`.

Enums:
- `units` ∈ {dimensionless, us_per_layer, us_per_request, us_per_step, us_per_hop}
- `method` ∈ {measured, literature, vendor_spec, copied, assumed, not_charged}
- `fitted` is bool; `fitted: true` only permitted when `method: measured`.

Required-by-method:
- `rationale` required for every method except `measured`.
- `sources` (non-empty list) required for `literature` and `vendor_spec`.
- `copied_from` required for `copied`.

Value rules:
- `value: 0` (zero) is rejected unless `method: not_charged`.
- `value` must be a finite number (reject NaN/Inf/non-numeric).

Scope: an open-keyed map. Known scope keys: `hardware`, `tp`, `ep`, `nodes_spanned`,
`model_class`. An unknown scope key is an error (strict), but adding a known one later
is a schema addition, not a redesign.

Backend manifest: `backends/<name>.yaml` declares the coefficient names the backend
consumes (`consumes: [ ... ]`). A set whose backend consumes a name absent from its
`coefficients` (after resolving `extends`) is refused, naming the coefficient.
ABSENT (name not in `consumes`) ≠ ZERO (`value: 0` + `method: not_charged`).

## Behavioral contracts (GIVEN/WHEN/THEN)

- **BC-1 (missing required entry field):** GIVEN an entry missing `method`, `units`, or
  `scope`, WHEN validated, THEN rejected with a message naming the entry and the field.
- **BC-2 (method companion field):** GIVEN a `method` lacking its required companion
  (`rationale` for non-measured; `sources` for literature/vendor_spec; `copied_from`
  for copied), WHEN validated, THEN rejected naming the entry and the missing companion.
- **BC-3 (zero value):** GIVEN `value: 0`, WHEN validated, THEN rejected UNLESS
  `method: not_charged`.
- **BC-4 (strict parse):** GIVEN an unknown top-level key, an unknown entry field, or an
  unknown `scope` key, WHEN validated, THEN rejected naming the offending key. GIVEN a
  duplicate mapping key, THEN rejected (no silent last-wins).
- **BC-5 (backend refuses by name):** GIVEN a set whose declared backend `consumes` a
  coefficient absent from the resolved `coefficients`, WHEN validated, THEN rejected
  naming the coefficient. A coefficient present in the set but not in `consumes` is
  **allowed** — that is the ABSENT case and the shared-base "drop" case (a child
  `extends` a base but its backend does not price every inherited term). Only the
  omitted-consumed direction is an error, per the acceptance criteria.
- **BC-6 (ci95 optional):** GIVEN `ci95` present, THEN accepted; GIVEN it absent, THEN
  never required.
- **BC-7 (enum + type validation):** GIVEN an out-of-enum `units`/`method`, a non-bool
  `fitted`, `fitted: true` with a non-measured method, or a non-finite `value`, THEN
  rejected naming the entry.
- **BC-8 (extends resolution):** GIVEN `extends: base`, an entry defined only in `base`
  satisfies the backend `consumes` check for the child; a child entry overrides the
  base entry of the same name. An `extends` naming a set not present is rejected.
- **BC-9 (valid set passes):** GIVEN the committed `operators/example.yaml`, WHEN
  validated against its declared backend manifest, THEN it passes with exit code 0.
- **BC-10 (CI gate):** The CI workflow runs the validator over every file in
  `operators/` and the unit test suite; a rejection or a failing test fails the job.

## TDD task breakdown

1. **T1 — scaffold + strict YAML loader.** `validator/loader.py`: a PyYAML `SafeLoader`
   subclass that raises on duplicate keys. Test: duplicate key raises; normal map loads.
2. **T2 — entry field presence + enums (BC-1, BC-7).** `validator/schema.py` entry
   checks. Tests: missing units/method/scope each rejected naming entry; bad enum
   rejected; non-bool `fitted`; `fitted:true` non-measured rejected.
3. **T3 — value rules (BC-3, BC-7).** Zero rejected unless not_charged; NaN/Inf/string
   rejected. Tests for each.
4. **T4 — required-by-method (BC-2).** rationale/sources/copied_from rules. Tests each.
5. **T5 — strict top-level + scope keys (BC-4).** Unknown top-level key, unknown entry
   field, unknown scope key each rejected. Tests each.
6. **T6 — backend manifest + extends (BC-5, BC-8).** Load `backends/<name>.yaml`,
   resolve `extends`, refuse omitted-consumed-by-name, allow present-but-not-consumed,
   reject unknown `extends`. Tests each.
7. **T7 — ci95 optional + valid set (BC-6, BC-9).** ci95 present passes / absent passes;
   example set validates clean.
8. **T8 — CLI entrypoint + repo files.** `validator/validate.py` main: validate one file
   or a directory, exit non-zero on any rejection, print one line per error. Backend
   manifests (`roofline`, `trained-physics`), `operators/example.yaml`, README,
   `.gitignore`, `.gitattributes`, `requirements.txt`.
9. **T9 — CI workflow.** `.github/workflows/validate.yml`: install PyYAML, run tests,
   run validator over `operators/`.

## Sanity checklist (relevant)

- **Determinism:** validator output is sorted (errors sorted before print); no map-order
  dependence in messages.
- **Strict parse (R1-style):** unknown keys never silently defaulted; duplicate keys
  never silently last-wins.
- **Input robustness:** empty file, non-mapping root, empty `coefficients`, non-finite
  numbers, wrong types all produce a named error, never a stack trace escaping to CI.
- **No simulator wiring:** this PR touches only the registry repo (out-of-scope guard).
- **Value-preserving:** no real coefficient values transcribed (example values are
  labeled illustrative).

## Deviation log

- **CLARIFICATION (validator language):** Python 3 + PyYAML — see Source Document Audit.
- **CLARIFICATION (committed set):** ships an `example.yaml` + real backend manifests, not
  real coefficient values (those are G2/G3/G3b/G4; R2 is value-preserving).
- **DESIGN (present-but-not-consumed is allowed, not an error):** the issue mandates
  refusing only an *omitted* consumed coefficient by name. A coefficient present but not
  in `consumes` is deliberately allowed — it is the ABSENT case and the shared-base
  "drop" scenario the issue calls out (reuse shared bases while dropping/zeroing terms
  without a schema change). Enforcing symmetric equality would wrongly reject a valid
  `extends` where the child backend inherits but does not price every base term.

## Code-review outcome (Step 4.5)

An adversarial code review (crash-input focus) found and I fixed, before commit:

- **CRITICAL — three crash-on-malformed-input paths** violating the plan's "never a
  stack trace escaping to CI" contract, all reachable from a committed `operators/*.yaml`:
  (1) mixed-type mapping keys crashed `sorted()` → now `sorted(..., key=str)` plus a
  named "coefficient name must be a string" error; (2) a non-UTF-8 file raised
  `UnicodeDecodeError` (a `ValueError`, not `OSError`) uncaught → `LOAD_ERRORS` now
  covers `ValueError`/`TypeError`; (3) a complex/unhashable YAML key crashed the loader
  → the loader now names it as an unsupported complex key.
- **IMPORTANT — malformed backend manifest** was indistinguishable from a missing one
  and silently skipped the consumed-names check → `_backend_consumes` now returns a
  distinct reason for missing vs malformed and names the real problem.
- **MINOR — provenance typing:** `rationale`/`copied_from` now require non-empty
  *strings* and `sources` a list of non-empty strings (a bare number no longer passes).
- **Coverage:** added `operators/example-extends.yaml` so CI exercises the `extends`
  inheritance path over a committed set (BC-8 end-to-end), plus tests for every crash
  input above. Test count 39 → 48.
