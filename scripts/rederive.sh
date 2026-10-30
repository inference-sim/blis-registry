#!/usr/bin/env bash
# The re-derivation gates, with the data present: every fitted value must be what its
# writer derives from the pinned upstream data, and no test may skip except the eight
# by-design ramp-form ones and a named cited-bodies-absent skip (below). A skip is a gate
# that did not run, so it fails here.
#
# Run from the repository root after `pip install -r requirements-fit.txt`:
#
#   export AISIMULATE_DATA=<aisimulate>/python/aisimulate/src/aisimulate_core/systems/data
#   export BLIS_CATALOG=<blis-catalog checkout>        # at CATALOG_SHA in rederive.yml
#   export COLLECTIVEX_DATA=<dir of cx_run_*.body>     # test_deepep_ll_graph.py
#   export INFERENCEX_GOLDEN=<inferencex>/inferencex-e2e/infx/golden_al_distribution  # optional
#   scripts/rederive.sh
#
# The pinned commits (AISIMULATE_SHA, CATALOG_SHA, INFERENCEX_SHA) live in
# .github/workflows/rederive.yml, which runs this same script on manual dispatch.
set -euo pipefail

: "${AISIMULATE_DATA:?set AISIMULATE_DATA to the AISimulate systems/data directory}"
: "${BLIS_CATALOG:?set BLIS_CATALOG to a blis-catalog checkout at the pinned CATALOG_SHA}"

n=$(find "$AISIMULATE_DATA" -name '*.parquet' | wc -l)
echo "AISimulate parquets present: $n"
test "$n" -gt 500

# The vendored descriptors must be the catalog being checked against.
diff -r validator/testdata/catalog/hardware "$BLIS_CATALOG/hardware"
diff -r validator/testdata/catalog/networks "$BLIS_CATALOG/networks"
for g in validator/testdata/catalog/models/*/graph.yaml; do
  diff "$g" "$BLIS_CATALOG/models/${g#validator/testdata/catalog/models/}"
done
# Every catalog model's model.yaml, and no other: the acceptance emitter refuses a
# model absent from this list, so a partial copy would refuse a real one.
diff <(cd validator/testdata/catalog/models && ls -1 */model.yaml) \
     <(cd "$BLIS_CATALOG/models" && ls -1 */model.yaml)
for m in validator/testdata/catalog/models/*/model.yaml; do
  diff "$m" "$BLIS_CATALOG/models/${m#validator/testdata/catalog/models/}"
done

# With an InferenceX checkout, the vendored golden curves must be its curves, all of
# them, and the emitter must cite INFERENCEX_SHA (when given) and reproduce the set.
if [ -n "${INFERENCEX_GOLDEN:-}" ]; then
  vendored=validator/testdata/inferencex/golden_al_distribution
  diff <(cd "$vendored" && ls -1 *.yaml) <(cd "$INFERENCEX_GOLDEN" && ls -1 *.yaml)
  for f in "$vendored"/*.yaml; do diff "$f" "$INFERENCEX_GOLDEN/${f##*/}"; done
  if [ -n "${INFERENCEX_SHA:-}" ]; then
    grep -q "^INFERENCEX_SHA = \"$INFERENCEX_SHA\"$" scripts/emit_acceptance_profiles.py
  fi
  python scripts/emit_acceptance_profiles.py --check
fi

log=$(mktemp)
trap 'rm -f "$log"' EXIT
python -m pytest validator/ -q -rs | tee "$log"
# With the data present, the only skips are the eight by-design ramp-form ones (a part
# carries the one- or three-factor ramp, so the other gate does not apply). Any other
# skip -- missing data, a missing catalog, a gate whose premise quietly disappeared --
# is a gate that did not run, and fails.
# One data skip is accepted, by name: a CollectiveX writer whose cited run bodies are no
# longer served (the API keeps its latest 15 runs) refuses to re-cite them and skips with
# "the cited CollectiveX bodies are not present". Its entries then cannot be re-derived from
# public data, and the skip line says which runs are missing.
unexpected=$(grep '^SKIPPED' "$log" \
  | grep -Ev 'carries the (one|three)-factor ramp|the cited CollectiveX bodies are not present' \
  || true)
if [ -n "$unexpected" ]; then
  echo "unexpected skips with the data present:" >&2
  echo "$unexpected" >&2
  exit 1
fi
echo "rederive: every gate ran and passed"
