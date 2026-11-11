#!/usr/bin/env bash
# pimpleFoam path byte gate (docs/fvpimple.md): the static-mesh cases of
# tests/data/fv/p* must not move when the path gains a feature.
#   pimple_bytes.sh BIN W [BASE]
# Runs the six stage-3a cases (and, when BASE holds them, the moving-mesh
# cases mspin mduct msst) with BIN into W/<case>{/,_stats.csv}; with BASE (a W
# of an earlier run, e.g. of a pre-change binary) byte-compares every stats
# column but `ms` and every output file. Run under `simjob submit --gpu`.
# FV_FLAGS (env): extra flags, e.g. "--fv-precision fp64". Exit 1 on any difference.
set -u
R=$(cd "$(dirname "$0")/../.." && pwd)
B=$1; W=$2; BASE=${3:-}
X=${FV_FLAGS:-}
rm -rf "$W"; mkdir -p "$W"
fail=0
cases="pcavity pcavity_skew pchannel pduct psst_plate psst_skew"
for m in mspin mduct msst; do
  if [ -z "$BASE" ] || [ -f "$BASE/${m}_stats.csv" ]; then cases="$cases $m"; fi
done
msCol='/^step,/ { for (i = 1; i <= NF; ++i) if ($i == "ms") c = i }'
noms() { awk -F, -v OFS=, "$msCol"' !/^#/ { if (c) $c = ""; print } /^#/ { print }' "$1"; }
for c in $cases; do
  "$B" flow --polymesh "$R/tests/data/fv/$c" --out "$W/$c" --quiet $X > "$W/$c.log" 2>&1; rc=$?
  if [ -z "$BASE" ]; then echo "case $c rc $rc"; continue; fi
  same=1; n=0
  cmp -s <(noms "$BASE/${c}_stats.csv") <(noms "$W/${c}_stats.csv") || { same=0; echo "  stats differ: $c"; }
  for f in $(cd "$BASE/$c" && find . -type f | sort); do
    n=$((n+1)); cmp -s "$BASE/$c/$f" "$W/$c/$f" || { same=0; echo "  differs: $c/$f"; }
  done
  if [ $rc -eq 0 ] && [ $same -eq 1 ]; then echo "case $c rc 0: stats + $n files identical"; else echo "case $c DIFFERS (rc $rc)"; fail=1; fi
done
echo "fail=$fail"
exit $fail
