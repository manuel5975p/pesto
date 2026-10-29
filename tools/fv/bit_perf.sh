#!/usr/bin/env bash
# FV performance byte gate (plans/masterplan_performance.md, G-BIT / G-TIME).
#   bit_perf.sh BIN W [BASE]
# Runs the 14 FV gate cases with BIN into W/<case>{/,_stats.csv}; with BASE (a
# W of an earlier run, e.g. of build/perf_base/dns) byte-compares every stats
# column but `ms` and every output file. Then the G-TIME runs (rotor2 L1 120 it, L1 SST
# 60, F1 40): ms/it = mean of the stats `ms` column over iterations 6+.
# Run under `simjob submit --gpu --bench` (timing). Exit 1 on any difference.
# FV_FLAGS (env): extra flags for BIN, e.g. "--fv-precision fp64" (the FP32
# round: FP64 mode vs a pre-FP32 binary, which takes no such flag).
set -u
R=/home/manuel/Documents/pesto
B=$1; W=$2; BASE=${3:-}
X=${FV_FLAGS:-}
rm -rf "$W"; mkdir -p "$W"
fail=0
cases=(
  "cavity $R/tests/data/fv/cavity 60"
  "channel $R/tests/data/fv/channel 60"
  "duct $R/tests/data/fv/duct 40"
  "cavity_lim $R/tests/data/fv/cavity_lim 60"
  "mrf2 $R/tests/data/fv/mrf2 40"
  "cavity_skew $R/tests/data/fv/cavity_skew 60"
  "sa_skew $R/tests/data/fv/sa_skew 60"
  "sst_plate $R/tests/data/fv/sst_plate 60"
  "sst_skew $R/tests/data/fv/sst_skew 60"
  "rotor2_L1_lim $R/runs/fv/mrf/rotor2_L1_lim 30"
  "rotor2_L1 $R/build/final/dns/rotor2_L1 120"
  "rotor2_L2 $R/build/final/dns/rotor2_L2 60"
  "rotor2_L1_sst $R/build/final/dns/rotor2_L1_sst 60"
  "F1 $R/build/final/dns/F1 40"
)
msCol='/^iter,/ { for (i = 1; i <= NF; ++i) if ($i == "ms") c = i }'
noms() { awk -F, -v OFS=, "$msCol"' !/^#/ { if (c) $c = ""; print }' "$1"; }
ms() { awk -F, "$msCol"' !/^#/ && $1 ~ /^[0-9]+$/ && $1 >= 6 { s += $c; n++ } END { printf "%.3f", s / n }' "$1"; }
for x in "${cases[@]}"; do
  set -- $x
  "$B" flow --polymesh "$2" --max-iters "$3" --out "$W/$1" --quiet $X > "$W/$1.log" 2>&1; rc=$?
  if [ -z "$BASE" ]; then echo "case $1 rc $rc"; continue; fi
  same=1; n=0
  # every stats column but `ms`
  cmp -s <(noms "$BASE/$1_stats.csv") <(noms "$W/$1_stats.csv") || { same=0; echo "  stats differ: $1"; }
  for f in $(cd "$BASE/$1" && find . -type f | sort); do
    n=$((n+1)); cmp -s "$BASE/$1/$f" "$W/$1/$f" || { same=0; echo "  differs: $1/$f"; }
  done
  if [ $rc -eq 0 ] && [ $same -eq 1 ]; then echo "case $1 rc 0: stats + $n files identical"; else echo "case $1 DIFFERS (rc $rc)"; fail=1; fi
done
for x in "rotor2_L1 120" "rotor2_L1_sst 60" "F1 40"; do
  [ -n "${NO_TIME:-}" ] && break   # NO_TIME (env): byte gate only, skip G-TIME
  set -- $x
  "$B" flow --polymesh "$R/build/final/dns/$1" --max-iters "$2" --out "$W/t_$1" --quiet $X > /dev/null 2>&1
  echo "time $1 $2 it: $(ms "$W/t_$1_stats.csv") ms/it (it 6+)"
done
rm -rf "$W"/t_*/[0-9]* 2>/dev/null
echo "fail=$fail"
exit $fail
