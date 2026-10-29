#!/usr/bin/env bash
# Kernels, memsets and memory copies per SIMPLE iteration of the FV path
# (plans/masterplan_performance.md): rocprofv3 kernel + HIP API traces of BIN on rotor2 L1 at
# 10 and 40 iterations, (count(40) - count(10)) / 30. Run under simjob --gpu.
#   count_launches.sh BIN W [CASE]
set -u
R=/home/manuel/Documents/pesto
B=$1; W=$2; C=${3:-$R/build/final/dns/rotor2_L1}
rm -rf "$W"; mkdir -p "$W"
for n in 10 40; do
  rocprofv3 --kernel-trace --hip-runtime-trace --output-format csv -d "$W/p$n" -o tr -- \
    "$B" flow --polymesh "$C" --max-iters $n --out "$W/o$n" --quiet > "$W/p$n.log" 2>&1
done
cnt() { # file pattern(awk regex on the row) -> rows
  local f; f=$(find "$1" -name "$2" | head -1)
  [ -n "$f" ] || { echo 0; return; }
  awk -v re="$3" 'NR > 1 && $0 ~ re { n++ } END { print n + 0 }' "$f"
}
for n in 10 40; do
  k[$n]=$(cnt "$W/p$n" '*kernel_trace.csv' '.')
  f[$n]=$(cnt "$W/p$n" '*kernel_trace.csv' '__amd_rocclr_fill')
  d[$n]=$(cnt "$W/p$n" '*hip_api_trace.csv' '"hipMemcpy"')
  a[$n]=$(cnt "$W/p$n" '*kernel_trace.csv' '__amd_rocclr_copy')
done
awk -v k1="${k[10]}" -v k4="${k[40]}" -v f1="${f[10]}" -v f4="${f[40]}" -v d1="${d[10]}" -v d4="${d[40]}" \
    -v a1="${a[10]}" -v a4="${a[40]}" 'BEGIN {
  printf "per iteration (it 11-40): dispatches %.1f = solver kernels %.1f + memset fills %.1f + copy kernels %.1f; hipMemcpy calls %.1f\n",
         (k4 - k1) / 30, (k4 - k1 - f4 + f1 - a4 + a1) / 30, (f4 - f1) / 30, (a4 - a1) / 30, (d4 - d1) / 30 }'
