#!/usr/bin/env bash
# G32 (docs/nozzle.md): dispatches per step of the compressible FV route, none vs --shock hll, by
# rocprofv3's kernel trace between two run lengths. Under simjob --gpu.
#   shock_cost.sh BIN WORKDIR CASE BC
# CASE, BC: a polyMesh with a --bc file that runs without --shock (a silent hll run).
set -u
B=$1; W=$2; C=$3; F=$4
rm -rf "$W"; mkdir -p "$W"
A="flow --physics compressible --polymesh $C --bc $F --re 40 --mach 0.2 --quiet --emit none"
cnt() { # tag tt extra -> kernels steps
  local t=$1 tt=$2; shift 2
  rocprofv3 --kernel-trace --output-format csv -d "$W/$t" -o tr -- "$B" $A --tt "$tt" "$@" --out "$W/$t.o" > "$W/$t.log" 2>&1
  local f; f=$(find "$W/$t" -name '*kernel_trace.csv' | head -1)
  local k; k=$(awk 'NR > 1 { n++ } END { print n + 0 }' "$f")
  local s; s=$(grep -oE 'done: [0-9]+ steps' "$W/$t.log" | awk '{print $2}')
  echo "$k $s"
}
rate() { # tag extra: dispatches per step over the longer run's extra steps
  local t=$1; shift
  read -r k1 s1 < <(cnt "${t}1" 0.05 "$@")
  read -r k2 s2 < <(cnt "${t}2" 0.2 "$@")
  awk -v k1="$k1" -v k2="$k2" -v s1="$s1" -v s2="$s2" -v t="$t" 'BEGIN { printf "%s: %.2f dispatches per step (%d kernels, %d to %d steps)\n", t, (k2 - k1) / (s2 - s1), k2, s1, s2 }'
}
rate none
rate hll --shock hll
rate d40 --shock hll --shock-d4 0
rate sense --shock sense
