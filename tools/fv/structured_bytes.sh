#!/usr/bin/env bash
# Byte gates of `dns flow --solver simple` (docs/fvstructured.md).
#   structured_bytes.sh BIN BASE W
# (a) --polymesh simpleFoam cases and structured flows: BIN vs BASE (a pre-change
#     binary): stats but `ms`, force.dat, moment.dat, fields, byte for byte.
# (b) `--solver simple --out A` vs `--export B` + `--polymesh B --out C`.
# Run under `simjob submit --gpu -- /bin/bash structured_bytes.sh ...`: simjob
# execs a script by fd, so $0 would not find the repo. Exit 1 on any difference.
set -u
R=$(cd "$(dirname "$0")/../.." && pwd)
B=$1; BASE=$2; W=$3
rm -rf "$W"; mkdir -p "$W"
fail=0
msCol='/^iter,/ { for (i = 1; i <= NF; ++i) if ($i == "ms") c = i }'
noms() { awk -F, -v OFS=, "$msCol"' !/^#/ { if (c) $c = ""; print }' "$1"; }
same() { # same TAG A B
  if cmp -s "$2" "$3"; then echo "  same $1"; else echo "  DIFFERENT $1"; fail=1; fi
}
# Every file of tree A equals its twin in tree B.
sametree() { # sametree TAG A B
  local n=0 bad=0
  while IFS= read -r f; do
    n=$((n + 1)); cmp -s "$2/$f" "$3/$f" || { bad=$((bad + 1)); echo "  DIFFERENT $1/$f"; }
  done < <(cd "$2" && find . -type f | sort)
  echo "  $1: $n files, $bad different"
  [ "$bad" -eq 0 ] && [ "$n" -gt 0 ] || fail=1 # 0 files = the run did not happen
}

echo "(a) --polymesh, base vs this binary"
for x in "cavity 60" "channel 60" "duct 40" "cavity_lim 60" "mrf2 40" "cavity_skew 60" "sa_skew 60" "sst_plate 60" "sst_skew 60"; do
  set -- $x
  for fp in fp32 fp64; do
    "$BASE" flow --polymesh "$R/tests/data/fv/$1" --max-iters "$2" --fv-precision $fp --out "$W/b_$1_$fp" --quiet > /dev/null 2>&1
    "$B" flow --polymesh "$R/tests/data/fv/$1" --max-iters "$2" --fv-precision $fp --out "$W/n_$1_$fp" --quiet > /dev/null 2>&1
    noms "$W/b_$1_${fp}_stats.csv" > "$W/b.csv"; noms "$W/n_$1_${fp}_stats.csv" > "$W/n.csv"
    same "$1 $fp stats" "$W/b.csv" "$W/n.csv"
    sametree "$1 $fp output" "$W/b_$1_$fp" "$W/n_$1_$fp"
  done
done

echo "(a) structured flows, base vs this binary"
i=0
while IFS= read -r a; do
  i=$((i + 1))
  "$BASE" flow $a --out "$W/sb$i" --emit stats --quiet > /dev/null 2>&1
  "$B" flow $a --out "$W/sn$i" --emit stats --quiet > /dev/null 2>&1
  awk -F, -v OFS=, '{ print }' "$W/sb${i}_stats.csv" > /dev/null 2>&1
  same "flow $a" "$W/sb${i}_stats.csv" "$W/sn${i}_stats.csv"
done <<'EOF'
--body cylinder --n 128 --dr 8 --re 60 --ft 0.2
--body cylinder --n 128 --dr 8 --re 60 --ft 0.2 --physics compressible
--ogrid --body cylinder --physics compressible --n 64 --ny 34 --re 60 --ft 0.05
EOF

echo "(b) --solver simple vs --export + --polymesh"
i=0
while IFS= read -r a; do
  i=$((i + 1))
  "$B" flow --solver simple $a --fv-precision fp64 --out "$W/run$i" > /dev/null 2>&1
  "$B" flow --solver simple $a --export "$W/case$i" > /dev/null 2>&1
  "$B" flow --polymesh "$W/case$i" --fv-precision fp64 --out "$W/fromcase$i" > /dev/null 2>&1
  noms "$W/run${i}_stats.csv" > "$W/a.csv"; noms "$W/fromcase${i}_stats.csv" > "$W/b.csv"
  same "$a: stats" "$W/a.csv" "$W/b.csv"
  sametree "$a: results" "$W/fromcase$i" "$W/run$i"
done <<'EOF'
--ogrid --body cylinder --n 64 --ny 34 --re 40 --max-iters 40
--ogrid --body sphere --nz 1 --n 32 --ny 26 --re 100 --max-iters 40
--ogrid --body sphere --nz 8 --n 16 --ny 14 --re 100 --max-iters 20
--ogrid --body cylinder --n 48 --ny 26 --rans sst-2003m --re 1000 --max-iters 30
--body none --far periodic --noslip --re-tau 180 --n 4 --ny 32 --stretch 2 --rans sa-neg --max-iters 60
--body none --n 64 --ny 64 --focal 2,4,2,4,0,0 --focal-ratio 2 --noslip --re 100 --max-iters 40
EOF
echo "fail=$fail"
exit $fail
