#!/usr/bin/env bash
# Existing paths unmoved by the compressible FV family (docs/fvcomp.md): byte
# compares stats and fields of a pre-change binary vs this tree's, one simjob
# per run. Usage: comp_bit_identity.sh BASE_BINARY NEW_BINARY [OUTDIR]
set -u
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
BASE=$1; NEW=$2; OUT=${3:-/tmp/dns/bit}
mkdir -p "$OUT"
fails=0
run() { # tag -- args
    local tag=$1; shift
    for w in base new; do
        local exe=$BASE; [ $w = new ] && exe=$NEW
        rm -rf "$OUT/$tag.$w"*
        "$ROOT/build/simjob" submit --gpu -t 295 -m 8192 --wait --label "bit-$tag-$w" -- "$exe" "$@" --out "$OUT/$tag.$w" \
            > "$OUT/$tag.$w.job" 2>&1 || { echo "$tag $w: job failed"; cat "$OUT/$tag.$w.job"; fails=$((fails + 1)); }
    done
    local bad=0 bytes=0
    for f in "$OUT/$tag.base"*; do
        case "$f" in *.job) continue;; esac
        local g=${f/$tag.base/$tag.new}
        [ -f "$f" ] && bytes=$((bytes + $(stat -c %s "$f")))
        if [[ "$f" == *.json ]]; then
            # wall time is not a result
            cmp <(sed -E 's/"msPerStep":[0-9.eE+-]+//' "$f") <(sed -E 's/"msPerStep":[0-9.eE+-]+//' "$g") > /dev/null ||
                { echo "  DIFF $f"; bad=1; }
        elif [ -d "$f" ]; then
            diff -r "$f" "$g" > /dev/null || { echo "  DIFF dir $f"; bad=1; }
        elif [[ "$f" == *_stats.csv ]] && head -3 "$f" | grep -q ',ms'; then
            # the FV path's ms column is wall time, not a result
            cmp <(sed 's/,[0-9.]*$//' "$f") <(sed 's/,[0-9.]*$//' "$g") > /dev/null || { echo "  DIFF $f"; bad=1; }
        else
            cmp "$f" "$g" > /dev/null || { echo "  DIFF $f"; bad=1; }
        fi
    done
    [ "$bytes" -gt 0 ] || { echo "  EMPTY $tag: nothing was compared"; bad=1; }
    [ $bad = 0 ] && echo "$tag: byte-identical ($(ls "$OUT/$tag.base"* | grep -vc '\.job') artefacts, $bytes bytes)"
    fails=$((fails + bad))
}
run comp flow --physics compressible --body cylinder --n 192 --ny 128 --re 100 --mach 0.2 --ft 0.3 --emit stats
run incomp flow --body cylinder --n 128 --ny 128 --re 100 --ft 0.2 --emit stats
run ogrid flow --physics compressible --ogrid --body cylinder --n 96 --ny 48 --re 100 --mach 0.2 --ft 0.2 --ogrid-trip 0.05 --emit stats
run rans flow --physics compressible --ogrid --body cylinder --n 96 --ny 48 --ogrid-beta 4 --re 1000 --mach 0.2 --ft 0.02 --rans sst-2003m --emit stats
run simple32 flow --polymesh "$ROOT/tests/data/fv/cavity" --max-iters 50
run simple64 flow --polymesh "$ROOT/tests/data/fv/channel" --max-iters 50 --fv-precision fp64
# The FV family with no species, no closure, no gravity.
run fvbox flow --physics compressible --solver fv --body none --n 64 --ny 64 --re 100 --mach 0.2 --ft 0.2 --far periodic --emit stats
run fvogrid40 flow --physics compressible --solver fv --ogrid --body cylinder --n 96 --ny 48 --re 40 --mach 0.2 --ft 0.2 --compensated --emit stats
run fvcomp flow --physics compressible --solver fv --ogrid --body cylinder --n 96 --ny 48 --re 100 --mach 0.2 --ft 0.2 --ogrid-trip 0.05 --emit stats
run fvles flow --physics compressible --solver fv --ogrid --body cylinder --n 96 --ny 48 --re 1000 --mach 0.2 --ft 0.1 --les 0.17 --emit stats
run fvcpair flow --physics compressible --solver fv --ogrid --body cylinder --n 96 --ny 48 --re 100 --mach 0.2 --ft 0.2 --compensated --emit stats
run fv64 flow --physics compressible --solver fv --ogrid --body cylinder --n 96 --ny 48 --re 100 --mach 0.2 --ft 0.1 --fv-precision fp64 --emit stats
run fvpoly flow --physics compressible --polymesh "$ROOT/tests/data/mesh/cyl2d_snappy" --bc "$ROOT/examples/fvcomp_cyl2d_snappy.toml" --re 40 --mach 0.2 --tt 0.2 --emit stats
run fvpoly64 flow --physics compressible --polymesh "$ROOT/tests/data/mesh/cyl2d_snappy" --bc "$ROOT/examples/fvcomp_cyl2d_snappy.toml" --re 40 --mach 0.2 --tt 0.5 --ramp 0.1 --fv-precision fp64 --emit stats
run fvincomp flow --solver fv --ogrid --body cylinder --n 96 --ny 48 --re 100 --ft 0.2 --emit stats
run fvtgv tgv --solver fv --n 32 --nz 1 --re 100 --ft 4 --emit stats # 0.2 writes no row
# The FV flame at nY 1: the species kernels' other instantiation (dns flame --mode diffusion) stays this binary's.
run fvflame flame --solver fv --cells-per-delta 8 --tt 1 --fv-precision fp64 --emit stats
run fvflamec flame --solver fv --cells-per-delta 8 --tt 1 --compensated --emit stats
# The structured flames (their IC moved into premixedPoint), conv (its IC into pointwise helpers).
run flame flame --cells-per-delta 8 --tt 1 --emit stats
run bunsen flame --mode bunsen --cells-per-delta 6 --tt 0.3 --emit stats
run lowmach flame --physics lowmach --cells-per-delta 8 --tt 1 --emit stats
run convc conv --n 48 --ny 24 --nz 1 --rayleigh 5e4 --tt 1 --emit stats
run convb conv --physics incompressible --n 48 --ny 24 --nz 1 --rayleigh 5e4 --tt 2 --emit stats
run convcl conv --physics incompressible --sides closed --n 24 --ny 12 --nz 8 --rayleigh 5e4 --tt 1 --emit stats
echo "failures: $fails"
exit $fails
