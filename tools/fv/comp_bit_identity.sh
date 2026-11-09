#!/usr/bin/env bash
# Existing paths unmoved by the compressible FV family (docs/fvcomp.md): byte
# compares stats and fields of a pre-change binary vs this tree's, one simjob
# per run. Usage: comp_bit_identity.sh BASE_BINARY NEW_BINARY [OUTDIR]
# ONLY=REGEX runs the tags it matches (tools/fv: --shock, masterplan_shockflux).
set -u
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
BASE=$1; NEW=$2; OUT=${3:-/tmp/dns/bit}
mkdir -p "$OUT"
fails=0
run() { # tag -- args
    local tag=$1; shift
    [[ -n "${ONLY:-}" && ! $tag =~ $ONLY ]] && return
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
# The species kernels' LES / TFLES instantiations sit beside these (docs/fvcomp.md "Species LES and TFLES").
run fvflame3d flame --solver fv --nz 8 --ny 8 --cells-per-delta 6 --tt 0.3 --emit stats
run fvjet flame --mode diffusion --solver fv --tt 1 --fv-precision fp64 --emit stats
# M5a: the FV jet's rows and probe grew the candle's branches (g 0 = the jet); the structured candle's validator.
run fvjet32 flame --mode diffusion --solver fv --tt 1 --emit stats
run candle flame --mode candle --cells-per-unit 4 --tt 0.5 --emit stats
# The gravity passes (dns conv, compressible FV, compensated by default): MC's inventory moved their refusals.
run fvconv conv --solver fv --rayleigh 2e4 --ny 16 --tt 0.5 --emit stats
# The structured flame LES + TFLES (its label and dry-run text gained FV branches).
run flameles flame --nz 8 --ny 8 --cells-per-delta 6 --les 1.35 --sgs sigma --tfles 2 --tt 0.3 --emit stats
# The structured flames (their IC moved into premixedPoint), conv (its IC into pointwise helpers).
run flame flame --cells-per-delta 8 --tt 1 --emit stats
run bunsen flame --mode bunsen --cells-per-delta 6 --tt 0.3 --emit stats
run lowmach flame --physics lowmach --cells-per-delta 8 --tt 1 --emit stats
# The low-Mach FV planar flame (docs/fvincomp.md "Low-Mach"): C0 on the box, both precisions,
# and a sheared polyMesh (M3: two reactants, burner rows are other instantiations).
# The structured low-Mach burner (its IC through lmBunsenPoint's cell rule, M3), C0 and C2.
run lmbunsen flame --physics lowmach --mode bunsen --cells-per-delta 4 --tt 0.5 --emit stats
run lmbunsen2 flame --physics lowmach --mode bunsen --equivalence 1.3 --expansion 6.88 --pilot-width 1.5 \
    --cells-per-delta 4 --tt 0.5 --emit stats
run fvlm flame --physics lowmach --solver fv --cells-per-delta 8 --tt 1 --emit stats
run fvlm64 flame --physics lowmach --solver fv --cells-per-delta 8 --tt 1 --fv-precision fp64 --emit stats
python3 "$ROOT/tools/chem/fv_flame_mesh.py" --nx 180 --ny 8 --lx 36.0371 --ly 1.60165 --shear 30 --lowmach \
    --out "$OUT/lmmesh" > /dev/null
run fvlmpoly flame --physics lowmach --polymesh "$OUT/lmmesh" --bc "$OUT/lmmesh/bc.toml" --upstream 12 --tt 1 \
    --fv-precision fp64 --emit stats
# M5: the structured --turb box (its IC window through turbWindow), the low-Mach 3D planar
# flame on both solvers (kLmMomT's Turb dispatch off; the FV runner's turb hooks off).
run flameturb flame --turb 1 --turb-length 1 --cells-per-delta 4 --ny 24 --nz 24 --upstream 6 --downstream 4 \
    --tt 0.3 --emit stats
run lm3d flame --physics lowmach --nz 8 --ny 8 --cells-per-delta 6 --tt 0.3 --emit stats
run fvlm3d flame --physics lowmach --solver fv --nz 8 --ny 8 --cells-per-delta 6 --tt 0.3 --emit stats
run convc conv --n 48 --ny 24 --nz 1 --rayleigh 5e4 --tt 1 --emit stats
run convb conv --physics incompressible --n 48 --ny 24 --nz 1 --rayleigh 5e4 --tt 2 --emit stats
run convcl conv --physics incompressible --sides closed --n 24 --ny 12 --nz 8 --rayleigh 5e4 --tt 1 --emit stats
# mci: the Boussinesq FV stage (the force passes now gated on betaG, kFaceT's Les instantiation beside).
run fvconvb conv --solver fv --physics incompressible --n 48 --ny 24 --nz 1 --rayleigh 5e4 --tt 2 --emit stats
run fvconvbr conv --solver fv --physics incompressible --n 32 --ny 16 --nz 1 --rayleigh 5e4 --tt 2 --mesh-perturb 0.2 \
    --mesh-perturb-kind random --fv-precision fp64 --emit stats
run fvconvb3 conv --solver fv --physics incompressible --n 16 --ny 8 --nz 16 --rayleigh 1e5 --aspect 2 --tt 2 --emit stats
# --shock (masterplan_shockflux): the chamber, a sigma 10 flame, the wick, the channels: stage() gained a branch.
run chamber flame --mode chamber --tt 0.5 --emit stats --emit bin
run flamesig10 flame --expansion 10 --cells-per-delta 4 --tt 1 --emit stats
run wick flame --mode candle --wick --tt 0.2 --emit stats
run chan3d channel --n 32 --ny 32 --nz 16 --tt 0.05 --emit stats
run chanrans channel --n 16 --ny 32 --nz 1 --rans sa-neg --tt 0.05 --emit stats
# masterplan_cdnozzle: the rows the nozzle's edits cross (FV RANS, LTS, --shock structured).
run fvrans flow --physics compressible --solver fv --ogrid --body cylinder --n 96 --ny 48 --re 1000 --mach 0.2 --ft 0.02 --rans sst-2003m --emit stats
run fvsteady flow --physics compressible --solver fv --ogrid --body cylinder --n 96 --ny 48 --re 40 --mach 0.2 --steady 1e-3 --steady-steps 200 --compensated --emit stats
run fvpolysteady flow --physics compressible --polymesh "$ROOT/tests/data/mesh/cyl2d_snappy" --bc "$ROOT/examples/fvcomp_cyl2d_snappy.toml" --re 40 --mach 0.2 --steady 1e-3 --steady-steps 200 --emit stats
run shockhll flow --physics compressible --body none --n 96 --ny 64 --re 100 --mach 0.3 --ft 0.2 --far nscbc --shock hll --emit stats
run shocksense flow --physics compressible --body cylinder --n 192 --ny 128 --re 100 --mach 0.2 --ft 0.3 --far nscbc --shock sense --emit stats
run shockflame flame --expansion 10 --cells-per-delta 4 --tt 1 --shock hll --emit stats
run shockchamber flame --mode chamber --tt 0.5 --shock hll --emit stats
# masterplan_combnozzle: the polyMesh nozzles under --shock (the ledger, [thrust] absent) and the
# FV flame polyMesh route (--gamma absent, --shock absent).
run fvnozA flow --physics compressible --polymesh "$ROOT/tests/data/fv/nozzle_A" --bc "$ROOT/tests/data/fv/nozzle_A/bc.toml" --shock hll --inviscid --mach 1 --ramp 40 --tt 20 --emit stats
run fvnozW flow --physics compressible --polymesh "$ROOT/tests/data/fv/nozzle_W" --bc "$ROOT/tests/data/fv/nozzle_W/bc.toml" --shock hll --inviscid --mach 1 --ramp 40 --tt 20 --emit stats
FLAMEMESH="$OUT/flmesh"
python3 "$ROOT/tools/chem/fv_flame_mesh.py" --nx 197 --ny 4 --lx 38.6532 --ly 3.41688 --out "$FLAMEMESH" > /dev/null
run fvflpoly flame --polymesh "$FLAMEMESH" --bc "$FLAMEMESH/bc.toml" --upstream 14.1335 --inflow-speed 0.98 --tt 2 --fv-precision fp64 --emit stats
echo "failures: $fails"
exit $fails
