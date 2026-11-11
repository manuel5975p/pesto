#!/usr/bin/env bash
# Copies the OpenFOAM v2606 IF1 references (tools/fv/make_cases.py +
# tools/fv/run_foam.sh, already run under ROOT) into tests/data/fv/<case>/:
# the case (0/, constant/, system/), the reference time directories ref/100
# and ref/<last> (U, p, phi) and ref/residuals.csv (initial residuals per
# iteration from log.simpleFoam). Host only, no OpenFOAM needed.
#   tools/fv/make_test_data.sh runs/fv/if [CASE...]
# nozzle_A nozzle_B nozzle_W nozzle_G (tools/fv/nozzle_cases.py ROOT A B W G, rhoCentralFoam, 80x16), nozzle_QA
# nozzle_RT (ROOT QA RT: one row, nc 200 and the react twin 400, masterplan_combnozzle P2): the mesh, 0/, bc.toml,
# ref/<last>/{p,T,U,rho}.
set -euo pipefail
root=$1
shift
cases=${*:-cavity cavity_skew channel duct cavity_lim mrf2}
here=$(cd "$(dirname "$0")/../.." && pwd)
for c in $cases; do
    dst=$here/tests/data/fv/$c
    if [[ $c == nozzle_* ]]; then
        src=$root/${c#nozzle_}
        rm -rf "$dst"
        mkdir -p "$dst/constant/polyMesh" "$dst/0"
        cp "$src"/0/{p,T,U} "$dst/0/"
        cp "$src"/bc.toml "$dst/"
        cp "$src"/constant/polyMesh/{boundary,faces,neighbour,owner,points} "$dst/constant/polyMesh/"
        last=$(ls -d "$src"/[0-9]* | xargs -n1 basename | sort -n | tail -1)
        mkdir -p "$dst/ref/$last"
        cp "$src/$last"/{p,T,U,rho} "$dst/ref/$last/"
        echo "wrote $dst ($(du -sh "$dst" | cut -f1))"
        continue
    fi
    src=$root/$c
    rm -rf "$dst"
    mkdir -p "$dst/constant/polyMesh" "$dst/system" "$dst/0" "$dst/ref"
    cp "$src"/0/U "$src"/0/p "$dst/0/"
    cp "$src"/constant/polyMesh/{boundary,faces,neighbour,owner,points} "$dst/constant/polyMesh/"
    cp "$src"/constant/{transportProperties,momentumTransport,turbulenceProperties} "$dst/constant/"
    [ -f "$src/constant/polyMesh/cellZones" ] && cp "$src/constant/polyMesh/cellZones" "$dst/constant/polyMesh/"
    [ -f "$src/constant/MRFProperties" ] && cp "$src/constant/MRFProperties" "$dst/constant/"
    cp "$src"/system/{controlDict,fvSchemes,fvSolution,blockMeshDict} "$dst/system/"
    for f in fvOptions topoSetDict; do
        [ -f "$src/system/$f" ] && cp "$src/system/$f" "$dst/system/"
    done
    # function-object and fvOptions output files (forces, forceCoeffs,
    # actuationDiskSource): the header and the rows of iterations 1..120
    # (the transient; check-in size)
    if [ -d "$src/postProcessing" ]; then
        (cd "$src" && find postProcessing -name '*.dat' | while read -r f; do
            mkdir -p "$dst/ref/$(dirname "$f")" && awk '/^#/ || $1 <= 120' "$f" > "$dst/ref/$f"
        done)
    fi
    last=$(ls -d "$src"/[0-9]* | xargs -n1 basename | sort -n | tail -1)
    for t in 100 $last; do
        mkdir -p "$dst/ref/$t"
        cp "$src/$t"/{U,p,phi} "$dst/ref/$t/"
    done
    # iter,<field>... initial residuals in log order (first p solve only)
    awk '
        /^Time = / { if (it) print line; it = $3; line = it; seenp = 0 }
        /Solving for U[xyz]/ { split($0, a, "Initial residual = "); split(a[2], b, ","); line = line "," b[1] }
        /Solving for p/ { if (!seenp) { split($0, a, "Initial residual = "); split(a[2], b, ","); line = line "," b[1]; seenp = 1 } }
        END { if (it) print line }
    ' "$src/log.simpleFoam" > "$dst/ref/residuals.csv"
    echo "wrote $dst ($(du -sh "$dst" | cut -f1))"
done
