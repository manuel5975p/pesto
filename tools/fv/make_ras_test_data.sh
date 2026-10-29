#!/usr/bin/env bash
# Copies the OpenFOAM v2606 RAS references (tools/fv/make_ras_cases.py +
# tools/fv/run_foam.sh, already run under ROOT, plus `checkMesh -writeFields
# '(wallDistance)' -time 0` for ref/wallDistance) into tests/data/fv/<case>/:
# the case (0/, constant/, system/), ref/100 and ref/<last> (U, p, phi and the
# model's k, omega | nuTilda, nut), ref/wallDistance, and ref/residuals.csv
# (initial residuals per iteration from log.simpleFoam: U components, the
# first p, then the turbulence solves in log order). Host only.
#   tools/fv/make_ras_test_data.sh runs/fv/ras [CASE...]
set -euo pipefail
root=$1
shift
cases=${*:-sst_plate sst_skew sa_skew}
here=$(cd "$(dirname "$0")/../.." && pwd)
for c in $cases; do
    src=$root/$c
    dst=$here/tests/data/fv/$c
    rm -rf "$dst"
    mkdir -p "$dst/constant/polyMesh" "$dst/system" "$dst/0" "$dst/ref"
    for f in U p k omega nut nuTilda; do [ -f "$src/0/$f" ] && cp "$src/0/$f" "$dst/0/"; done
    cp "$src"/constant/polyMesh/{boundary,faces,neighbour,owner,points} "$dst/constant/polyMesh/"
    cp "$src"/constant/{transportProperties,momentumTransport,turbulenceProperties} "$dst/constant/"
    cp "$src"/system/{controlDict,fvSchemes,fvSolution,blockMeshDict} "$dst/system/"
    cp "$src/0/wallDistance" "$dst/ref/wallDistance"
    last=$(ls -d "$src"/[0-9]* | xargs -n1 basename | sort -n | tail -1)
    for t in 100 $last; do
        mkdir -p "$dst/ref/$t"
        for f in U p phi k omega nut nuTilda; do [ -f "$src/$t/$f" ] && cp "$src/$t/$f" "$dst/ref/$t/"; done
    done
    awk '
        /^Time = / { if (it) print line; it = $3; line = it; seenp = 0 }
        /Solving for U[xyz]/ { split($0, a, "Initial residual = "); split(a[2], b, ","); line = line "," b[1] }
        /Solving for p/ { if (!seenp) { split($0, a, "Initial residual = "); split(a[2], b, ","); line = line "," b[1]; seenp = 1 } }
        /Solving for (omega|k|nuTilda),/ { split($0, a, "Initial residual = "); split(a[2], b, ","); line = line "," b[1] }
        END { if (it) print line }
    ' "$src/log.simpleFoam" > "$dst/ref/residuals.csv"
    echo "wrote $dst ($(du -sh "$dst" | cut -f1))"
done
