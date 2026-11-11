#!/usr/bin/env bash
# Copies the OpenFOAM v2606 pimpleFoam references (tools/fv/make_pimple_cases.py
# + tools/fv/run_foam.sh, already run under ROOT) into tests/data/fv/<case>/:
# the case (0/, constant/, system/), TWO reference time directories (the first
# and the last written: U, p, phi and the model's k, omega, nut), the forces
# files, and ref/log.csv: one row per time step from log.pimpleFoam,
#   time, max Courant number (the one deltaT was set from), deltaT (0 when the
#   log prints none: fixed step), then every "Initial residual" of the step in
#   log order (outer correctors: [moving mesh: pcorr solves,] U components, p
#   solves, omega, k).
# Moving-mesh cases (tools/fv/make_motion_cases.py: mspin mduct msst) also get
# constant/dynamicMeshDict, polyMesh/cellZones, and per reference time Uf,
# meshPhi and polyMesh/points; ref/meshco.csv when the log prints mesh Courant
# numbers.
# Host only, no OpenFOAM needed.
#   tools/fv/make_pimple_test_data.sh runs/pimple/foam [CASE...]
#   tools/fv/make_pimple_test_data.sh runs/motion/foam mspin mduct msst
set -euo pipefail
root=$1
shift
cases=${*:-pcavity pcavity_skew pchannel pduct psst_plate psst_skew}
here=$(cd "$(dirname "$0")/../.." && pwd)
for c in $cases; do
    src=$root/$c
    dst=$here/tests/data/fv/$c
    rm -rf "$dst"
    mkdir -p "$dst/constant/polyMesh" "$dst/system" "$dst/0" "$dst/ref"
    for f in U p k omega nut; do [ -f "$src/0/$f" ] && cp "$src/0/$f" "$dst/0/"; done
    cp "$src"/constant/polyMesh/{boundary,faces,neighbour,owner,points} "$dst/constant/polyMesh/"
    [ -f "$src/constant/polyMesh/cellZones" ] && cp "$src/constant/polyMesh/cellZones" "$dst/constant/polyMesh/"
    [ -f "$src/constant/dynamicMeshDict" ] && cp "$src/constant/dynamicMeshDict" "$dst/constant/"
    cp "$src"/constant/{transportProperties,momentumTransport,turbulenceProperties} "$dst/constant/"
    cp "$src"/system/{controlDict,fvSchemes,fvSolution,blockMeshDict} "$dst/system/"
    if [ -d "$src/postProcessing" ]; then
        (cd "$src" && find postProcessing -name '*.dat' | while read -r f; do
            mkdir -p "$dst/ref/$(dirname "$f")" && cp "$f" "$dst/ref/$f"
        done)
    fi
    times=$(ls -d "$src"/[0-9]* | xargs -n1 basename | grep -vx 0 | sort -g)
    first=$(echo "$times" | head -1)
    last=$(echo "$times" | tail -1)
    for t in $first $last; do
        mkdir -p "$dst/ref/$t"
        for f in U p phi k omega nut Uf meshPhi; do [ -f "$src/$t/$f" ] && cp "$src/$t/$f" "$dst/ref/$t/"; done
        if [ -f "$src/$t/polyMesh/points" ]; then # a moving mesh's points at that time
            mkdir -p "$dst/ref/$t/polyMesh" && cp "$src/$t/polyMesh/points" "$dst/ref/$t/polyMesh/"
        fi
    done
    # moving mesh with checkMeshCourantNo: the max mesh Courant number of every step
    if grep -q '^Mesh Courant Number' "$src/log.pimpleFoam"; then
        awk '/^Mesh Courant Number mean:/ { print $NF }' "$src/log.pimpleFoam" > "$dst/ref/meshco.csv"
    fi
    awk '
        /^Starting time loop/ { on = 1 }
        !on { next }
        /^Courant Number mean:/ { co = $NF; dt = 0 }
        /^deltaT = / { dt = $3 }
        /^Time = / { if (line != "") print line; line = $3 "," co "," dt }
        /Solving for / { split($0, a, "Initial residual = "); split(a[2], b, ","); line = line "," b[1] }
        END { if (line != "") print line }
    ' "$src/log.pimpleFoam" > "$dst/ref/log.csv"
    echo "wrote $dst ($(du -sh "$dst" | cut -f1)), reference times $first $last"
done
