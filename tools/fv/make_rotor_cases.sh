#!/usr/bin/env bash
# Builds the rotor2 MRF parity cases of the FV path (docs/fv.md, phase MRF)
# from the parity suite's simpleFoam + MRF cases (runs/parity/B/rotor2/foam_L*,
# tools/parity): the SAME polyMesh and cellZone, laminar Re 220, J 0.63,
# n = 1 rev/s, D = 1, rho 1 (CT = -Fx, CQ = -Mx). Host only.
#   tools/fv/make_rotor_cases.sh DEST [ITERS]
# writes DEST/rotor2_L{1,2}_{unlim,lim}/ (0/, constant/, system/):
#   unlim  Gauss linear corrected laplacian / snGrad, grad(U) Gauss linear
#          (the suite's foam_L*_unlim schemes)
#   lim    Gauss linear limited corrected 0.33, grad(U) cellLimited Gauss
#          linear 1 (the suite's foam_L* schemes)
# Cold start from t = 0, ITERS (default 1500) SIMPLEC iterations, written once
# at the end; decomposeParDict np 6 (the suite's rank count).
set -euo pipefail
dest=$1
iters=${2:-1500}
here=$(cd "$(dirname "$0")/../.." && pwd)
src=$here/runs/parity/B/rotor2
for L in L1 L2; do
    for v in unlim lim; do
        s=$src/foam_${L}
        d=$dest/rotor2_${L}_$v
        rm -rf "$d"
        mkdir -p "$d/constant" "$d/system" "$d/0"
        cp -r "$s/constant/polyMesh" "$d/constant/"
        rm -rf "$d/constant/polyMesh/sets"
        cp "$s"/constant/{MRFProperties,transportProperties,momentumTransport,turbulenceProperties} "$d/constant/"
        cp "$s"/0/U "$s"/0/p "$d/0/"
        cp "$s"/system/{fvSolution,decomposeParDict} "$d/system/"
        sed -e "s/^endTime .*/endTime $iters;/" -e "s/^writeInterval .*/writeInterval $iters;/" \
            -e "s/^startFrom .*/startFrom startTime;/" "$s/system/controlDict" > "$d/system/controlDict"
        if [ $v = unlim ]; then
            cp "$src/foam_${L}_unlim/system/fvSchemes" "$d/system/"
        else
            cp "$s/system/fvSchemes" "$d/system/"
        fi
        echo "wrote $d"
    done
done
