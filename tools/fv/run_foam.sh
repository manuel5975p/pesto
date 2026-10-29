#!/usr/bin/env bash
# Runs the OpenFOAM v2606 references of tools/fv/make_cases.py (docs/fv.md).
# Submit through the job server with the OpenFOAM environment sourced:
#   source /opt/OpenFOAM/OpenFOAM-v2606/etc/bashrc
#   build/simjob submit --cpu -t 1500 -m 12000 --inherit-env --wait -- \
#       /usr/bin/bash tools/fv/run_foam.sh ROOT CASE...
# CASE: cavity | cavity_skew | channel | duct | cavity_lim | mrf2
#       sst_plate | sst_skew | sa_skew | plate (tools/fv/make_ras_cases.py)
#       (serial blockMesh + simpleFoam)
#       sphere-mesh (blockMesh + snappyHexMesh + checkMesh, serial)
#       sphere (decomposePar + simpleFoam np 12 + reconstructPar)
set -euo pipefail
root=$(cd "$1" && pwd); shift
for c in "$@"; do
    case $c in
    cavity|cavity_skew|channel|duct|cavity_lim|mrf2|sst_plate|sst_skew|sa_skew|plate)
        cd "$root/$c"
        blockMesh > log.blockMesh 2>&1
        checkMesh > log.checkMesh 2>&1 || true
        simpleFoam > log.simpleFoam 2>&1
        ;;
    sphere-mesh)
        cd "$root/sphere"
        blockMesh > log.blockMesh 2>&1
        snappyHexMesh -overwrite > log.snappyHexMesh 2>&1
        checkMesh > log.checkMesh 2>&1 || true
        ;;
    sphere)
        cd "$root/sphere"
        decomposePar -force > log.decomposePar 2>&1
        /usr/bin/mpirun -np 12 --bind-to core --map-by core simpleFoam -parallel > log.simpleFoam 2>&1
        reconstructPar -latestTime > log.reconstructPar 2>&1
        ;;
    *) echo "unknown case $c" >&2; exit 2 ;;
    esac
    echo "done $c"
done
