#!/usr/bin/env bash
# Runs the OpenFOAM v2606 references of tools/fv/make_cases.py (docs/fv.md).
# Submit through the job server with the OpenFOAM environment sourced:
#   source /opt/OpenFOAM/OpenFOAM-v2606/etc/bashrc
#   build/simjob submit --cpu -t 1500 -m 12000 --inherit-env --wait -- \
#       /usr/bin/bash tools/fv/run_foam.sh ROOT CASE...
# CASE: cavity | cavity_skew | channel | duct | cavity_lim | mrf2
#       disc (blockMesh + topoSet + simpleFoam)
#       sst_plate | sst_skew | sa_skew | plate (tools/fv/make_ras_cases.py)
#       (serial blockMesh + simpleFoam)
#       pcavity | pcavity_skew | pchannel | pduct | psst_plate | psst_skew
#       (tools/fv/make_pimple_cases.py: serial blockMesh + pimpleFoam)
#       mspin | mduct | msst (tools/fv/make_motion_cases.py: the same, moving mesh)
#       aplanar | amatch | amatch_int | acyl | amix2d | amix3d
#       (tools/fv/make_ami_flow_cases.py: the same, cyclicAMI)
#       apartial | amixtilt (the same, faces without overlap: FOAM_SIGFPE off)
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
    pcavity|pcavity_skew|pchannel|pduct|psst_plate|psst_skew|mspin|mduct|msst|aplanar|amatch|amatch_int|acyl|amix2d|amix3d) # make_pimple_cases.py, make_motion_cases.py, make_ami_flow_cases.py (serial pimpleFoam)
        cd "$root/$c"
        blockMesh > log.blockMesh 2>&1
        checkMesh > log.checkMesh 2>&1 || true
        pimpleFoam > log.pimpleFoam 2>&1
        ;;
    apartial|amixtilt) # cyclicAMI faces without overlap: 1 / UEqn.A() divides by the patch value 0 there, so
        # pimpleFoam runs these only with floating-point trapping off (the patch values are not read).
        cd "$root/$c"
        blockMesh > log.blockMesh 2>&1
        checkMesh > log.checkMesh 2>&1 || true
        FOAM_SIGFPE=false pimpleFoam > log.pimpleFoam 2>&1
        ;;
    disc)
        cd "$root/$c"
        blockMesh > log.blockMesh 2>&1
        topoSet > log.topoSet 2>&1
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
