#!/usr/bin/env bash
# The OpenFOAM v2606 references of tools/fv/make_ami_cases.py (stage 3c,
# dns/fvami): serial blockMesh + checkMesh + foamAmiDump per case and rotor
# angle. Submit through the job server with the OpenFOAM environment sourced:
#   source /opt/OpenFOAM/OpenFOAM-v2606/etc/bashrc
#   build/simjob submit --cpu -t 600 -m 8000 --inherit-env --wait -- \
#       /usr/bin/bash tools/fv/run_ami_foam.sh ROOT DUMPBIN CASE...
# DUMPBIN: the built tools/fv/foamAmiDump (its header says how to build it).
# CASE: ami_planar | ami_match | ami_partial | ami_match_int | ami_cyl | ami_mixer2d | ami_mixer3d
# Writes ROOT/CASE/amiDump_<k>: k = 0 at rest (with -fv), k >= 1 the cellZone
# rotor turned about z by the k-th angle below (with -fv); the angle is in the
# dump. The last two angles of a rotating case put rotor vertices on stator
# vertices (to rounding) and 1e-9 rad past them.
set -euo pipefail
root=$(cd "$1" && pwd); shift
dump=$1; shift
for c in "$@"; do
    cd "$root/$c"
    blockMesh > log.blockMesh 2>&1
    checkMesh > log.checkMesh 2>&1 || true
    case $c in
    ami_match_int) echo "done $c"; continue ;;
    ami_cyl) angles="0.05 0.3 1.0 1.7 0.13 0.130000001" ;;
    ami_mixer2d|ami_mixer3d) angles="0.05 0.3 1.0 1.7 0.27453292519943295 0.27453292619943295" ;;
    *) angles="" ;;
    esac
    "$dump" -fv -out "$PWD/amiDump_0" > log.foamAmiDump_0 2>&1
    k=1
    for a in $angles; do
        "$dump" -fv -zone rotor -angle "$a" -out "$PWD/amiDump_$k" > "log.foamAmiDump_$k" 2>&1
        k=$((k + 1))
    done
    if [ "$c" = ami_cyl ]; then # the rotor tilted about x: its ends leave the stator (restart, uncovered)
        for a in 0.3 0.6; do
            "$dump" -fv -zone rotor -angle "$a" -axis '(1 0 0)' -origin '(0 0 0.2)' -out "$PWD/amiDump_$k" \
                > "log.foamAmiDump_$k" 2>&1
            k=$((k + 1))
        done
    fi
    echo "done $c"
done
