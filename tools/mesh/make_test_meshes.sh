#!/bin/bash
# Regenerates the checked-in polyMesh test data (tests/data/mesh/*) with the
# OpenFOAM v2606 mesh tools, each with its own log.checkMesh -- the reference
# test_facemesh compares FaceMesh geometry against. Mesh tools only, no solver.
# Run it once, through the job server (CLAUDE.md):
#
#   build/simjob submit --cpu -t 300 -m 4096 --wait -- /bin/bash \
#       tools/mesh/make_test_meshes.sh [WORKDIR]
#
#   box_cyclic      blockMesh 4x3x2 on 1 x 0.75 x 0.5, x/z cyclic, y walls (ASCII)
#   box_cyclic_bin  the same through foamFormatConvert (binary)
#   box_refined     8x6x4 cyclic on every axis + refineMesh of an interior box
#   cyl2d_snappy    snappyHexMesh cylinder (levels 1-2, 3 layers) extruded to
#                   one layer, `empty` front/back
set -e
here=$(cd "$(dirname "$0")" && pwd)
root=$(cd "$here/../.." && pwd)
dicts=$here/dicts
out=$root/tests/data/mesh
work=${1:-$(mktemp -d)}
mkdir -p "$work"

if [ -z "$WM_PROJECT_DIR" ]; then
    set +e
    . /opt/OpenFOAM/OpenFOAM-v2606/etc/bashrc
    set -e
fi

# newcase DIR: a case skeleton with the common system files.
newcase() {
    rm -rf "$1"
    mkdir -p "$1/system" "$1/constant"
    cp "$dicts/common/controlDict" "$dicts/common/fvSchemes" \
       "$dicts/common/fvSolution" "$1/system/"
}

# run CASE TOOL ARGS...: TOOL in CASE, log in CASE/log.TOOL; stops on failure.
run() {
    local c=$1 t=$2
    shift 2
    (cd "$c" && "$t" "$@" > "log.$t" 2>&1) || {
        echo "FAILED: $t in $c (tail of log.$t):"
        tail -30 "$c/log.$t"
        exit 1
    }
}

# keep CASE NAME: copy the polyMesh files and the checkMesh log to the data dir.
keep() {
    local c=$1 n=$2
    rm -rf "${out:?}/$n"
    mkdir -p "$out/$n/constant/polyMesh"
    for f in points faces owner neighbour boundary cellZones; do
        [ -f "$c/constant/polyMesh/$f" ] && cp "$c/constant/polyMesh/$f" "$out/$n/constant/polyMesh/"
    done
    cp "$c/log.checkMesh" "$out/$n/log.checkMesh"
}

# --- box_cyclic / box_cyclic_bin --------------------------------------------
c=$work/box_cyclic
newcase "$c"
cp "$dicts/box_cyclic/blockMeshDict" "$c/system/"
run "$c" blockMesh
run "$c" checkMesh
keep "$c" box_cyclic

b=$work/box_cyclic_bin
rm -rf "$b"
cp -r "$c" "$b"
foamDictionary -entry writeFormat -set binary "$b/system/controlDict" > /dev/null
run "$b" foamFormatConvert -constant
run "$b" checkMesh
keep "$b" box_cyclic_bin

# --- box_refined --------------------------------------------------------------
c=$work/box_refined
newcase "$c"
cp "$dicts/box_refined/blockMeshDict" "$dicts/box_refined/topoSetDict" \
   "$dicts/box_refined/refineMeshDict" "$c/system/"
run "$c" blockMesh
run "$c" topoSet
run "$c" refineMesh -overwrite
run "$c" checkMesh
keep "$c" box_refined

# --- cyl2d_snappy -------------------------------------------------------------
s=$work/cyl2d/snappy
newcase "$s"
cp "$dicts/cyl2d_snappy/blockMeshDict" "$dicts/cyl2d_snappy/snappyHexMeshDict" "$s/system/"
run "$s" blockMesh
run "$s" snappyHexMesh -overwrite
e=$work/cyl2d/extrude
newcase "$e"
cp "$dicts/cyl2d_extrude/extrudeMeshDict" "$dicts/cyl2d_extrude/createPatchDict" "$e/system/"
run "$e" extrudeMesh
run "$e" createPatch -overwrite
run "$e" checkMesh
keep "$e" cyl2d_snappy

echo "test meshes written to $out (work: $work)"
du -sh "$out"
