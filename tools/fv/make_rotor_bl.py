#!/usr/bin/env python3
"""The rotor2 L1 boundary-layer case of the FV path (docs/fv.md, phase BL,
gate BL-b): the parity suite's rotor2 L1 snappy mesh (runs/parity/meshes/
rotor2_L1: blockMesh + surfaceFeatureExtract + snappyHexMesh, cellZone
`rotor`) re-meshed WITH PRISM LAYERS on the blade (cheap addLayers settings,
relative first layer for y+ ~ 1 at the chosen nu), and a simpleFoam
kOmegaSST + MRF case on it (J 0.63, n 1 rev/s, D 1, rho 1: CT = -Fx,
CQ = -Mx). Writes DEST/ (0/, constant/, system/) and DEST/Allmesh,
DEST/Allrun (OpenFOAM v2606; run them through simjob --cpu --inherit-env).

usage: make_rotor_bl.py DEST [--nu NU] [--layers N] [--first F] [--expansion E]
                        [--iters N] [--np P] [--wall spalding|lowre]
                        [--medial R] [--feature DEG] [--medial-angle DEG] [--relax-tet]
                        [--buffer N]
"""
import argparse
import os
import re
import shutil

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(HERE, "..", ".."))
MESH = os.path.join(ROOT, "runs", "parity", "meshes", "rotor2_L1")
CASE = os.path.join(ROOT, "runs", "parity", "B", "rotor2", "foam_L1")

HDR = """FoamFile
{{
    version     2.0;
    format      ascii;
    class       {cls};
    object      {obj};
}}
"""


def put(root, rel, cls, obj, body):
    path = os.path.join(root, rel)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write(HDR.format(cls=cls, obj=obj) + body)


def layers_block(n, first, expansion, medial, feature, medialAngle, buffer, extra=""):
    return f"""addLayersControls
{{
    relativeSizes true;
    layers {{ propeller {{ nSurfaceLayers {n}; }} }}
    firstLayerThickness {first};
    expansionRatio {expansion};
    minThickness 0.005;
    nGrow 0;
    featureAngle {feature};
    slipFeatureAngle 30;
    nRelaxIter 5;
    nSmoothSurfaceNormals 1;
    nSmoothNormals 3;
    nSmoothThickness 10;
    maxFaceThicknessRatio 0.5;
    maxThicknessToMedialRatio {medial};
    minMedialAxisAngle {medialAngle};
    nBufferCellsNoExtrude {buffer};
    nLayerIter 50;
    nRelaxedIter 20;
{extra}}}
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dest")
    ap.add_argument("--nu", type=float, default=5e-4)
    ap.add_argument("--layers", type=int, default=8)
    ap.add_argument("--first", type=float, default=0.03)
    ap.add_argument("--expansion", type=float, default=1.3)
    ap.add_argument("--iters", type=int, default=800)
    ap.add_argument("--np", type=int, default=6)
    ap.add_argument("--wall", choices=["spalding", "lowre"], default="spalding")
    ap.add_argument("--medial", type=float, default=0.3)
    ap.add_argument("--feature", type=float, default=130)
    ap.add_argument("--medial-angle", type=float, default=90)
    ap.add_argument("--relax-tet", action="store_true")
    # snappy's layer truncation: with 0, a face ring around every unextruded
    # spot steps 1, 2, ... layers (on a blade: ~2 layers everywhere); -1 = none
    ap.add_argument("--buffer", type=int, default=0)
    ap.add_argument("--extra", default="", help="addLayersControls entries appended verbatim")
    # central (`bounded Gauss linear`, the laminar prop reference) diverges at
    # nu 2e-5 on this mesh (simpleFoam: it ~110); linearUpwind is the default
    ap.add_argument("--divU", default="bounded Gauss linearUpwind grad(U)")
    a = ap.parse_args()
    d = os.path.abspath(a.dest)
    if os.path.exists(d):
        shutil.rmtree(d)
    os.makedirs(os.path.join(d, "constant"))
    shutil.copytree(os.path.join(MESH, "constant", "triSurface"), os.path.join(d, "constant", "triSurface"))
    shutil.copytree(os.path.join(MESH, "system"), os.path.join(d, "system"))
    for f in ["topoSetDict.hubcheck"]:
        p = os.path.join(d, "system", f)
        if os.path.exists(p):
            os.remove(p)
    snappy = open(os.path.join(MESH, "system", "snappyHexMeshDict")).read()
    snappy = snappy.replace("addLayers       false;", "addLayers       true;")
    i = snappy.index("addLayersControls")
    j = snappy.index("meshQualityControls")
    snappy = snappy[:i] + layers_block(a.layers, a.first, a.expansion, a.medial, a.feature, a.medial_angle, a.buffer, a.extra) + "\n" + snappy[j:]
    if a.relax_tet:
        # layers only: the relaxed set (used after nRelaxedIter) accepts any tet quality
        snappy = snappy.replace("relaxed { maxNonOrtho 75; }", "relaxed { maxNonOrtho 75; minTetQuality -1e30; }")
    open(os.path.join(d, "system", "snappyHexMeshDict"), "w").write(snappy)
    dec = open(os.path.join(d, "system", "decomposeParDict")).read()
    dec = re.sub(r"numberOfSubdomains\s+\d+;", f"numberOfSubdomains {a.np};", dec)
    open(os.path.join(d, "system", "decomposeParDict"), "w").write(dec)
    for f in ["MRFProperties"]:
        shutil.copy(os.path.join(CASE, "constant", f), os.path.join(d, "constant", f))
    put(d, "constant/transportProperties", "dictionary", "transportProperties",
        f"transportModel Newtonian;\nnu {a.nu:g};\n")
    rasb = """simulationType RAS;
RAS { RASModel kOmegaSST; turbulence on; printCoeffs on; }
"""
    put(d, "constant/turbulenceProperties", "dictionary", "turbulenceProperties", rasb)
    put(d, "constant/momentumTransport", "dictionary", "momentumTransport", rasb)
    k_inf = 1.5 * (0.01 * 0.63) ** 2
    om_inf = k_inf / a.nu  # freestream eddy-viscosity ratio 1
    E = "    #includeEtc \"caseDicts/setConstraintTypes\"\n"
    put(d, "0/U", "volVectorField", "U", """dimensions [0 1 -1 0 0 0 0];
internalField uniform (0.63 0 0);
boundaryField
{
""" + E + """    inlet { type fixedValue; value uniform (0.63 0 0); }
    outlet { type inletOutlet; inletValue uniform (0 0 0); value uniform (0.63 0 0); }
    sides { type slip; }
    propeller { type fixedValue; value uniform (0 0 0); }
}
""")
    put(d, "0/p", "volScalarField", "p", """dimensions [0 2 -2 0 0 0 0];
internalField uniform 0;
boundaryField
{
""" + E + """    inlet { type zeroGradient; }
    outlet { type fixedValue; value uniform 0; }
    sides { type slip; }
    propeller { type zeroGradient; }
}
""")
    put(d, "0/k", "volScalarField", "k", f"""dimensions [0 2 -2 0 0 0 0];
internalField uniform {k_inf:.6g};
boundaryField
{{
{E}    inlet {{ type fixedValue; value uniform {k_inf:.6g}; }}
    outlet {{ type inletOutlet; inletValue uniform {k_inf:.6g}; value uniform {k_inf:.6g}; }}
    sides {{ type slip; }}
    propeller {{ type kqRWallFunction; value uniform {k_inf:.6g}; }}
}}
""")
    put(d, "0/omega", "volScalarField", "omega", f"""dimensions [0 0 -1 0 0 0 0];
internalField uniform {om_inf:.6g};
boundaryField
{{
{E}    inlet {{ type fixedValue; value uniform {om_inf:.6g}; }}
    outlet {{ type inletOutlet; inletValue uniform {om_inf:.6g}; value uniform {om_inf:.6g}; }}
    sides {{ type slip; }}
    propeller {{ type omegaWallFunction; value uniform {om_inf:.6g}; }}
}}
""")
    wf = "nutUSpaldingWallFunction" if a.wall == "spalding" else "nutLowReWallFunction"
    put(d, "0/nut", "volScalarField", "nut", f"""dimensions [0 2 -1 0 0 0 0];
internalField uniform 0;
boundaryField
{{
{E}    inlet {{ type calculated; value uniform 0; }}
    outlet {{ type calculated; value uniform 0; }}
    sides {{ type calculated; value uniform 0; }}
    propeller {{ type {wf}; value uniform 0; }}
}}
""")
    put(d, "system/fvSchemes", "dictionary", "fvSchemes", """ddtSchemes { default steadyState; }
gradSchemes { default Gauss linear; grad(U) cellLimited Gauss linear 1; }
divSchemes
{
    default none;
    div(phi,U) """ + a.divU + """;
    div(phi,k) bounded Gauss upwind;
    div(phi,omega) bounded Gauss upwind;
    div((nuEff*dev2(T(grad(U))))) Gauss linear;
}
laplacianSchemes { default Gauss linear limited corrected 0.33; }
interpolationSchemes { default linear; }
snGradSchemes { default limited corrected 0.33; }
wallDist { method meshWave; }
""")
    put(d, "system/fvSolution", "dictionary", "fvSolution", """solvers
{
    p { solver GAMG; smoother GaussSeidel; tolerance 1e-10; relTol 0.01; }
    U { solver smoothSolver; smoother symGaussSeidel; tolerance 1e-12; relTol 0.01; }
    "(k|omega)" { solver smoothSolver; smoother symGaussSeidel; tolerance 1e-12; relTol 0.01; }
}
SIMPLE { nNonOrthogonalCorrectors 1; consistent yes; }
relaxationFactors { fields { p 1; } equations { U 0.7; k 0.7; omega 0.7; } }
cache { grad(U); }
""")
    put(d, "system/controlDict", "dictionary", "controlDict", f"""application simpleFoam;
startFrom startTime;
startTime 0;
stopAt endTime;
endTime {a.iters};
deltaT 1;
writeControl timeStep;
writeInterval {a.iters};
purgeWrite 0;
writeFormat ascii;
writePrecision 12;
timeFormat general;
timePrecision 10;
runTimeModifiable false;
functions
{{
    forces
    {{
        type forces; libs (forces); writeControl timeStep; writeInterval 1;
        patches (propeller); rho rhoInf; rhoInf 1; CofR (0 0 0); log false;
    }}
}}
""")
    mpirun = f"/usr/bin/mpirun -np {a.np} --bind-to core --map-by core"
    with open(os.path.join(d, "Allmesh"), "w") as f:
        f.write(f"""#!/usr/bin/env bash
# rotor2 L1 + prism layers (make_rotor_bl.py). OpenFOAM v2606 sourced.
set -euo pipefail
cd "$(dirname "$0")"
rm -rf constant/polyMesh constant/extendedFeatureEdgeMesh processor*
blockMesh > log.blockMesh 2>&1
surfaceFeatureExtract > log.surfaceFeatureExtract 2>&1
decomposePar -force -no-fields > log.decomposePar0 2>&1
{mpirun} snappyHexMesh -parallel -overwrite > log.snappyHexMesh 2>&1
reconstructParMesh -constant -mergeTol 1e-6 > log.reconstructParMesh 2>&1
rm -rf processor* constant/polyMesh/sets
checkMesh > log.checkMesh 2>&1 || true
echo meshed
""")
    with open(os.path.join(d, "Allrun"), "w") as f:
        f.write(f"""#!/usr/bin/env bash
# simpleFoam kOmegaSST + MRF on the layered mesh, np {a.np}.
set -euo pipefail
cd "$(dirname "$0")"
rm -rf processor* postProcessing [1-9]*
decomposePar -force > log.decomposePar 2>&1
{mpirun} simpleFoam -parallel > log.simpleFoam 2>&1
reconstructPar -latestTime > log.reconstructPar 2>&1
simpleFoam -postProcess -func yPlus -latestTime > log.yPlus 2>&1 || true
echo solved
""")
    for s in ["Allmesh", "Allrun"]:
        os.chmod(os.path.join(d, s), 0o755)
    print("wrote", d, f"(nu {a.nu:g}, {a.layers} layers, first {a.first} expansion {a.expansion}, "
                       f"k_inf {k_inf:.4g}, omega_inf {om_inf:.4g})")


if __name__ == "__main__":
    main()
