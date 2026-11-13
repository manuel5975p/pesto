#!/usr/bin/env python3
"""M17 isolated-propeller mesh case (blade STL -> full 360 deg rotor -> snappyHexMesh).

Source: the M17 CAD export, ONE blade per prop (mm, axis x through the origin,
rotor plane x ~ 1.0-1.05 m). Writes DEST/ with constant/triSurface/{prop,hub}.stl,
system/*, and DEST/Allmesh (blockMesh, surfaceFeatureExtract, decomposePar,
mpirun snappyHexMesh -parallel, reconstructParMesh, checkMesh). The reader of
this repo rejects rotational cyclics, so the rotor is the whole disc: N copies
of the blade, 2 pi/N apart about +x. The hub is a stand-in capsule (radius 40 mm,
the mesh's wall-patch root radius), not the airframe's spinner.

Solve: simpleFoam kOmegaSST + MRF (zone `rotor`, axis +x, omega of the M17
dynamicMeshDict: fwd +837.758, aft -837.758 rad/s), the M17 case's BCs (freestream
U/p on inlet|outlet|sides, k/omega inletOutlet at M17's 0.184 / 15.7, nutk/kqR/omega
wall functions) and the steady form of its schemes; DEST/Allrun (np P, forces on
the prop + hub in postProcessing/, rho 1.225, CofR origin).

usage: make_m17_prop.py DEST --prop fwd|aft [--np P] [--base H] [--layers N] [--first F] [--iters N]
Run:   source /opt/OpenFOAM/OpenFOAM-v2606/etc/bashrc
       build/simjob submit --cpu -t 14400 -m 16000 --inherit-env --wait -- /usr/bin/bash DEST/Allmesh
       build/simjob submit --cpu -t 14400 -m 16000 --inherit-env --wait -- /usr/bin/bash DEST/Allrun
Freestream is -x (the high-x end is upstream); inlet = x max.
"""
import argparse
import math
import os
import shutil
import struct

CAD = os.path.expanduser("~/Documents/M17-SntlMainOml/M17-SntlMainOml-CFDOutput-Prop{}.stl")
PROPS = {"fwd": ("Fwd", 5, 837.758), "aft": ("Aft", 6, -837.758)}   # tag, blades, omega [rad/s]
U_INF = 35.0           # freestream speed, along -x
NU = 1.5e-5
K_INF, OMEGA_INF = 0.184, 15.7   # the M17 case's 0/k, 0/omega
D = 0.3
X_ROT = 1.043          # rotor plane (blade x centre), fwd
HUB_R = 0.040


def read_stl(path):
    b = open(path, "rb").read()
    n = struct.unpack("<I", b[80:84])[0]
    assert len(b) == 84 + 50 * n, "binary STL expected"
    return [struct.unpack("<9f", b[84 + 50 * i + 12:84 + 50 * i + 48]) for i in range(n)]


def write_stl(path, name, tris):
    out = bytearray(name.encode().ljust(80, b"\0"))
    out += struct.pack("<I", len(tris))
    for t in tris:
        a, b, c = t[0:3], t[3:6], t[6:9]
        u = [b[i] - a[i] for i in range(3)]
        v = [c[i] - a[i] for i in range(3)]
        n = [u[1] * v[2] - u[2] * v[1], u[2] * v[0] - u[0] * v[2], u[0] * v[1] - u[1] * v[0]]
        m = math.sqrt(sum(x * x for x in n)) or 1.0
        out += struct.pack("<12fH", n[0] / m, n[1] / m, n[2] / m, *t, 0)
    open(path, "wb").write(bytes(out))


def rot_x(t, ang):
    c, s = math.cos(ang), math.sin(ang)
    r = []
    for k in range(3):
        x, y, z = t[3 * k:3 * k + 3]
        r += [x, c * y - s * z, s * y + c * z]
    return tuple(r)


def capsule(x0, x1, r, nphi=96, ncap=16):
    """Closed capsule about +x: cylinder x0..x1 with hemispherical caps."""
    rings = []  # (x, rho)
    for i in range(ncap + 1):  # pole at x0 - r .. equator at x0
        a = 0.5 * math.pi * i / ncap
        rings.append((x0 - r * math.cos(a), r * math.sin(a)))
    rings.append((x1, r))
    for i in range(1, ncap + 1):  # equator at x1 .. pole at x1 + r
        a = 0.5 * math.pi * i / ncap
        rings.append((x1 + r * math.sin(a), r * math.cos(a)))
    tris = []
    pt = lambda x, rho, j: (x, rho * math.cos(2 * math.pi * j / nphi), rho * math.sin(2 * math.pi * j / nphi))
    for (xa, ra), (xb, rb) in zip(rings[:-1], rings[1:]):
        for j in range(nphi):
            p00, p01 = pt(xa, ra, j), pt(xa, ra, j + 1)
            p10, p11 = pt(xb, rb, j), pt(xb, rb, j + 1)
            if ra > 1e-12:
                tris.append(p00 + p01 + p11)
            if rb > 1e-12:
                tris.append(p00 + p11 + p10)
    return tris


HDR = "FoamFile\n{{\n    version     2.0;\n    format      ascii;\n    class       {cls};\n    object      {obj};\n}}\n"


def put(root, rel, cls, obj, body):
    path = os.path.join(root, rel)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    open(path, "w").write(HDR.format(cls=cls, obj=obj) + body)


def write_physics(d, name, omega):
    """constant/ (nu, kOmegaSST, MRF zone `rotor`) and 0/ with the M17 case's BCs."""
    put(d, "constant/transportProperties", "dictionary", "transportProperties", f"transportModel Newtonian;\nnu {NU:g};\n")
    ras = "simulationType RAS;\nRAS { RASModel kOmegaSST; turbulence on; printCoeffs on; }\n"
    put(d, "constant/turbulenceProperties", "dictionary", "turbulenceProperties", ras)
    put(d, "constant/MRFProperties", "dictionary", "MRFProperties", f"""rotor
{{
    cellZone    rotor;
    active      yes;
    nonRotatingPatches ();
    origin      (0 0 0);
    axis        (1 0 0);
    omega       {omega};
}}
""")
    walls = f"({name}|hub)"
    far = "(inlet|outlet|sides)"
    uinf = f"uniform ({-U_INF:g} 0 0)"
    put(d, "0/U", "volVectorField", "U", f"""dimensions [0 1 -1 0 0 0 0];
internalField {uinf};
boundaryField
{{
    "{far}" {{ type freestreamVelocity; freestreamValue {uinf}; value {uinf}; }}
    "{walls}" {{ type fixedValue; value uniform (0 0 0); }}
}}
""")
    put(d, "0/p", "volScalarField", "p", f"""dimensions [0 2 -2 0 0 0 0];
internalField uniform 0;
boundaryField
{{
    "{far}" {{ type freestreamPressure; freestreamValue uniform 0; value uniform 0; }}
    "{walls}" {{ type zeroGradient; }}
}}
""")
    for fld, val, dim, wf in (("k", K_INF, "[0 2 -2 0 0 0 0]", "kqRWallFunction"),
                              ("omega", OMEGA_INF, "[0 0 -1 0 0 0 0]", "omegaWallFunction")):
        put(d, f"0/{fld}", "volScalarField", fld, f"""dimensions {dim};
internalField uniform {val:g};
boundaryField
{{
    "{far}" {{ type inletOutlet; inletValue uniform {val:g}; value uniform {val:g}; }}
    "{walls}" {{ type {wf}; value uniform {val:g}; }}
}}
""")
    put(d, "0/nut", "volScalarField", "nut", f"""dimensions [0 2 -1 0 0 0 0];
internalField uniform 0;
boundaryField
{{
    "{far}" {{ type calculated; value uniform 0; }}
    "{walls}" {{ type nutkWallFunction; value uniform 0; }}
}}
""")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dest")
    ap.add_argument("--prop", choices=PROPS, default="fwd")
    ap.add_argument("--np", type=int, default=12)
    ap.add_argument("--base", type=float, default=0.05, help="background cell size [m]")
    ap.add_argument("--blade-level", type=int, default=6)
    ap.add_argument("--layers", type=int, default=0)
    ap.add_argument("--first", type=float, default=0.5, help="relative first layer (0.5: centroid y+ ~60 at 0.75R)")
    ap.add_argument("--iters", type=int, default=2000, help="simpleFoam iterations")
    ap.add_argument("--relax-layers", action="store_true",
                    help="layers: featureAngle 180, medial ratio 0.6, relaxed minTetQuality -1e30 (thin blades)")
    a = ap.parse_args()
    tag, nb, omega = PROPS[a.prop]
    name = "prop" + tag
    d = os.path.abspath(a.dest)
    shutil.rmtree(d, ignore_errors=True)
    os.makedirs(os.path.join(d, "constant", "triSurface"))

    blade = [tuple(1e-3 * v for v in t) for t in read_stl(CAD.format(tag))]
    xs = [t[3 * k] for t in blade for k in range(3)]
    xc = 0.5 * (min(xs) + max(xs))
    rmax = max(math.hypot(t[3 * k + 1], t[3 * k + 2]) for t in blade for k in range(3))
    tris = [rot_x(t, 2 * math.pi * k / nb) for k in range(nb) for t in blade]
    write_stl(os.path.join(d, "constant", "triSurface", name + ".stl"), name, tris)
    hx0, hx1 = xc - 0.028, xc + 0.032  # cylinder part; caps add HUB_R each side
    write_stl(os.path.join(d, "constant", "triSurface", "hub.stl"), "hub", capsule(hx0, hx1, HUB_R))

    zx0, zx1, zr = hx0 - HUB_R - 0.015, hx1 + HUB_R + 0.015, rmax + 0.02   # MRF zone
    xin, xout, half = xc + 4 * D, xc - 8 * D, 3 * D
    h = a.base
    n = lambda L: max(1, round(L / h))
    nx, ny = n(xin - xout), n(2 * half)
    L = a.blade_level
    put(d, "system/blockMeshDict", "dictionary", "blockMeshDict", f"""scale 1;
vertices
(
    ({xout:.6f} {-half} {-half}) ({xin:.6f} {-half} {-half}) ({xin:.6f} {half} {-half}) ({xout:.6f} {half} {-half})
    ({xout:.6f} {-half} {half}) ({xin:.6f} {-half} {half}) ({xin:.6f} {half} {half}) ({xout:.6f} {half} {half})
);
blocks ( hex (0 1 2 3 4 5 6 7) ({nx} {ny} {ny}) simpleGrading (1 1 1) );
edges ();
boundary
(
    inlet  {{ type patch; faces ((1 2 6 5)); }}
    outlet {{ type patch; faces ((0 4 7 3)); }}
    sides  {{ type patch; faces ((0 1 5 4) (3 7 6 2) (0 3 2 1) (4 5 6 7)); }}
);
""")
    put(d, "system/surfaceFeatureExtractDict", "dictionary", "surfaceFeatureExtractDict", f"""{name}.stl
{{
    extractionMethod    extractFromSurface;
    includedAngle       150;
    subsetFeatures {{ nonManifoldEdges no; openEdges yes; }}
    writeObj            no;
}}
""")
    put(d, "system/decomposeParDict", "dictionary", "decomposeParDict", f"numberOfSubdomains {a.np};\nmethod scotch;\n")
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
        patches ({name} hub); rho rhoInf; rhoInf 1.225; CofR (0 0 0); log false;
    }}
}}
""")
    put(d, "system/fvSchemes", "dictionary", "fvSchemes", """ddtSchemes { default steadyState; }
gradSchemes { default cellLimited Gauss linear 1; }
divSchemes
{
    default none;
    div(phi,U) bounded Gauss linearUpwind grad(U);
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
    p { solver GAMG; smoother GaussSeidel; tolerance 1e-8; relTol 0.01; }
    U { solver smoothSolver; smoother symGaussSeidel; tolerance 1e-10; relTol 0.1; }
    "(k|omega)" { solver smoothSolver; smoother symGaussSeidel; tolerance 1e-10; relTol 0.1; }
}
SIMPLE { nNonOrthogonalCorrectors 1; consistent yes; }
relaxationFactors { fields { p 1; } equations { U 0.7; k 0.7; omega 0.7; } }
""")
    write_physics(d, name, omega)
    layers = f"""addLayersControls
{{
    relativeSizes true;
    layers {{ {name} {{ nSurfaceLayers {a.layers}; }} hub {{ nSurfaceLayers {a.layers}; }} }}
    firstLayerThickness {a.first};
    expansionRatio 1.2;
    minThickness 0.005;
    nGrow 0;
    featureAngle {180 if a.relax_layers else 130};
    slipFeatureAngle 30;
    nRelaxIter 5;
    nSmoothSurfaceNormals 1;
    nSmoothNormals 3;
    nSmoothThickness 10;
    maxFaceThicknessRatio 0.5;
    maxThicknessToMedialRatio {0.6 if a.relax_layers else 0.3};
    minMedialAxisAngle 90;
    nBufferCellsNoExtrude 0;
    nLayerIter 50;
    nRelaxedIter 20;
}}
"""
    fx = lambda x: f"{x:.6f}"
    put(d, "system/snappyHexMeshDict", "dictionary", "snappyHexMeshDict", f"""castellatedMesh true;
snap            true;
addLayers       {'true' if a.layers else 'false'};
geometry
{{
    {name}.stl {{ type triSurfaceMesh; name {name}; }}
    hub.stl {{ type triSurfaceMesh; name hub; }}
    rotor
    {{
        type cylinder;
        point1 ({fx(zx0)} 0 0);
        point2 ({fx(zx1)} 0 0);
        radius {zr:.6f};
    }}
    wake
    {{
        type cylinder;
        point1 ({fx(zx1)} 0 0);
        point2 ({fx(xout)} 0 0);
        radius 0.2;
    }}
    upstream
    {{
        type cylinder;
        point1 ({fx(zx1)} 0 0);
        point2 ({fx(zx1 + 0.7)} 0 0);
        radius 0.3;
    }}
}}
castellatedMeshControls
{{
    maxLocalCells       4000000;
    maxGlobalCells      20000000;
    minRefinementCells  0;
    maxLoadUnbalance    0.10;
    nCellsBetweenLevels 3;
    features ( {{ file "{name}.eMesh"; level {L}; }} );
    refinementSurfaces
    {{
        {name} {{ level ({L} {L}); patchInfo {{ type wall; }} }}
        hub {{ level ({L - 1} {L - 1}); patchInfo {{ type wall; }} }}
        rotor
        {{
            level (4 4);
            faceType       internal;
            faceZone       rotor;
            cellZone       rotor;
            cellZoneInside inside;
        }}
    }}
    resolveFeatureAngle 20;
    refinementRegions
    {{
        rotor    {{ mode inside; levels ((1e15 4)); }}
        wake     {{ mode inside; levels ((1e15 3)); }}
        upstream {{ mode inside; levels ((1e15 2)); }}
    }}
    locationInMesh (-1.0 0.7 0.5);
    allowFreeStandingZoneFaces false;
}}
snapControls
{{
    nSmoothPatch 3;
    tolerance 2.0;
    nSolveIter 100;
    nRelaxIter 5;
    nFeatureSnapIter 10;
    implicitFeatureSnap true;
    explicitFeatureSnap false;
    multiRegionFeatureSnap false;
}}
{layers}meshQualityControls
{{
    #includeEtc "caseDicts/meshQualityDict"
    minFaceWeight 0.02;
    nSmoothScale 4;
    errorReduction 0.75;
    relaxed {{ maxNonOrtho 75;{' minTetQuality -1e30;' if a.relax_layers else ''} }}
}}
mergeTolerance 1e-6;
""")
    mpirun = f"/usr/bin/mpirun -np {a.np} --bind-to core --map-by core"
    allmesh = os.path.join(d, "Allmesh")
    open(allmesh, "w").write(f"""#!/usr/bin/env bash
# M17 {tag} rotor mesh (make_m17_prop.py). OpenFOAM v2606 sourced.
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
    os.chmod(allmesh, 0o755)
    allrun = os.path.join(d, "Allrun")
    open(allrun, "w").write(f"""#!/usr/bin/env bash
# M17 {tag} isolated rotor: simpleFoam kOmegaSST + MRF, np {a.np} (make_m17_prop.py). OpenFOAM v2606 sourced.
set -euo pipefail
cd "$(dirname "$0")"
rm -rf processor* postProcessing [1-9]*
decomposePar -force > log.decomposePar 2>&1
{mpirun} simpleFoam -parallel > log.simpleFoam 2>&1
reconstructPar -latestTime > log.reconstructPar 2>&1
rm -rf processor*
echo solved
""")
    os.chmod(allrun, 0o755)
    print(f"wrote {d}: {tag} N={nb} blades, rmax {rmax:.4f} m, blade x centre {xc:.4f}, zone x [{zx0:.3f},{zx1:.3f}] r {zr:.3f},"
          f" domain x [{xout:.3f},{xin:.3f}] +-{half}, background {nx}x{ny}x{ny}, blade level {L} ({h / 2 ** L * 1e3:.2f} mm)")


if __name__ == "__main__":
    main()
