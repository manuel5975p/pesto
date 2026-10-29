#!/usr/bin/env python3
"""Writes the OpenFOAM v2606 parity cases of the incompressible FV path
(docs/fv.md, gates IF1/IF2) into DEST/<name>/: system/, constant/, 0/.

  cavity       lid-driven cavity Re 100, 24x24 (orthogonal), SIMPLE
  cavity_skew  the same in a 26.6 deg parallelogram, SIMPLEC + 2 non-orth
               correctors (non-orthogonal correction, p relax 1)
  channel      graded 2D channel Re_H 100: inlet fixedValue, outlet
               inletOutlet + fixedValue p, noSlip floor, slip lid,
               linearUpwind grad(U) + cellLimited grad(U)
  duct         3D non-orthogonal z-periodic duct: freestreamVelocity /
               freestreamPressure inlet+outlet, noSlip floor, symmetryPlane
               top, cyclic front/back (SIMPLEC, 1 non-orth corrector)
  sphere       snappy sphere D 1 Re 100 with prism layers (IF2): inlet,
               inletOutlet outlet, slip sides, noSlip sphere; the B_G1s
               schemes (bounded Gauss linear, corrected, consistent)
  cavity_lim   cavity_skew with `limited corrected 0.33` laplacians + snGrad
  mrf2         two adjacent MRF zones + an inactive one, nonRotatingPatches
               (literal + regex), rotating wall, MRF-included open/slip/
               symmetryPlane faces (phase MRF)

Linear solvers of the IF1 cases are converged TIGHT (PCG/smoothSolver to
1e-12, relTol 0): every SIMPLE iteration is then the algorithm's own, so an
identical implementation reproduces the history iteration by iteration.

usage: make_cases.py DEST [name ...]
"""
import os
import sys

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


def common(root, nu, end, write, schemes, solution):
    put(root, "constant/transportProperties", "dictionary", "transportProperties",
        f"transportModel Newtonian;\nnu {nu};\n")
    put(root, "constant/momentumTransport", "dictionary", "momentumTransport",
        "simulationType laminar;\n")
    put(root, "constant/turbulenceProperties", "dictionary", "turbulenceProperties",
        "simulationType laminar;\n")
    put(root, "system/fvSchemes", "dictionary", "fvSchemes", schemes)
    put(root, "system/fvSolution", "dictionary", "fvSolution", solution)
    put(root, "system/controlDict", "dictionary", "controlDict", f"""application simpleFoam;
startFrom startTime;
startTime 0;
stopAt endTime;
endTime {end};
deltaT 1;
writeControl timeStep;
writeInterval {write};
purgeWrite 0;
writeFormat ascii;
writePrecision 17;
timeFormat general;
timePrecision 10;
runTimeModifiable false;
""")
    put(root, "system/decomposeParDict", "dictionary", "decomposeParDict",
        "numberOfSubdomains 12;\nmethod scotch;\n")


SCHEMES_LINEAR = """ddtSchemes      { default steadyState; }
gradSchemes     { default Gauss linear; }
divSchemes
{
    default         none;
    div(phi,U)      bounded Gauss linear;
    div((nuEff*dev2(T(grad(U))))) Gauss linear;
}
laplacianSchemes { default Gauss linear corrected; }
interpolationSchemes { default linear; }
snGradSchemes   { default corrected; }
"""

SCHEMES_LUD = """ddtSchemes      { default steadyState; }
gradSchemes     { default Gauss linear; grad(U) cellLimited Gauss linear 1; }
divSchemes
{
    default         none;
    div(phi,U)      bounded Gauss linearUpwind grad(U);
    div((nuEff*dev2(T(grad(U))))) Gauss linear;
}
laplacianSchemes { default Gauss linear corrected; }
interpolationSchemes { default linear; }
snGradSchemes   { default corrected; }
"""


def tight_solution(simple, relax):
    return f"""solvers
{{
    p {{ solver PCG; preconditioner DIC; tolerance 1e-13; relTol 0; maxIter 5000; }}
    U {{ solver smoothSolver; smoother symGaussSeidel; tolerance 1e-14; relTol 0; maxIter 5000; }}
}}
SIMPLE
{{
{simple}
}}
relaxationFactors {{ {relax} }}
"""


def cavity(root, skew):
    x2, x3 = ("1.5", "0.5") if skew else ("1", "0")
    put(root, "system/blockMeshDict", "dictionary", "blockMeshDict", f"""scale 0.1;
vertices
(
    (0 0 0) (1 0 0) ({x2} 1 0) ({x3} 1 0)
    (0 0 0.1) (1 0 0.1) ({x2} 1 0.1) ({x3} 1 0.1)
);
blocks ( hex (0 1 2 3 4 5 6 7) (24 24 1) simpleGrading (1 1 1) );
edges ();
boundary
(
    movingWall {{ type wall; faces ((3 7 6 2)); }}
    fixedWalls {{ type wall; faces ((0 4 7 3) (2 6 5 1) (1 5 4 0)); }}
    frontAndBack {{ type empty; faces ((0 3 2 1) (4 5 6 7)); }}
);
""")
    put(root, "0/U", "volVectorField", "U", """dimensions [0 1 -1 0 0 0 0];
internalField uniform (0 0 0);
boundaryField
{
    movingWall { type fixedValue; value uniform (1 0 0); }
    fixedWalls { type noSlip; }
    frontAndBack { type empty; }
}
""")
    put(root, "0/p", "volScalarField", "p", """dimensions [0 2 -2 0 0 0 0];
internalField uniform 0;
boundaryField
{
    movingWall { type zeroGradient; }
    fixedWalls { type zeroGradient; }
    frontAndBack { type empty; }
}
""")
    if skew:
        simple = "    nNonOrthogonalCorrectors 2;\n    consistent yes;\n    pRefCell 0;\n    pRefValue 0;"
        relax = "fields { p 1; } equations { U 0.9; }"
    else:
        simple = "    nNonOrthogonalCorrectors 0;\n    consistent no;\n    pRefCell 0;\n    pRefValue 0;"
        relax = "fields { p 0.3; } equations { U 0.7; }"
    common(root, 0.001, 400, 100, SCHEMES_LINEAR, tight_solution(simple, relax))


def channel(root):
    put(root, "system/blockMeshDict", "dictionary", "blockMeshDict", """vertices
(
    (0 0 0) (1 0 0) (1 0.2 0) (0 0.2 0)
    (0 0 0.02) (1 0 0.02) (1 0.2 0.02) (0 0.2 0.02)
);
blocks ( hex (0 1 2 3 4 5 6 7) (50 16 1) simpleGrading (1 4 1) );
edges ();
boundary
(
    inlet { type patch; faces ((0 4 7 3)); }
    outlet { type patch; faces ((2 6 5 1)); }
    floor { type wall; faces ((1 5 4 0)); }
    lid { type patch; faces ((3 7 6 2)); }
    frontAndBack { type empty; faces ((0 3 2 1) (4 5 6 7)); }
);
""")
    put(root, "0/U", "volVectorField", "U", """dimensions [0 1 -1 0 0 0 0];
internalField uniform (1 0 0);
boundaryField
{
    inlet { type fixedValue; value uniform (1 0 0); }
    outlet { type inletOutlet; inletValue uniform (0 0 0); value uniform (1 0 0); }
    floor { type noSlip; }
    lid { type slip; }
    #includeEtc "caseDicts/setConstraintTypes"
}
""")
    put(root, "0/p", "volScalarField", "p", """dimensions [0 2 -2 0 0 0 0];
internalField uniform 0;
boundaryField
{
    inlet { type zeroGradient; }
    outlet { type fixedValue; value uniform 0; }
    floor { type zeroGradient; }
    lid { type slip; }
    #includeEtc "caseDicts/setConstraintTypes"
}
""")
    simple = "    nNonOrthogonalCorrectors 0;\n    consistent no;"
    relax = "fields { p 0.3; } equations { U 0.7; }"
    common(root, 0.002, 400, 100, SCHEMES_LUD, tight_solution(simple, relax))


def sphere(root):
    # Box [-4, 12] x [-4, 4]^2, base h 0.5; sphere level 4 (h 1/32, D/h 32),
    # wake box level 2; 5 layers (expansion 1.2, final layer 0.5 of the cell).
    put(root, "system/blockMeshDict", "dictionary", "blockMeshDict", """vertices
(
    (-4 -4 -4) (12 -4 -4) (12 4 -4) (-4 4 -4)
    (-4 -4 4) (12 -4 4) (12 4 4) (-4 4 4)
);
blocks ( hex (0 1 2 3 4 5 6 7) (32 16 16) simpleGrading (1 1 1) );
edges ();
boundary
(
    inlet { type patch; faces ((0 4 7 3)); }
    outlet { type patch; faces ((1 2 6 5)); }
    sides { type patch; faces ((0 1 5 4) (3 7 6 2) (0 3 2 1) (4 5 6 7)); }
);
""")
    put(root, "system/snappyHexMeshDict", "dictionary", "snappyHexMeshDict", """castellatedMesh true;
snap true;
addLayers true;
geometry
{
    sphere { type searchableSphere; centre (0 0 0); radius 0.5; }
    wake { type searchableBox; min (-1 -1.25 -1.25); max (6 1.25 1.25); }
    near { type searchableSphere; centre (0 0 0); radius 0.8; }
}
castellatedMeshControls
{
    maxLocalCells 2000000;
    maxGlobalCells 4000000;
    minRefinementCells 0;
    nCellsBetweenLevels 3;
    features ();
    refinementSurfaces { sphere { level (4 4); patchInfo { type wall; } } }
    resolveFeatureAngle 30;
    refinementRegions
    {
        wake { mode inside; levels ((1e15 2)); }
        near { mode inside; levels ((1e15 3)); }
    }
    locationInMesh (3.13 2.17 1.11);
    allowFreeStandingZoneFaces true;
}
snapControls
{
    nSmoothPatch 3;
    tolerance 2.0;
    nSolveIter 50;
    nRelaxIter 5;
}
addLayersControls
{
    relativeSizes true;
    layers { sphere { nSurfaceLayers 5; } }
    expansionRatio 1.2;
    finalLayerThickness 0.5;
    minThickness 0.1;
    nGrow 0;
    featureAngle 60;
    nRelaxIter 5;
    nSmoothSurfaceNormals 1;
    nSmoothNormals 3;
    nSmoothThickness 10;
    maxFaceThicknessRatio 0.5;
    maxThicknessToMedialRatio 0.3;
    minMedialAxisAngle 90;
    nBufferCellsNoExtrude 0;
    nLayerIter 50;
}
meshQualityControls
{
    maxNonOrtho 65;
    maxBoundarySkewness 20;
    maxInternalSkewness 4;
    maxConcave 80;
    minVol 1e-13;
    minTetQuality 1e-15;
    minArea -1;
    minTwist 0.02;
    minDeterminant 0.001;
    minFaceWeight 0.05;
    minVolRatio 0.01;
    minTriangleTwist -1;
    nSmoothScale 4;
    errorReduction 0.75;
}
mergeTolerance 1e-6;
""")
    put(root, "0/U", "volVectorField", "U", """dimensions [0 1 -1 0 0 0 0];
internalField uniform (1 0 0);
boundaryField
{
    inlet { type fixedValue; value uniform (1 0 0); }
    outlet { type inletOutlet; inletValue uniform (0 0 0); value uniform (1 0 0); }
    sides { type slip; }
    sphere { type noSlip; }
}
""")
    put(root, "0/p", "volScalarField", "p", """dimensions [0 2 -2 0 0 0 0];
internalField uniform 0;
boundaryField
{
    inlet { type zeroGradient; }
    outlet { type fixedValue; value uniform 0; }
    sides { type slip; }
    sphere { type zeroGradient; }
}
""")
    schemes = """ddtSchemes      { default steadyState; }
gradSchemes     { default Gauss linear; grad(U) Gauss linear; }
divSchemes
{
    default         none;
    div(phi,U)      bounded Gauss linear;
    div((nuEff*dev2(T(grad(U))))) Gauss linear;
}
laplacianSchemes { default Gauss linear corrected; }
interpolationSchemes { default linear; }
snGradSchemes   { default corrected; }
"""
    solution = """solvers
{
    p { solver GAMG; smoother GaussSeidel; tolerance 1e-10; relTol 0.01; }
    U { solver smoothSolver; smoother symGaussSeidel; tolerance 1e-12; relTol 0.01; }
}
SIMPLE
{
    nNonOrthogonalCorrectors 1;
    consistent      yes;
}
relaxationFactors { fields { p 1.0; } equations { U 0.7; } }
cache { grad(U); }
"""
    common(root, 0.01, 1500, 1500, schemes, solution)
    # forces on the sphere: rhoInf 1, CofR origin (the B_G1s object)
    path = os.path.join(root, "system/controlDict")
    with open(path, "a") as f:
        f.write("""functions
{
    forces
    {
        type forces; libs (forces); writeControl timeStep; writeInterval 1;
        patches (sphere); rho rhoInf; rhoInf 1; CofR (0 0 0); log false;
    }
}
""")


def duct(root):
    # 3D, non-orthogonal (top shifted 0.1 in x), z periodic: freestream
    # inlet/outlet (both mixed BCs), noSlip floor, symmetryPlane top.
    put(root, "system/blockMeshDict", "dictionary", "blockMeshDict", """vertices
(
    (0 0 0) (1 0 0) (1.1 0.5 0) (0.1 0.5 0)
    (0 0 0.25) (1 0 0.25) (1.1 0.5 0.25) (0.1 0.5 0.25)
);
blocks ( hex (0 1 2 3 4 5 6 7) (16 8 4) simpleGrading (1 2 1) );
edges ();
boundary
(
    inlet { type patch; faces ((0 4 7 3)); }
    outlet { type patch; faces ((2 6 5 1)); }
    floor { type wall; faces ((1 5 4 0)); }
    top { type symmetryPlane; faces ((3 7 6 2)); }
    front { type cyclic; neighbourPatch back; transform translational; separationVector (0 0 0.25); faces ((0 3 2 1)); }
    back { type cyclic; neighbourPatch front; transform translational; separationVector (0 0 -0.25); faces ((4 5 6 7)); }
);
""")
    put(root, "0/U", "volVectorField", "U", """dimensions [0 1 -1 0 0 0 0];
internalField uniform (1 0 0);
boundaryField
{
    inlet { type freestreamVelocity; freestreamValue uniform (1 0 0); value uniform (1 0 0); }
    outlet { type freestreamVelocity; freestreamValue uniform (1 0 0); value uniform (1 0 0); }
    floor { type noSlip; }
    #includeEtc "caseDicts/setConstraintTypes"
}
""")
    put(root, "0/p", "volScalarField", "p", """dimensions [0 2 -2 0 0 0 0];
internalField uniform 0;
boundaryField
{
    inlet { type freestreamPressure; freestreamValue uniform 0; }
    outlet { type freestreamPressure; freestreamValue uniform 0; }
    floor { type zeroGradient; }
    #includeEtc "caseDicts/setConstraintTypes"
}
""")
    simple = "    nNonOrthogonalCorrectors 1;\n    consistent yes;"
    relax = "fields { p 1; } equations { U 0.9; }"
    common(root, 0.01, 300, 100, SCHEMES_LINEAR, tight_solution(simple, relax))


def cavity_lim(root):
    # cavity_skew with OpenFOAM's snappy-mesh default `limited corrected 0.33`
    # on both laplacians (U, p) and on fvc::snGrad (SIMPLEC's phiHbyA term).
    cavity(root, True)
    put(root, "system/fvSchemes", "dictionary", "fvSchemes",
        SCHEMES_LINEAR.replace("Gauss linear corrected;", "Gauss linear limited corrected 0.33;")
        .replace("snGradSchemes   { default corrected; }", "snGradSchemes   { default limited corrected 0.33; }"))


def mrf2(root):
    # Two MRF zones (MRFZoneList) on a 3-block non-orthogonal 3D box: zone
    # rotA (x in [0,1]) spins about x, zone rotB (x in [1,2], ADJACENT: the
    # zone/zone faces take both zones' frame fluxes) about a non-normalised
    # oblique axis with `omega constant`; block 3 is static. nonRotatingPatches:
    # rotA's inlet (literal), rotB's floor (regex "fl.*"); the floor under rotA
    # is a rotating wall (U_b = Omega x r), the open top / slip / symmetryPlane
    # faces in the zones are MRF-included (phi = 0). An inactive third zone.
    put(root, "system/blockMeshDict", "dictionary", "blockMeshDict", """vertices
(
    (0 0 0) (1 0 0) (2 0 0) (3 0 0)
    (0.1 1 0) (1.1 1 0) (2.1 1 0) (3.1 1 0)
    (0 0 1) (1 0 1) (2 0 1) (3 0 1)
    (0.1 1 1) (1.1 1 1) (2.1 1 1) (3.1 1 1)
);
blocks
(
    hex (0 1 5 4 8 9 13 12) rotA (8 8 6) simpleGrading (1 1 1)
    hex (1 2 6 5 9 10 14 13) rotB (8 8 6) simpleGrading (1 1 1)
    hex (2 3 7 6 10 11 15 14) (8 8 6) simpleGrading (1 1 1)
);
edges ();
boundary
(
    inlet { type patch; faces ((0 8 12 4)); }
    outlet { type patch; faces ((7 15 11 3)); }
    floor { type wall; faces ((1 9 8 0) (2 10 9 1) (3 11 10 2)); }
    top { type patch; faces ((4 12 13 5) (5 13 14 6) (6 14 15 7)); }
    zlo { type symmetryPlane; faces ((0 4 5 1) (1 5 6 2) (2 6 7 3)); }
    zhi { type patch; faces ((8 9 13 12) (9 10 14 13) (10 11 15 14)); }
);
""")
    put(root, "0/U", "volVectorField", "U", """dimensions [0 1 -1 0 0 0 0];
internalField uniform (1 0 0);
boundaryField
{
    inlet { type fixedValue; value uniform (1 0 0); }
    outlet { type inletOutlet; inletValue uniform (0 0 0); value uniform (1 0 0); }
    floor { type noSlip; }
    top { type zeroGradient; }
    zhi { type slip; }
    #includeEtc "caseDicts/setConstraintTypes"
}
""")
    put(root, "0/p", "volScalarField", "p", """dimensions [0 2 -2 0 0 0 0];
internalField uniform 0;
boundaryField
{
    inlet { type zeroGradient; }
    outlet { type fixedValue; value uniform 0; }
    floor { type zeroGradient; }
    top { type fixedValue; value uniform 0; }
    zhi { type slip; }
    #includeEtc "caseDicts/setConstraintTypes"
}
""")
    put(root, "constant/MRFProperties", "dictionary", "MRFProperties", """rotA
{
    cellZone    rotA;
    active      yes;
    nonRotatingPatches (inlet);
    origin      (0.5 0.5 0.5);
    axis        (1 0 0);
    omega       2;
}
rotB
{
    cellZone    rotB;
    nonRotatingPatches ("fl.*");
    origin      (1.5 0.3 0.6);
    axis        (0 1 1);
    omega       constant -1.5;
}
ghost
{
    active      no;
}
""")
    simple = "    nNonOrthogonalCorrectors 1;\n    consistent yes;"
    relax = "fields { p 1; } equations { U 0.7; }"
    common(root, 0.05, 300, 100, SCHEMES_LINEAR, tight_solution(simple, relax))


CASES = {"cavity": lambda r: cavity(r, False), "cavity_skew": lambda r: cavity(r, True),
         "channel": channel, "duct": duct, "sphere": sphere, "cavity_lim": cavity_lim, "mrf2": mrf2}

if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    dest = sys.argv[1]
    names = sys.argv[2:] or list(CASES)
    for n in names:
        if n not in CASES:
            sys.exit(f"unknown case {n} (one of {', '.join(CASES)})")
        CASES[n](os.path.join(dest, n))
        print("wrote", os.path.join(dest, n))
