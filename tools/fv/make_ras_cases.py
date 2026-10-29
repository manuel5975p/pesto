#!/usr/bin/env python3
"""Writes the OpenFOAM v2606 RAS parity cases of the incompressible FV path
(docs/fv.md, phase BL) into DEST/<name>/: system/, constant/, 0/.

  sst_plate   2D flat plate (symmetryPlane run-up + wall), kOmegaSST,
              kqRWallFunction / omegaWallFunction (binomial, default) /
              nutUSpaldingWallFunction, a slip `wall` lid (it counts for the
              wall distance), bounded Gauss linearUpwind grad(U|k|omega),
              SIMPLEC, wallDist meshWave; omega is bounded every iteration
  sst_skew    2D non-orthogonal channel (26.6 deg), kOmegaSST: floor
              nutkWallFunction + omegaWallFunction stepwise + kqR, ceiling
              nutLowReWallFunction + omegaWallFunction tanh + k fixedValue,
              inletOutlet k / omega / U outlet, linearUpwind grad(k|omega) with
              cellLimited grad(k), Gauss linear corrected + 1 non-orth
              corrector, wallDist exactDistance
  sa_skew     the same mesh, SpalartAllmaras ft2 on (Prt irrelevant:
              incompressible, no T): nuTilda 0 on both walls, floor
              nutUSpaldingWallFunction, ceiling nutLowReWallFunction,
              linearUpwind grad(nuTilda), corrected, wallDist meshWave
  plate       the TMR-like zero-pressure-gradient flat plate (gate BL-a):
              OpenFOAM's tutorial turbulentFlatPlate at a quarter of its
              resolution (24 + 112 x 96, the tutorial's level-3 y+ 1 grading
              9300), Re_L 5e6, kOmegaSST, its schemes and solvers, run to its
              residualControl (1e-6)

The parity cases (sst_*, sa_*) converge every linear solve TIGHT (relTol 0),
so each SIMPLE iteration is the algorithm's own.

usage: make_ras_cases.py DEST [name ...]
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


def control(end, write, prec=17, extra=""):
    return f"""application simpleFoam;
startFrom startTime;
startTime 0;
stopAt endTime;
endTime {end};
deltaT 1;
writeControl timeStep;
writeInterval {write};
purgeWrite 0;
writeFormat ascii;
writePrecision {prec};
timeFormat general;
timePrecision 10;
runTimeModifiable false;
{extra}"""


def turb(root, model, coeffs=""):
    body = f"""simulationType RAS;
RAS
{{
    RASModel {model};
    turbulence on;
    printCoeffs on;
{coeffs}}}
"""
    put(root, "constant/turbulenceProperties", "dictionary", "turbulenceProperties", body)
    put(root, "constant/momentumTransport", "dictionary", "momentumTransport", body)


def transport(root, nu):
    put(root, "constant/transportProperties", "dictionary", "transportProperties",
        f"transportModel Newtonian;\nnu {nu};\n")
    put(root, "system/decomposeParDict", "dictionary", "decomposeParDict",
        "numberOfSubdomains 12;\nmethod scotch;\n")


def field(root, name, cls, obj, dims, internal, bcs):
    lines = "\n".join(f"    {p} {{ {b} }}" for p, b in bcs)
    put(root, f"0/{name}", cls, obj, f"""dimensions {dims};
internalField {internal};
boundaryField
{{
{lines}
}}
""")


TIGHT_SOLVERS = """    p { solver PCG; preconditioner DIC; tolerance 1e-13; relTol 0; maxIter 5000; }
    U { solver smoothSolver; smoother symGaussSeidel; tolerance 1e-14; relTol 0; maxIter 5000; }
    "(k|omega|nuTilda)" { solver PBiCGStab; preconditioner DILU; tolerance 1e-14; relTol 0; maxIter 5000; }
"""


def sst_plate(root):
    put(root, "system/blockMeshDict", "dictionary", "blockMeshDict", """vertices
(
    (-0.2 0 0) (0 0 0) (1 0 0) (1 0.3 0) (0 0.3 0) (-0.2 0.3 0)
    (-0.2 0 0.02) (0 0 0.02) (1 0 0.02) (1 0.3 0.02) (0 0.3 0.02) (-0.2 0.3 0.02)
);
blocks
(
    hex (0 1 4 5 6 7 10 11) (8 24 1) simpleGrading (0.5 30 1)
    hex (1 2 3 4 7 8 9 10) (32 24 1) simpleGrading (4 30 1)
);
edges ();
boundary
(
    inlet { type patch; faces ((0 6 11 5)); }
    outlet { type patch; faces ((2 3 9 8)); }
    top { type wall; faces ((5 11 10 4) (4 10 9 3)); }
    plate { type wall; faces ((1 2 8 7)); }
    symmetry { type symmetryPlane; faces ((0 1 7 6)); }
    frontAndBack { type empty; faces ((0 5 4 1) (1 4 3 2) (6 7 10 11) (7 8 9 10)); }
);
""")
    E = ("frontAndBack", "type empty;")
    S = ("symmetry", "type symmetryPlane;")
    field(root, "U", "volVectorField", "U", "[0 1 -1 0 0 0 0]", "uniform (1 0 0)", [
        ("inlet", "type fixedValue; value uniform (1 0 0);"), ("outlet", "type zeroGradient;"),
        ("top", "type slip;"), ("plate", "type noSlip;"), S, E])
    field(root, "p", "volScalarField", "p", "[0 2 -2 0 0 0 0]", "uniform 0", [
        ("inlet", "type zeroGradient;"), ("outlet", "type fixedValue; value uniform 0;"),
        ("top", "type zeroGradient;"), ("plate", "type zeroGradient;"), S, E])
    field(root, "k", "volScalarField", "k", "[0 2 -2 0 0 0 0]", "uniform 3.75e-05", [
        ("inlet", "type fixedValue; value uniform 3.75e-05;"),
        ("outlet", "type inletOutlet; inletValue uniform 3.75e-05; value uniform 3.75e-05;"),
        ("top", "type zeroGradient;"), ("plate", "type kqRWallFunction; value uniform 3.75e-05;"), S, E])
    field(root, "omega", "volScalarField", "omega", "[0 0 -1 0 0 0 0]", "uniform 5", [
        ("inlet", "type fixedValue; value uniform 5;"), ("outlet", "type zeroGradient;"),
        ("top", "type zeroGradient;"), ("plate", "type omegaWallFunction; value uniform 5;"), S, E])
    field(root, "nut", "volScalarField", "nut", "[0 2 -1 0 0 0 0]", "uniform 0", [
        ("inlet", "type calculated; value uniform 0;"), ("outlet", "type calculated; value uniform 0;"),
        ("top", "type calculated; value uniform 0;"),
        ("plate", "type nutUSpaldingWallFunction; value uniform 0;"), S, E])
    turb(root, "kOmegaSST")
    transport(root, "2e-05")
    put(root, "system/fvSchemes", "dictionary", "fvSchemes", """ddtSchemes { default steadyState; }
gradSchemes { default Gauss linear; }
divSchemes
{
    default none;
    div(phi,U) bounded Gauss linearUpwind grad(U);
    div(phi,k) bounded Gauss linearUpwind grad(k);
    div(phi,omega) bounded Gauss linearUpwind grad(omega);
    div((nuEff*dev2(T(grad(U))))) Gauss linear;
}
laplacianSchemes { default Gauss linear corrected; }
interpolationSchemes { default linear; }
snGradSchemes { default corrected; }
wallDist { method meshWave; }
""")
    put(root, "system/fvSolution", "dictionary", "fvSolution", "solvers\n{\n" + TIGHT_SOLVERS + """}
SIMPLE { nNonOrthogonalCorrectors 0; consistent yes; }
relaxationFactors { equations { U 0.9; k 0.8; omega 0.8; } }
""")
    # (limitedLinear is NOT a tight-parity scheme: on a field uniform to the
    # bit its r = 2000 sign(d . grad) - 1 takes the SIGN OF THE GRADIENT'S
    # ROUNDING NOISE, so the limiter is 0 or 1 by the summation order -- it
    # differs between OpenFOAM decompositions too. Its parity is gated on the
    # converged `plate` case instead.)
    put(root, "system/controlDict", "dictionary", "controlDict", control(400, 100))


def skew_mesh(root):
    # 26.6 deg parallelogram channel, wall floor + ceiling, graded to both walls
    put(root, "system/blockMeshDict", "dictionary", "blockMeshDict", """vertices
(
    (0 0 0) (1 0 0) (1.1 0.2 0) (0.1 0.2 0)
    (0 0 0.02) (1 0 0.02) (1.1 0.2 0.02) (0.1 0.2 0.02)
);
blocks ( hex (0 1 2 3 4 5 6 7) (30 20 1) simpleGrading (1 ((0.5 0.5 6) (0.5 0.5 0.1667)) 1) );
edges ();
boundary
(
    inlet { type patch; faces ((0 4 7 3)); }
    outlet { type patch; faces ((2 6 5 1)); }
    floor { type wall; faces ((1 5 4 0)); }
    ceiling { type wall; faces ((3 7 6 2)); }
    frontAndBack { type empty; faces ((0 3 2 1) (4 5 6 7)); }
);
""")
    E = ("frontAndBack", "type empty;")
    field(root, "U", "volVectorField", "U", "[0 1 -1 0 0 0 0]", "uniform (1 0 0)", [
        ("inlet", "type fixedValue; value uniform (1 0 0);"),
        ("outlet", "type inletOutlet; inletValue uniform (0 0 0); value uniform (1 0 0);"),
        ("floor", "type noSlip;"), ("ceiling", "type noSlip;"), E])
    field(root, "p", "volScalarField", "p", "[0 2 -2 0 0 0 0]", "uniform 0", [
        ("inlet", "type zeroGradient;"), ("outlet", "type fixedValue; value uniform 0;"),
        ("floor", "type zeroGradient;"), ("ceiling", "type zeroGradient;"), E])


def sst_skew(root):
    skew_mesh(root)
    E = ("frontAndBack", "type empty;")
    field(root, "k", "volScalarField", "k", "[0 2 -2 0 0 0 0]", "uniform 0.00375", [
        ("inlet", "type fixedValue; value uniform 0.00375;"),
        ("outlet", "type inletOutlet; inletValue uniform 0.00375; value uniform 0.00375;"),
        ("floor", "type kqRWallFunction; value uniform 0.00375;"),
        ("ceiling", "type fixedValue; value uniform 1e-10;"), E])
    field(root, "omega", "volScalarField", "omega", "[0 0 -1 0 0 0 0]", "uniform 20", [
        ("inlet", "type fixedValue; value uniform 20;"),
        ("outlet", "type inletOutlet; inletValue uniform 20; value uniform 20;"),
        ("floor", "type omegaWallFunction; blending stepwise; value uniform 20;"),
        ("ceiling", "type omegaWallFunction; blending tanh; value uniform 20;"), E])
    field(root, "nut", "volScalarField", "nut", "[0 2 -1 0 0 0 0]", "uniform 0", [
        ("inlet", "type calculated; value uniform 0;"), ("outlet", "type calculated; value uniform 0;"),
        ("floor", "type nutkWallFunction; value uniform 0;"),
        ("ceiling", "type nutLowReWallFunction; value uniform 0;"), E])
    turb(root, "kOmegaSST")
    transport(root, "5e-05")
    put(root, "system/fvSchemes", "dictionary", "fvSchemes", """ddtSchemes { default steadyState; }
gradSchemes { default Gauss linear; grad(k) cellLimited Gauss linear 1; }
divSchemes
{
    default none;
    div(phi,U) bounded Gauss linearUpwind grad(U);
    div(phi,k) bounded Gauss linearUpwind grad(k);
    div(phi,omega) bounded Gauss linearUpwind grad(omega);
    div((nuEff*dev2(T(grad(U))))) Gauss linear;
}
laplacianSchemes { default Gauss linear corrected; }
interpolationSchemes { default linear; }
snGradSchemes { default corrected; }
wallDist { method exactDistance; }
""")
    put(root, "system/fvSolution", "dictionary", "fvSolution", "solvers\n{\n" + TIGHT_SOLVERS + """}
SIMPLE { nNonOrthogonalCorrectors 1; consistent yes; }
relaxationFactors { equations { U 0.9; k 0.7; omega 0.7; } }
""")
    put(root, "system/controlDict", "dictionary", "controlDict", control(400, 100))


def sa_skew(root):
    skew_mesh(root)
    E = ("frontAndBack", "type empty;")
    field(root, "nuTilda", "volScalarField", "nuTilda", "[0 2 -1 0 0 0 0]", "uniform 0.00015", [
        ("inlet", "type fixedValue; value uniform 0.00015;"),
        ("outlet", "type inletOutlet; inletValue uniform 0.00015; value uniform 0.00015;"),
        ("floor", "type fixedValue; value uniform 0;"), ("ceiling", "type fixedValue; value uniform 0;"), E])
    field(root, "nut", "volScalarField", "nut", "[0 2 -1 0 0 0 0]", "uniform 0", [
        ("inlet", "type calculated; value uniform 0;"), ("outlet", "type calculated; value uniform 0;"),
        ("floor", "type nutUSpaldingWallFunction; value uniform 0;"),
        ("ceiling", "type nutLowReWallFunction; value uniform 0;"), E])
    turb(root, "SpalartAllmaras", "    SpalartAllmarasCoeffs { ft2 true; }\n")
    transport(root, "5e-05")
    put(root, "system/fvSchemes", "dictionary", "fvSchemes", """ddtSchemes { default steadyState; }
gradSchemes { default Gauss linear; }
divSchemes
{
    default none;
    div(phi,U) bounded Gauss linearUpwind grad(U);
    div(phi,nuTilda) bounded Gauss linearUpwind grad(nuTilda);
    div((nuEff*dev2(T(grad(U))))) Gauss linear;
}
laplacianSchemes { default Gauss linear corrected; }
interpolationSchemes { default linear; }
snGradSchemes { default corrected; }
wallDist { method meshWave; }
""")
    put(root, "system/fvSolution", "dictionary", "fvSolution", "solvers\n{\n" + TIGHT_SOLVERS + """}
SIMPLE { nNonOrthogonalCorrectors 1; consistent yes; }
relaxationFactors { equations { U 0.9; nuTilda 0.8; } }
""")
    put(root, "system/controlDict", "dictionary", "controlDict", control(400, 100))


def plate(root):
    # OpenFOAM tutorials/incompressible/simpleFoam/turbulentFlatPlate at 1/4
    # of its resolution (the tutorial is 96 + 449 x 385 at the level-5 y+ 1
    # grading 2200; its Allrun lists 9300 as the level-3 y+ 1 grading).
    put(root, "system/blockMeshDict", "dictionary", "blockMeshDict", """scale 0.333333333333333;
vertices
(
    (-1 0 0.3) (0 0 0.3) (6 0 0.3) (6 3 0.3) (0 3 0.3) (-1 3 0.3)
    (-1 0 0) (0 0 0) (6 0 0) (6 3 0) (0 3 0) (-1 3 0)
);
blocks
(
    hex (6 7 10 11 0 1 4 5) (24 96 1) simpleGrading (((1 9 0.15)(0.1 3 0.9)) 9300 1)
    hex (7 8 9 10 1 2 3 4) (112 96 1) simpleGrading (((0.1 22 1.5)(0.35 23 7)(0.515 12 1)) 9300 1)
);
edges ();
boundary
(
    inlet { type patch; faces ((6 0 5 11)); }
    outlet { type patch; faces ((8 9 3 2)); }
    topWall { type wall; faces ((11 5 4 10) (10 4 3 9)); }
    bottomWall { type wall; faces ((7 8 2 1)); }
    symmetry { type symmetryPlane; faces ((6 7 1 0)); }
    frontAndBack { type empty; faces ((0 1 4 5) (1 2 3 4) (6 11 10 7) (7 10 9 8)); }
);
""")
    E = ("frontAndBack", "type empty;")
    S = ("symmetry", "type symmetryPlane;")
    field(root, "U", "volVectorField", "U", "[0 1 -1 0 0 0 0]", "uniform (69.4 0 0)", [
        ("inlet", "type fixedValue; value uniform (69.4 0 0);"), ("outlet", "type zeroGradient;"),
        ("topWall", "type slip;"), ("bottomWall", "type fixedValue; value uniform (0 0 0);"), S, E])
    field(root, "p", "volScalarField", "p", "[0 2 -2 0 0 0 0]", "uniform 0", [
        ("inlet", "type zeroGradient;"), ("outlet", "type fixedValue; value uniform 0;"),
        ("topWall", "type zeroGradient;"), ("bottomWall", "type zeroGradient;"), S, E])
    field(root, "k", "volScalarField", "k", "[0 2 -2 0 0 0 0]", "uniform 1.08e-3", [
        ("inlet", "type fixedValue; value uniform 1.08e-3;"), ("outlet", "type zeroGradient;"),
        ("topWall", "type zeroGradient;"), ("bottomWall", "type kqRWallFunction; value uniform 1.08e-3;"), S, E])
    field(root, "omega", "volScalarField", "omega", "[0 0 -1 0 0 0 0]", "uniform 8675", [
        ("inlet", "type fixedValue; value uniform 8675;"), ("outlet", "type zeroGradient;"),
        ("topWall", "type zeroGradient;"), ("bottomWall", "type omegaWallFunction; value uniform 8675;"), S, E])
    field(root, "nut", "volScalarField", "nut", "[0 2 -1 0 0 0 0]", "uniform 1.25e-7", [
        ("inlet", "type calculated; value uniform 1.25e-7;"), ("outlet", "type calculated; value uniform 1.25e-7;"),
        ("topWall", "type calculated; value uniform 1.25e-7;"),
        ("bottomWall", "type nutUSpaldingWallFunction; value uniform 1.25e-7;"), S, E])
    turb(root, "kOmegaSST")
    transport(root, "1.388e-05")
    put(root, "system/fvSchemes", "dictionary", "fvSchemes", """ddtSchemes { default steadyState; }
gradSchemes { default Gauss linear; }
divSchemes
{
    default none;
    div(phi,U) bounded Gauss linearUpwind grad(U);
    div(phi,k) bounded Gauss limitedLinear 1;
    div(phi,omega) bounded Gauss limitedLinear 1;
    div((nuEff*dev2(T(grad(U))))) Gauss linear;
}
laplacianSchemes { default Gauss linear uncorrected; }
interpolationSchemes { default linear; }
snGradSchemes { default uncorrected; }
wallDist { method meshWave; }
""")
    put(root, "system/fvSolution", "dictionary", "fvSolution", """solvers
{
    p { solver GAMG; tolerance 1e-10; relTol 0.1; smoother DICGaussSeidel; nPreSweeps 0; nPostSweeps 2;
        cacheAgglomeration on; agglomerator faceAreaPair; nCellsInCoarsestLevel 10; mergeLevels 1; }
    U { solver smoothSolver; smoother symGaussSeidel; tolerance 1e-10; relTol 0.1; }
    "(omega|k)" { solver PBiCGStab; preconditioner DILU; tolerance 1e-10; relTol 0.1; }
}
SIMPLE
{
    nNonOrthogonalCorrectors 0;
    consistent yes;
    residualControl { p 1e-6; U 1e-6; "(k|omega)" 1e-6; }
}
relaxationFactors { equations { U 0.9; k 0.8; omega 0.8; } }
""")
    put(root, "system/controlDict", "dictionary", "controlDict", control(8000, 8000, 12, """functions
{
    forces
    {
        type forces; libs (forces); writeControl timeStep; writeInterval 1;
        patches (bottomWall); rho rhoInf; rhoInf 1; CofR (0 0 0); log false;
    }
}
"""))


CASES = {"sst_plate": sst_plate, "sst_skew": sst_skew, "sa_skew": sa_skew, "plate": plate}

if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    dest = sys.argv[1]
    names = sys.argv[2:] or list(CASES)
    for n in names:
        CASES[n](os.path.join(dest, n))
        print("wrote", os.path.join(dest, n))
