#!/usr/bin/env python3
"""Writes the OpenFOAM v2606 pimpleFoam MOVING-MESH parity cases of the FV path
(docs/fvpimple.md, stage 3b: solid-body motion of the whole mesh, no
non-conformal interface) into DEST/<name>/.

  mspin   M1: a closed 2D square box (16 x 16, empty front/back) rotating
          rigidly about z at omega 2, walls movingWallVelocity, fluid at rest
          in the lab frame -> spin-up. Fixed deltaT 0.005, 200 steps (2 rad),
          PISO mode (nOuterCorrectors 1, nCorrectors 2), Gauss linear.
          dynamicMotionSolverFvMesh, the flat tutorial spelling (motionSolver
          solidBody; solidBodyMotionFunction rotatingMotion; origin axis omega),
          correctPhi by default (mesh.dynamic()), checkMeshCourantNo yes.
  mduct   M2: a 3D duct of 26.6 deg sheared cross-section (10 x 8 x 8)
          rotating about an OFF-CENTRE x axis at omega 4: inlet fixedValue
          (1 0 0) / outlet inletOutlet + fixedValue p ON the rotating mesh,
          lateral walls movingWallVelocity. nOuterCorrectors 3, nCorrectors 1,
          nNonOrthogonalCorrectors 1, the M17 relaxation set and schemes,
          correctPhi yes, adjustTimeStep maxCo 0.5 maxDeltaT 0.02,
          adjustableRunTime. dynamicMotionSolverFvMesh with `solver solidBody;
          solidBodyCoeffs { cellZone rotor; ... rotatingMotionCoeffs { } }`.
  msst    M3: the M2 duct with kOmegaSST, kqRWallFunction / omegaWallFunction /
          nutkWallFunction on the rotating walls (the M17 wall set), the M17
          fvSchemes verbatim and its fvSolution shape, wallDist meshWave,
          dynamicMultiMotionSolverFvMesh in the M17 dictionary's exact
          spelling (one entry, cellZone rotor = every cell).

Linear solves are TIGHT (relTol 0) on every entry (pcorr included), so each
corrector is the algorithm's own.

usage: make_motion_cases.py DEST [name ...]
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import make_cases as mc  # noqa: E402
import make_pimple_cases as mp  # noqa: E402
import make_ras_cases as mr  # noqa: E402

put = mc.put

FORCES = """functions
{{
    forcesWalls
    {{
        type            forces;
        libs            (forces);
        writeControl    timeStep;
        writeInterval   1;
        patches         (walls);
        rho             rhoInf;
        rhoInf          1.225;
        CofR            {cofr};
    }}
}}
"""


def laminar(root, nu):
    put(root, "constant/transportProperties", "dictionary", "transportProperties",
        f"transportModel Newtonian;\nnu {nu};\n")
    put(root, "constant/momentumTransport", "dictionary", "momentumTransport", "simulationType laminar;\n")
    put(root, "constant/turbulenceProperties", "dictionary", "turbulenceProperties", "simulationType laminar;\n")


def mspin(root):
    put(root, "system/blockMeshDict", "dictionary", "blockMeshDict", """scale 1;
vertices
(
    (-0.5 -0.5 0) (0.5 -0.5 0) (0.5 0.5 0) (-0.5 0.5 0)
    (-0.5 -0.5 0.1) (0.5 -0.5 0.1) (0.5 0.5 0.1) (-0.5 0.5 0.1)
);
blocks ( hex (0 1 2 3 4 5 6 7) (16 16 1) simpleGrading (1 1 1) );
edges ();
boundary
(
    walls { type wall; faces ((3 7 6 2) (0 4 7 3) (2 6 5 1) (1 5 4 0)); }
    frontAndBack { type empty; faces ((0 3 2 1) (4 5 6 7)); }
);
""")
    put(root, "0/U", "volVectorField", "U", """dimensions [0 1 -1 0 0 0 0];
internalField uniform (0 0 0);
boundaryField
{
    walls { type movingWallVelocity; value uniform (0 0 0); }
    frontAndBack { type empty; }
}
""")
    put(root, "0/p", "volScalarField", "p", """dimensions [0 2 -2 0 0 0 0];
internalField uniform 0;
boundaryField
{
    walls { type zeroGradient; }
    frontAndBack { type empty; }
}
""")
    laminar(root, 0.01)
    put(root, "constant/dynamicMeshDict", "dictionary", "dynamicMeshDict", """dynamicFvMesh   dynamicMotionSolverFvMesh;
motionSolverLibs (fvMotionSolvers);
motionSolver    solidBody;
solidBodyMotionFunction rotatingMotion;
origin          (0 0 0);
axis            (0 0 1);
omega           2;
""")
    put(root, "system/fvSchemes", "dictionary", "fvSchemes",
        mp.schemes("default Gauss linear;", "Gauss linear", "corrected"))
    put(root, "system/fvSolution", "dictionary", "fvSolution", f"""solvers
{{
    "pcorr.*" {{ {mp.P_TIGHT} }}
    p {{ {mp.P_TIGHT} }}
    pFinal {{ {mp.P_TIGHT} }}
    UFinal {{ {mp.U_TIGHT} }}
}}
PIMPLE
{{
    nOuterCorrectors 1;
    nCorrectors 2;
    nNonOrthogonalCorrectors 0;
    checkMeshCourantNo yes;
    pRefCell 0;
    pRefValue 0;
}}
""")
    mp.control(root, 1, 0.005, "timeStep", 50, functions=FORCES.format(cofr="(0 0 0)"))


DUCT_MESH = """scale 1;
vertices
(
    (0 -0.3 -0.2) (1 -0.3 -0.2) (1 0.1 -0.2) (0 0.1 -0.2)
    (0 -0.1 0.2) (1 -0.1 0.2) (1 0.3 0.2) (0 0.3 0.2)
);
blocks ( hex (0 1 2 3 4 5 6 7) rotor (10 8 8) simpleGrading (1 1 1) );
edges ();
boundary
(
    inlet { type patch; faces ((0 4 7 3)); }
    outlet { type patch; faces ((2 6 5 1)); }
    walls { type wall; faces ((3 7 6 2) (1 5 4 0) (0 3 2 1) (4 5 6 7)); }
);
"""

DUCT_U = """dimensions [0 1 -1 0 0 0 0];
internalField uniform (1 0 0);
boundaryField
{
    inlet { type fixedValue; value uniform (1 0 0); }
    outlet { type inletOutlet; inletValue uniform (0 0 0); value uniform (1 0 0); }
    walls { type movingWallVelocity; value uniform (0 0 0); }
}
"""

DUCT_P = """dimensions [0 2 -2 0 0 0 0];
internalField uniform 0;
boundaryField
{
    inlet { type zeroGradient; }
    outlet { type fixedValue; value uniform 0; }
    walls { type zeroGradient; }
}
"""

# The M17 case's fvSolution with a pcorr entry, every solve tight.
M17_MOVING_SOLUTION = """solvers
{{
    "pcorr.*"
    {{
        {p}
    }}
""" + mp.M17_SOLUTION[len("solvers\n{{\n"):]


def mduct(root):
    put(root, "system/blockMeshDict", "dictionary", "blockMeshDict", DUCT_MESH)
    put(root, "0/U", "volVectorField", "U", DUCT_U)
    put(root, "0/p", "volScalarField", "p", DUCT_P)
    laminar(root, 0.005)
    put(root, "constant/dynamicMeshDict", "dictionary", "dynamicMeshDict", """dynamicFvMesh   dynamicMotionSolverFvMesh;
solver          solidBody;
solidBodyCoeffs
{
    cellZone        rotor;
    solidBodyMotionFunction rotatingMotion;
    rotatingMotionCoeffs
    {
        origin      (0 0.02 0.01);
        axis        (1 0 0);
        omega       4;
    }
}
""")
    put(root, "system/fvSchemes", "dictionary", "fvSchemes", mp.schemes(**mp.M17_SCHEMES))
    put(root, "system/fvSolution", "dictionary", "fvSolution",
        M17_MOVING_SOLUTION.format(p=mp.P_TIGHT, u=mp.U_TIGHT, outer=3, nonorth=1, extra=""))
    mp.control(root, 0.5, 0.002, "adjustableRunTime", 0.25,
               extra="adjustTimeStep yes;\nmaxCo 0.5;\nmaxDeltaT 0.02;\n",
               functions=FORCES.format(cofr="(0.5 0.02 0.01)"))


# The M17 case's fvSchemes, verbatim.
M17_FVSCHEMES = """ddtSchemes
{
    default         Euler;
}

gradSchemes
{
    default         cellLimited Gauss linear 1;
}

divSchemes
{
    default         none;
    div(phi,U)      Gauss linearUpwind grad(U);
    div(phi,k)      Gauss upwind;
    div(phi,omega)  Gauss upwind;
    div((nuEff*dev2(T(grad(U))))) Gauss linear;
}

laplacianSchemes
{
    default         Gauss linear limited corrected 0.33;
}

interpolationSchemes
{
    default         linear;
}

snGradSchemes
{
    default         limited corrected 0.33;
}

wallDist
{
    method          meshWave;
}
"""

# The M17 case's dynamicMeshDict spelling: one entry whose cellZone is every cell.
M17_DYNAMIC = """dynamicFvMesh   dynamicMultiMotionSolverFvMesh;

dynamicMultiMotionSolverFvMeshCoeffs
{
    fwd
    {
        solver   solidBody;
        cellZone rotor;
        solidBodyCoeffs
        {
            solidBodyMotionFunction rotatingMotion;
            rotatingMotionCoeffs
            {
                origin  (0 0.02 0.01);
                axis    (1 0 0);
                omega   -4;
            }
        }
    }
}
"""


def msst(root):
    put(root, "system/blockMeshDict", "dictionary", "blockMeshDict", DUCT_MESH)
    put(root, "0/U", "volVectorField", "U", DUCT_U)
    put(root, "0/p", "volScalarField", "p", DUCT_P)
    mr.field(root, "k", "volScalarField", "k", "[0 2 -2 0 0 0 0]", "uniform 0.00375", [
        ("inlet", "type fixedValue; value uniform 0.00375;"),
        ("outlet", "type inletOutlet; inletValue $internalField; value $internalField;"),
        ("walls", "type kqRWallFunction; value $internalField;")])
    mr.field(root, "omega", "volScalarField", "omega", "[0 0 -1 0 0 0 0]", "uniform 20", [
        ("inlet", "type fixedValue; value uniform 20;"),
        ("outlet", "type inletOutlet; inletValue $internalField; value $internalField;"),
        ("walls", "type omegaWallFunction; value $internalField;")])
    mr.field(root, "nut", "volScalarField", "nut", "[0 2 -1 0 0 0 0]", "uniform 0", [
        ("inlet", "type calculated; value uniform 0;"), ("outlet", "type calculated; value uniform 0;"),
        ("walls", "type nutkWallFunction; value uniform 0;")])
    mr.turb(root, "kOmegaSST")
    mr.transport(root, "1e-04")
    put(root, "constant/dynamicMeshDict", "dictionary", "dynamicMeshDict", M17_DYNAMIC)
    put(root, "system/fvSchemes", "dictionary", "fvSchemes", M17_FVSCHEMES)
    put(root, "system/fvSolution", "dictionary", "fvSolution",
        M17_MOVING_SOLUTION.format(p=mp.P_TIGHT, u=mp.S_TIGHT, outer=3, nonorth=1, extra=""))
    mp.control(root, 0.5, 0.002, "adjustableRunTime", 0.25,
               extra="adjustTimeStep yes;\nmaxCo 1;\nmaxDeltaT 0.02;\n",
               functions=FORCES.format(cofr="(0.5 0.02 0.01)"))


CASES = {"mspin": mspin, "mduct": mduct, "msst": msst}

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
