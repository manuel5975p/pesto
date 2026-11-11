#!/usr/bin/env python3
"""Writes the OpenFOAM v2606 pimpleFoam cyclicAMI cases of the FV path (stage
3c, docs/fvpimple.md "cyclicAMI") into DEST/<name>/: the meshes of
tools/fv/make_ami_cases.py (two unmerged block faces declared cyclicAMI in the
M17 spelling) plus the flow dictionaries. Serial references:
tools/fv/run_foam.sh; checked-in data: tools/fv/make_pimple_test_data.sh.

  aplanar  A1: 2D channel split at x = 1 by a planar cyclicAMI, 10 x 8 against
           10 x 12 cells (non-matching), inlet fixedValue / outlet inletOutlet +
           fixedValue p, the flow crosses the interface; deltaT 0.005 fixed,
           (2, 2, 1), the M17 schemes.
  amatch   A1 degenerate: 8 against 8 cells, faces match point for point.
  amatch_int  the same geometry as one conformal block (ordinary faces): with
           `ddtCorr` (the default) OpenFOAM's amatch is NOT this case
           (ddtCouplingCoeff is 0 on a cyclicAMI patch); a scratch comparison.
  acyl     A2: 3D static, a cylindrical cyclicAMI between a 20-gon and a 28-gon
           of flat quads, non-matching in z too (weight sums 1.0037 .. 1.0044
           before normalisation), non-orthogonal cells; (3, 1, 1) with the M17
           relaxation set.
  amix2d   A3: 2D mixer -- the rotor disc (cellZone rotor, a paddle with
           movingWallVelocity) turns at omega 4 behind the circular cyclicAMI
           inside a static box; laminar, (3, 1, 1), the M17 relaxation set,
           correctPhi yes, adjustTimeStep (maxCo 0.5), 0.45 s = 1.8 rad.
  amix3d   A4: the mixer extruded (rotor 2, stator 3 cells in z: the interface
           is a cylinder of partly overlapping quads), kOmegaSST, kqR / omega /
           nutk wall functions on every wall, meshWave, the M17 fvSchemes
           verbatim, dynamicMultiMotionSolverFvMesh in the M17 spelling,
           omega -4, maxCo 1.
  apartial A1p: aplanar with the right block shifted to y in [0.3, 1.3]: 2
           source and 3 target faces have NO overlap (sum(weights) 0, neighbour
           value 0, no lowWeightCorrection), one each a partial one. pimpleFoam
           runs it only without FOAM_SIGFPE (1 / UEqn.A() divides by the patch
           value 0 there): run_foam.sh unsets the trap for it.
  amixtilt A5: amix3d with the rotor turning about (0.3 0 1), 16.7 degrees off
           the cylinder's axis: the weight sums fall from 1 to 0 on both sides
           over the run (average 0.54 at the end, faces without overlap from
           step 49 of 74) -- kOmegaSST, correctPhi and the motion on faces of
           low and of no overlap. Without FOAM_SIGFPE, as apartial.

usage: make_ami_flow_cases.py DEST [name ...]
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import make_ami_cases as mk  # noqa: E402

SCHEMES = """ddtSchemes
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
%s    div((nuEff*dev2(T(grad(U))))) Gauss linear;
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
%s"""

SOLVERS = """solvers
{
    "pcorr.*"
    {
        solver GAMG; smoother GaussSeidel; tolerance 1e-13; relTol 0; maxIter 5000;
    }
    p
    {
        solver GAMG; smoother GaussSeidel; tolerance 1e-13; relTol 0; maxIter 5000;
    }
    pFinal
    {
        $p;
        relTol          0;
    }
    "(U|k|omega)"
    {
        solver smoothSolver; smoother symGaussSeidel; tolerance 1e-14; relTol 0; maxIter 5000;
    }
    "(U|k|omega)Final"
    {
        $U;
        relTol          0;
    }
}
"""

M17_RELAX = """relaxationFactors
{
    fields
    {
        p       0.7;
        pFinal  1;
    }
    equations
    {
        "U.*"          0.7;
        "(k|omega).*"  0.7;
    }
}
"""


def put(root, rel, cls, obj, body):
    mk.put(root, rel, cls, obj, body)


def common(root, nu, ras=False):
    put(root, "constant/transportProperties", "dictionary", "transportProperties",
        f"transportModel Newtonian;\nnu {nu};\n")
    sim = ("simulationType RAS;\nRAS\n{\n    RASModel kOmegaSST;\n    turbulence on;\n    printCoeffs on;\n}\n"
           if ras else "simulationType laminar;\n")
    put(root, "constant/momentumTransport", "dictionary", "momentumTransport", sim)
    put(root, "constant/turbulenceProperties", "dictionary", "turbulenceProperties", sim)
    put(root, "system/fvSchemes", "dictionary", "fvSchemes",
        SCHEMES % (("    div(phi,k)      Gauss upwind;\n    div(phi,omega)  Gauss upwind;\n" if ras else ""),
                   ("\nwallDist\n{\n    method          meshWave;\n}\n" if ras else "")))


def control(root, end, dt, write, extra=""):
    put(root, "system/controlDict", "dictionary", "controlDict", f"""application pimpleFoam;
startFrom startTime;
startTime 0;
stopAt endTime;
endTime {end};
deltaT {dt};
{write}
purgeWrite 0;
writeFormat ascii;
writePrecision 17;
timeFormat general;
timePrecision 10;
runTimeModifiable false;
{extra}""")


def channel_fields(root, walls):
    put(root, "0/U", "volVectorField", "U", f"""dimensions [0 1 -1 0 0 0 0];
internalField uniform (1 0 0);
boundaryField
{{
    inlet {{ type fixedValue; value uniform (1 0 0); }}
    outlet {{ type inletOutlet; inletValue uniform (0 0 0); value uniform (1 0 0); }}
    {walls} {{ type noSlip; }}
    #includeEtc "caseDicts/setConstraintTypes"
}}
""")
    put(root, "0/p", "volScalarField", "p", f"""dimensions [0 2 -2 0 0 0 0];
internalField uniform 0;
boundaryField
{{
    inlet {{ type zeroGradient; }}
    outlet {{ type fixedValue; value uniform 0; }}
    {walls} {{ type zeroGradient; }}
    #includeEtc "caseDicts/setConstraintTypes"
}}
""")


def planar(root, mesh):
    mesh(root)
    common(root, 0.01)
    channel_fields(root, "walls")
    put(root, "system/fvSolution", "dictionary", "fvSolution", SOLVERS + """PIMPLE
{
    nOuterCorrectors 2;
    nCorrectors 2;
    nNonOrthogonalCorrectors 1;
}
relaxationFactors
{
    fields { p 0.7; pFinal 1; }
    equations { U 0.7; UFinal 1; }
}
""")
    control(root, 0.5, 0.005, "writeControl timeStep;\nwriteInterval 20;")


def aplanar(root):
    planar(root, mk.mesh_ami_planar)


def amatch(root):
    planar(root, mk.mesh_ami_match)


def amatch_int(root):
    planar(root, mk.mesh_ami_match_int)


def apartial(root):
    planar(root, mk.mesh_ami_partial)


def acyl(root):
    mk.mesh_ami_cyl(root)
    common(root, 0.01)
    channel_fields(root, '"(walls|rotorEnds)"')
    put(root, "system/fvSolution", "dictionary", "fvSolution", SOLVERS + """PIMPLE
{
    nOuterCorrectors 3;
    nCorrectors 1;
    nNonOrthogonalCorrectors 1;
}
""" + M17_RELAX)
    control(root, 0.2, 0.002, "writeControl timeStep;\nwriteInterval 20;")


FORCES = """functions
{
    forcesBlade
    {
        type            forces;
        libs            (forces);
        writeControl    timeStep;
        writeInterval   1;
        patches         (blade);
        rho             rhoInf;
        rhoInf          1.225;
        CofR            (0 0 0);
    }
}
"""

MIX_PIMPLE = """PIMPLE
{
    correctPhi                  yes;
    nOuterCorrectors            3;
    nCorrectors                 1;
    nNonOrthogonalCorrectors    1;
    turbulenceOnFinalIterOnly   no;
    pRefCell                    0;
    pRefValue                   0;
}
"""


def mixer_fields(root, moving, static, constraints):
    put(root, "0/U", "volVectorField", "U", f"""dimensions [0 1 -1 0 0 0 0];
internalField uniform (0 0 0);
boundaryField
{{
    {static} {{ type noSlip; }}
    {moving} {{ type movingWallVelocity; value uniform (0 0 0); }}
    "AMI.*" {{ type cyclicAMI; }}
{constraints}}}
""")
    put(root, "0/p", "volScalarField", "p", f"""dimensions [0 2 -2 0 0 0 0];
internalField uniform 0;
boundaryField
{{
    {static} {{ type zeroGradient; }}
    {moving} {{ type zeroGradient; }}
    "AMI.*" {{ type cyclicAMI; }}
{constraints}}}
""")


def amix2d(root):
    mk.mesh_ami_mixer2d(root)
    common(root, 0.01)
    mixer_fields(root, "blade", "walls", '    #includeEtc "caseDicts/setConstraintTypes"\n')
    put(root, "constant/dynamicMeshDict", "dictionary", "dynamicMeshDict", """dynamicFvMesh   dynamicMotionSolverFvMesh;
solver          solidBody;
solidBodyCoeffs
{
    cellZone        rotor;
    solidBodyMotionFunction rotatingMotion;
    rotatingMotionCoeffs
    {
        origin      (0 0 0);
        axis        (0 0 1);
        omega       4;
    }
}
""")
    put(root, "system/fvSolution", "dictionary", "fvSolution", SOLVERS + MIX_PIMPLE + M17_RELAX)
    control(root, 0.45, 0.001, "writeControl adjustableRunTime;\nwriteInterval 0.15;",
            "adjustTimeStep yes;\nmaxCo 0.5;\nmaxDeltaT 0.01;\n" + FORCES)


def amix3d(root):
    mk.mesh_mixer(root, 0.3, 2, 3, True)
    common(root, 1e-4, ras=True)
    walls = '"(walls|blade|rotorEnds)"'
    mixer_fields(root, '"(blade|rotorEnds)"', "walls", "")
    put(root, "0/k", "volScalarField", "k", f"""dimensions [0 2 -2 0 0 0 0];
internalField uniform 0.00375;
boundaryField
{{
    {walls} {{ type kqRWallFunction; value $internalField; }}
    "AMI.*" {{ type cyclicAMI; }}
}}
""")
    put(root, "0/omega", "volScalarField", "omega", f"""dimensions [0 0 -1 0 0 0 0];
internalField uniform 20;
boundaryField
{{
    {walls} {{ type omegaWallFunction; value $internalField; }}
    "AMI.*" {{ type cyclicAMI; }}
}}
""")
    put(root, "0/nut", "volScalarField", "nut", f"""dimensions [0 2 -1 0 0 0 0];
internalField uniform 0;
boundaryField
{{
    {walls} {{ type nutkWallFunction; value uniform 0; }}
    "AMI.*" {{ type cyclicAMI; }}
}}
""")
    # the M17 dictionary's spelling, one rotor
    put(root, "constant/dynamicMeshDict", "dictionary", "dynamicMeshDict", """dynamicFvMesh   dynamicMultiMotionSolverFvMesh;

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
                origin  (0 0 0);
                axis    (0 0 1);
                omega   -4;
            }
        }
    }
}
""")
    put(root, "system/fvSolution", "dictionary", "fvSolution", SOLVERS + MIX_PIMPLE + M17_RELAX)
    control(root, 0.45, 0.001, "writeControl adjustableRunTime;\nwriteInterval 0.15;",
            "adjustTimeStep yes;\nmaxCo 1;\nmaxDeltaT 0.01;\n" + FORCES)


def amixtilt(root):
    amix3d(root)
    # the rotor turns about an axis 16.7 degrees off the cylinder's: it wobbles out of the stator's hole
    with open(os.path.join(root, "constant/dynamicMeshDict")) as f:
        text = f.read()
    assert "axis    (0 0 1);" in text
    with open(os.path.join(root, "constant/dynamicMeshDict"), "w") as f:
        f.write(text.replace("axis    (0 0 1);", "axis    (0.3 0 1);"))


CASES = {"aplanar": aplanar, "amatch": amatch, "amatch_int": amatch_int, "apartial": apartial, "acyl": acyl,
         "amix2d": amix2d, "amix3d": amix3d, "amixtilt": amixtilt}

if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    dest = sys.argv[1]
    for n in sys.argv[2:] or list(CASES):
        if n not in CASES:
            sys.exit(f"unknown case {n} (one of {', '.join(CASES)})")
        CASES[n](os.path.join(dest, n))
        print("wrote", os.path.join(dest, n))
