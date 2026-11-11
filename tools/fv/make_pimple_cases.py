#!/usr/bin/env python3
"""Writes the OpenFOAM v2606 pimpleFoam parity cases of the FV path
(docs/fvpimple.md, stage 3a: static mesh) into DEST/<name>/. Meshes and 0/
fields are the simpleFoam parity cases' (make_cases.py, make_ras_cases.py);
system/ is pimpleFoam's.

  pcavity       P1: the IF1a lid-driven cavity, fixed deltaT 0.002, PISO mode
                (nOuterCorrectors 1, nCorrectors 2), Gauss linear, no relaxation
  pcavity_skew  P2: the 26.6 deg cavity, nOuterCorrectors 3, nCorrectors 1,
                nNonOrthogonalCorrectors 1, the M17 case's fvSolution shape
                (pFinal { $p; }, "(U|k|omega)Final" { $U; }, p 0.7 / pFinal 1,
                "U.*" 0.7, turbulenceOnFinalIterOnly, correctPhi) and schemes
                (cellLimited Gauss linear 1 default, linearUpwind,
                limited corrected 0.33); a forces object on the fixed walls
  pchannel      P3: the IF1c graded channel (inletOutlet, fixedValue p, slip),
                adjustTimeStep yes maxCo 2 maxDeltaT 0.03, writeControl
                adjustableRunTime 0.5 (deltaT is also bent to the write times),
                nOuterCorrectors 2, nCorrectors 2, nNonOrthogonalCorrectors 1,
                p 0.7 / pFinal 1, U 0.7 / UFinal 1
  pduct         P3b: the IF1d 3D duct (freestreamVelocity / freestreamPressure,
                symmetryPlane, cyclic z), fixed deltaT, PISO mode with 1
                non-orthogonal corrector
  psst_plate    P4: the IF-T flat plate, kOmegaSST with wall functions,
                nOuterCorrectors 3, turbulence on the final corrector only
                (the v2606 default the M17 case runs with), "(k|omega).*" 0.7
  psst_skew     P4b: the IF-T skewed channel, kOmegaSST, nOuterCorrectors 2,
                turbOnFinalIterOnly no (k/omega and kFinal/omegaFinal both used),
                1 non-orthogonal corrector, exactDistance

Linear solves are TIGHT (relTol 0) on every entry, Final or not, so each
corrector is the algorithm's own.

usage: make_pimple_cases.py DEST [name ...]
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import make_cases as mc  # noqa: E402
import make_ras_cases as mr  # noqa: E402

put = mc.put


def control(root, end, dt, write_control, write, extra="", functions=""):
    put(root, "system/controlDict", "dictionary", "controlDict", f"""application pimpleFoam;
startFrom startTime;
startTime 0;
stopAt endTime;
endTime {end};
deltaT {dt};
writeControl {write_control};
writeInterval {write};
purgeWrite 0;
writeFormat ascii;
writePrecision 17;
timeFormat general;
timePrecision 10;
runTimeModifiable false;
{extra}{functions}""")


def schemes(grad, div_u, lap, extra_div="", wall_dist=""):
    return f"""ddtSchemes      {{ default Euler; }}
gradSchemes     {{ {grad} }}
divSchemes
{{
    default         none;
    div(phi,U)      {div_u};
{extra_div}    div((nuEff*dev2(T(grad(U))))) Gauss linear;
}}
laplacianSchemes {{ default Gauss linear {lap}; }}
interpolationSchemes {{ default linear; }}
snGradSchemes   {{ default {lap}; }}
{wall_dist}"""


P_TIGHT = "solver PCG; preconditioner DIC; tolerance 1e-13; relTol 0; maxIter 5000;"
U_TIGHT = "solver smoothSolver; smoother symGaussSeidel; tolerance 1e-14; relTol 0; maxIter 5000;"
S_TIGHT = "solver PBiCGStab; preconditioner DILU; tolerance 1e-14; relTol 0; maxIter 5000;"


def pcavity(root):
    mc.cavity(root, False)
    put(root, "system/fvSchemes", "dictionary", "fvSchemes",
        schemes("default Gauss linear;", "Gauss linear", "corrected"))
    put(root, "system/fvSolution", "dictionary", "fvSolution", f"""solvers
{{
    p {{ {P_TIGHT} }}
    pFinal {{ {P_TIGHT} }}
    UFinal {{ {U_TIGHT} }}
}}
PIMPLE
{{
    nOuterCorrectors 1;
    nCorrectors 2;
    nNonOrthogonalCorrectors 0;
    pRefCell 0;
    pRefValue 0;
}}
""")
    control(root, 0.2, 0.002, "timeStep", 20)


M17_SOLUTION = """solvers
{{
    p
    {{
        {p}
    }}
    pFinal
    {{
        $p;
        relTol          0;
    }}
    "(U|k|omega)"
    {{
        {u}
    }}
    "(U|k|omega)Final"
    {{
        $U;
        relTol          0;
    }}
}}
PIMPLE
{{
    correctPhi                  yes;
    nOuterCorrectors            {outer};
    nCorrectors                 1;
    nNonOrthogonalCorrectors    {nonorth};
    turbulenceOnFinalIterOnly   no;
{extra}}}
relaxationFactors
{{
    fields
    {{
        p       0.7;
        pFinal  1;
    }}
    equations
    {{
        "U.*"          0.7;
        "(k|omega).*"  0.7;
    }}
}}
"""

M17_SCHEMES = dict(grad="default cellLimited Gauss linear 1;", div_u="Gauss linearUpwind grad(U)",
                   lap="limited corrected 0.33")


def pcavity_skew(root):
    mc.cavity(root, True)
    put(root, "system/fvSchemes", "dictionary", "fvSchemes", schemes(**M17_SCHEMES))
    put(root, "system/fvSolution", "dictionary", "fvSolution",
        M17_SOLUTION.format(p=P_TIGHT, u=U_TIGHT, outer=3, nonorth=1, extra="    pRefCell 0;\n    pRefValue 0;\n"))
    control(root, 0.2, 0.002, "timeStep", 20, functions="""functions
{
    forcesWalls
    {
        type            forces;
        libs            (forces);
        writeControl    timeStep;
        writeInterval   1;
        patches         (fixedWalls);
        rho             rhoInf;
        rhoInf          1.225;
        CofR            (0.05 0.05 0);
    }
}
""")


def pchannel(root):
    mc.channel(root)
    put(root, "system/fvSchemes", "dictionary", "fvSchemes",
        schemes("default Gauss linear; grad(U) cellLimited Gauss linear 1;", "Gauss linearUpwind grad(U)", "corrected"))
    put(root, "system/fvSolution", "dictionary", "fvSolution", f"""solvers
{{
    "p.*" {{ {P_TIGHT} }}
    "U.*" {{ {U_TIGHT} }}
}}
PIMPLE
{{
    nOuterCorrectors 2;
    nCorrectors 2;
    nNonOrthogonalCorrectors 1;
}}
relaxationFactors
{{
    fields {{ p 0.7; pFinal 1; }}
    equations {{ U 0.7; UFinal 1; }}
}}
""")
    control(root, 2, 0.001, "adjustableRunTime", 0.5, extra="adjustTimeStep yes;\nmaxCo 2;\nmaxDeltaT 0.03;\n")


def pduct(root):
    mc.duct(root)
    put(root, "system/fvSchemes", "dictionary", "fvSchemes",
        schemes("default Gauss linear;", "Gauss linear", "corrected"))
    put(root, "system/fvSolution", "dictionary", "fvSolution", f"""solvers
{{
    p {{ {P_TIGHT} }}
    pFinal {{ {P_TIGHT} }}
    UFinal {{ {U_TIGHT} }}
}}
PIMPLE
{{
    nOuterCorrectors 1;
    nCorrectors 2;
    nNonOrthogonalCorrectors 1;
}}
""")
    control(root, 1.2, 0.02, "runTime", 0.4)


def psst_plate(root):
    mr.sst_plate(root)
    put(root, "system/fvSchemes", "dictionary", "fvSchemes",
        schemes("default Gauss linear;", "Gauss linearUpwind grad(U)", "corrected",
                extra_div="    div(phi,k)      Gauss linearUpwind grad(k);\n"
                          "    div(phi,omega)  Gauss linearUpwind grad(omega);\n",
                wall_dist="wallDist { method meshWave; }\n"))
    put(root, "system/fvSolution", "dictionary", "fvSolution",
        M17_SOLUTION.format(p=P_TIGHT, u=S_TIGHT, outer=3, nonorth=0, extra=""))
    control(root, 0.5, 0.005, "timeStep", 20)


def psst_skew(root):
    mr.sst_skew(root)
    put(root, "system/fvSchemes", "dictionary", "fvSchemes",
        schemes("default Gauss linear; grad(k) cellLimited Gauss linear 1;", "Gauss linearUpwind grad(U)", "corrected",
                extra_div="    div(phi,k)      Gauss linearUpwind grad(k);\n"
                          "    div(phi,omega)  Gauss upwind;\n",
                wall_dist="wallDist { method exactDistance; }\n"))
    put(root, "system/fvSolution", "dictionary", "fvSolution", f"""solvers
{{
    "p.*" {{ {P_TIGHT} }}
    "(U|k|omega).*" {{ {S_TIGHT} }}
}}
PIMPLE
{{
    nOuterCorrectors 2;
    nCorrectors 1;
    nNonOrthogonalCorrectors 1;
    turbOnFinalIterOnly no;
}}
relaxationFactors
{{
    fields {{ p 0.7; pFinal 1; }}
    equations {{ U 0.8; UFinal 1; k 0.7; omega 0.7; "(k|omega)Final" 1; }}
}}
""")
    control(root, 0.4, 0.004, "timeStep", 20)


CASES = {"pcavity": pcavity, "pcavity_skew": pcavity_skew, "pchannel": pchannel, "pduct": pduct,
         "psst_plate": psst_plate, "psst_skew": psst_skew}

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
