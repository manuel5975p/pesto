#!/usr/bin/env python3
"""OpenFOAM parity cases on the dns REVOLVED WEDGE mesh: sph1e4 (RANS, Re 1e4)
and sph100 (laminar, Re 100). masterplan_flowrans section 8; docs/rans.md
"OpenFOAM parity".

    sph1e4.py case  <dns>_mesh.bin CASEDIR MODEL RE [--ft2 true|false]
                    [--np N] [--iters N] [--u-scheme linear|linearUpwind]
    sph1e4.py run   CASEDIR [--np N]
    sph1e4.py read  CASEDIR            # the OpenFOAM observables, JSON
    sph1e4.py dns   <dns-out-prefix>   # the dns observables, JSON

MODEL: sa (SpalartAllmaras { ft2 true; Ct3 1.2; Ct4 0.5; } -- NOT "SA-neg":
OpenFOAM bounds nuTilda >= 0 and lacks TMR's grad rho term), sst (kOmegaSST,
2003, with its -2/3 divU terms), laminar. Tier B: simpleFoam, incompressible
(the M -> 0 limit dns is extrapolated to). Units: D = 1, U = 1, nu = 1/RE;
the mesh is the dns mesh scaled by 1/D_box (dnsmesh.py), never regenerated.

BCs (section 8.5): wall U noSlip, nuTilda/nut/k fixedValue 0, omega a
NONUNIFORM fixedValue 60 nu/(beta1 d1^2) from dns's own d1 (the same
face-Dirichlet image dns uses; not omegaWallFunction); far U
freestreamVelocity, p freestreamPressure, scalars inletOutlet at the
freestream units of decision 4 (nuTilda 3 nu; k 2.25e-7 U^2, omega =
k/(0.009 nu)). Prt 0.9 (OpenFOAM's default is 1.0). Schemes: central Gauss
linear for U (default since 2026-10-26: the parity reference, it matches
dns's dissipation-free KG mean flow) or, with --u-scheme linearUpwind, the
original reference; MEASURED, the choice moves
OpenFOAM's own SST cd by +3.7% (L1, Re 1e3) / +2.8% (L2, Re 1e4) and L_r
by -28% / -14%, so a parity number names its scheme (docs/validation.md).
First-order upwind for the turbulence (as dns), SIMPLE consistent,
relaxation U 0.9 / turbulence 0.8. Wall distance exactDistance.
"""
import json
import math
import os
import re
import subprocess
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import dnsmesh  # noqa: E402

FOAM = "/opt/OpenFOAM/OpenFOAM-v2606/etc/bashrc"
BETA1 = 0.075


def hdr(cls, obj, loc):
    return ("FoamFile\n{\n    version 2.0;\n    format ascii;\n"
            f"    class {cls};\n    location \"{loc}\";\n    object {obj};\n}}\n\n")


def write(case, path, text):
    p = os.path.join(case, path)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    with open(p, "w") as f:
        f.write(text)


U_SCHEMES = {"linearUpwind": "bounded Gauss linearUpwind grad(U)",
             "linear": "bounded Gauss linear"}


def make_case(meshbin, case, model, re_, ft2=True, iters=8000,
              uscheme="linear"):
    m = dnsmesh.read_mesh(meshbin)
    cellij, d1 = dnsmesh.write_polymesh(case, m)
    nu = 1.0 / re_
    wedge_deg = math.degrees(m["dphi"])
    aref = math.pi * 0.25 * wedge_deg / 360.0
    json.dump({"model": model, "re": re_, "nu": nu, "wedgeDeg": wedge_deg,
               "aref": aref, "cellij": cellij, "nTheta": m["ntheta"],
               "ny": m["ny"]},
              open(os.path.join(case, "parity.json"), "w"))
    write(case, "system/controlDict", hdr("dictionary", "controlDict", "system") + f"""
application simpleFoam;
startFrom latestTime;
stopAt endTime;
endTime {iters};
deltaT 1;
writeControl timeStep;
writeInterval {iters};
writeFormat ascii;
writePrecision 12;
runTimeModifiable false;
functions
{{
    forces1
    {{
        type forces; libs (forces); writeControl timeStep; writeInterval 1;
        patches (sphere); rho rhoInf; rhoInf 1; CofR (0 0 0); log false;
    }}
}}
""")
    write(case, "system/fvSchemes", hdr("dictionary", "fvSchemes", "system") + """
ddtSchemes { default steadyState; }
gradSchemes { default Gauss linear; }
divSchemes
{
    default none;
    div(phi,U) @USCHEME@;
    div(phi,nuTilda) bounded Gauss upwind;
    div(phi,k) bounded Gauss upwind;
    div(phi,omega) bounded Gauss upwind;
    div((nuEff*dev2(T(grad(U))))) Gauss linear;
}
laplacianSchemes { default Gauss linear corrected; }
interpolationSchemes { default linear; }
snGradSchemes { default corrected; }
wallDist { method exactDistance; }
""".replace("@USCHEME@", U_SCHEMES[uscheme]))
    write(case, "system/fvSolution", hdr("dictionary", "fvSolution", "system") + """
solvers
{
    p { solver GAMG; smoother GaussSeidel; tolerance 1e-10; relTol 0.05; }
    "(U|nuTilda|k|omega)" { solver smoothSolver; smoother symGaussSeidel; tolerance 1e-12; relTol 0.1; }
}
SIMPLE { nNonOrthogonalCorrectors 0; consistent yes; }
relaxationFactors { equations { U 0.9; nuTilda 0.8; k 0.8; omega 0.8; } }
""")
    for f in ("transportProperties", "physicalProperties"):
        write(case, f"constant/{f}", hdr("dictionary", f, "constant") +
              f"transportModel Newtonian;\nviscosityModel constant;\nnu {nu!r};\n")
    if model == "laminar":
        turb = "simulationType laminar;\n"
    else:
        name = "SpalartAllmaras" if model == "sa" else "kOmegaSST"
        coeffs = (f"{name}Coeffs {{ ft2 {'true' if ft2 else 'false'}; Ct3 1.2; "
                  "Ct4 0.5; Prt 0.9; }" if model == "sa"
                  else f"{name}Coeffs {{ Prt 0.9; }}")
        turb = (f"simulationType RAS;\nRAS {{ RASModel {name}; model {name}; "
                f"turbulence on; printCoeffs on; {coeffs} }}\n")
    for f in ("turbulenceProperties", "momentumTransport"):
        write(case, f"constant/{f}", hdr("dictionary", f, "constant") + turb)

    def field(name, cls, dim, internal, wall, far):
        write(case, f"0/{name}", hdr(cls, name, "0") + f"""
dimensions {dim};
internalField uniform {internal};
boundaryField
{{
    sphere {{ {wall} }}
    farfield {{ {far} }}
    wedge0 {{ type wedge; }}
    wedge1 {{ type wedge; }}
}}
""")
    field("U", "volVectorField", "[0 1 -1 0 0 0 0]", "(1 0 0)", "type noSlip;",
          "type freestreamVelocity; freestreamValue uniform (1 0 0); "
          "value uniform (1 0 0);")
    field("p", "volScalarField", "[0 2 -2 0 0 0 0]", "0", "type zeroGradient;",
          "type freestreamPressure; freestreamValue uniform 0; value uniform 0;")
    if model == "laminar":
        return
    nut_inf = 3 * nu * (27 / (27 + 7.1 ** 3)) if model == "sa" else 0.009 * nu
    field("nut", "volScalarField", "[0 2 -1 0 0 0 0]", f"{nut_inf!r}",
          "type fixedValue; value uniform 0;", "type calculated; value uniform 0;")
    if model == "sa":
        nt = 3 * nu
        field("nuTilda", "volScalarField", "[0 2 -1 0 0 0 0]", f"{nt!r}",
              "type fixedValue; value uniform 0;",
              f"type inletOutlet; inletValue uniform {nt!r}; value uniform {nt!r};")
    else:
        k = 2.25e-7
        w = k / (0.009 * nu)
        # float(): numpy 2 reprs np.float64 as "np.float64(...)".
        ww = "\n".join(f"{float(60 * nu / (BETA1 * d * d))!r}" for d in d1)
        field("k", "volScalarField", "[0 2 -2 0 0 0 0]", f"{k!r}",
              "type fixedValue; value uniform 0;",
              f"type inletOutlet; inletValue uniform {k!r}; value uniform {k!r};")
        field("omega", "volScalarField", "[0 0 -1 0 0 0 0]", f"{w!r}",
              f"type fixedValue; value nonuniform List<scalar> {len(d1)}\n(\n{ww}\n);",
              f"type inletOutlet; inletValue uniform {w!r}; value uniform {w!r};")


def foam(case, cmd, log, np_=1):
    full = f"source {FOAM} >/dev/null 2>&1; cd {case} && "
    if np_ > 1:
        full += (f"foamDictionary -entry numberOfSubdomains -set {np_} "
                 "system/decomposeParDict >/dev/null 2>&1; ")
        full += f"decomposePar -force > log.decompose 2>&1 && mpirun -np {np_} {cmd} -parallel"
    else:
        full += cmd
    with open(os.path.join(case, log), "w") as f:
        return subprocess.run(["bash", "-c", full], stdout=f,
                              stderr=subprocess.STDOUT).returncode


def run_case(case, np_=1):
    if np_ > 1:
        write(case, "system/decomposeParDict",
              hdr("dictionary", "decomposeParDict", "system") +
              f"numberOfSubdomains {np_};\nmethod scotch;\n")
    rc = foam(case, "checkMesh", "log.checkMesh")
    if rc:
        return rc
    import time
    t0 = time.time()
    rc = foam(case, "simpleFoam", "log.simpleFoam", np_)
    wall = time.time() - t0
    if np_ > 1:
        foam(case, "reconstructPar -latestTime", "log.reconstruct")
    with open(os.path.join(case, "time.txt"), "w") as f:
        f.write(f"{wall}\n")
    # wall shear and y+ on the final state, for theta_s, Cf, y+.
    foam(case, "simpleFoam -postProcess -latestTime -func wallShearStress",
         "log.wss")
    foam(case, "simpleFoam -postProcess -latestTime -func yPlus", "log.yplus")
    foam(case, "postProcess -func writeCellCentres -latestTime", "log.centres")
    return rc


def forces_history(case):
    """Per-iteration total, pressure and viscous x force on the wedge."""
    rows = []
    base = os.path.join(case, "postProcessing", "forces1")
    for t in sorted(os.listdir(base), key=float):
        p = os.path.join(base, t, "force.dat")
        for line in open(p):
            if line.startswith("#"):
                continue
            v = re.findall(r"[-+0-9.eE]+", line)
            if len(v) >= 10:
                rows.append([float(v[0]), float(v[1]), float(v[4]), float(v[7])])
    return np.array(rows)


def settle(x, tol):
    """The first sample after which |x - x_final| <= tol |x_final| forever."""
    f = x[-1]
    bad = np.nonzero(np.abs(x - f) > tol * abs(f))[0]
    return int(bad[-1]) + 1 if len(bad) else 0


def last_time(case):
    ts = [d for d in os.listdir(case) if re.fullmatch(r"[0-9.]+", d) and d != "0"]
    return max(ts, key=float)


def read_internal(case, t, name):
    txt = open(os.path.join(case, t, name)).read()
    m = re.search(r"internalField\s+nonuniform\s+List<(\w+)>\s*(\d+)\s*\(", txt)
    if not m:
        u = re.search(r"internalField\s+uniform\s+([^;]+);", txt)
        return u.group(1)
    n = int(m.group(2))
    body = txt[m.end():]
    vals = re.findall(r"\(([^()]*)\)" if m.group(1) == "vector" else r"[-+0-9.eE]+",
                      body)[:n]
    if m.group(1) == "vector":
        return np.array([[float(a) for a in v.split()] for v in vals])
    return np.array([float(v) for v in vals])


def read_patch(case, t, name, patch, vector):
    txt = open(os.path.join(case, t, name)).read()
    i = txt.find(patch)
    m = re.search(r"value\s+nonuniform\s+List<\w+>\s*(\d+)\s*\(", txt[i:])
    n = int(m.group(1))
    body = txt[i + m.end():]
    if vector:
        vals = re.findall(r"\(([^()]*)\)", body)[:n]
        return np.array([[float(a) for a in v.split()] for v in vals])
    return np.array([float(v) for v in re.findall(r"[-+0-9.eE]+", body)[:n]])


def observables_of(case):
    meta = json.load(open(os.path.join(case, "parity.json")))
    nT, aref = meta["nTheta"], meta["aref"]
    h = forces_history(case)
    q = 0.5
    cd, cdp, cdf = h[:, 1] / (q * aref), h[:, 2] / (q * aref), h[:, 3] / (q * aref)
    t = last_time(case)
    U = read_internal(case, t, "U")
    p = read_internal(case, t, "p")
    cellij = meta["cellij"]
    # Wall faces: the sphere patch's faces in i order; their first cells.
    wss = read_patch(case, t, "wallShearStress", "sphere", True)
    pw = np.array([p[k] for k, (i, j) in enumerate(cellij) if j == 1])
    out = dict(cd=float(cd[-1]), cdp=float(cdp[-1]), cdf=float(cdf[-1]),
               iters=int(h[-1, 0]),
               settle_1e3=settle(cd, 1e-3), settle_1e4=settle(cd, 1e-4))
    th = [(i - 0.5) * 180.0 / nT for i in range(1, nT + 1)]
    # Downstream tangent at polar angle theta in the plane z = 0.
    tau = []
    for a, w in zip(th, wss):
        r = math.radians(a)
        e = np.array([math.sin(r), math.cos(r), 0.0])
        tau.append(-float(np.dot(w, e)))  # OF wallShearStress = -tau_wall/rho
    out["theta"] = th
    out["cf"] = [2 * x for x in tau]
    out["cp"] = [2 * float(x) for x in pw]
    out["theta_s"] = sep_angle(th, out["cf"])
    out["cp_base"] = out["cp"][-1]
    # L_r on the downstream axis column (i = nT): first u >= 0 crossing.
    col = [(j, U[k][0]) for k, (i, j) in enumerate(cellij) if i == nT]
    out["lr"] = recirc(case, t, meta, col)
    if os.path.exists(os.path.join(case, "time.txt")):
        out["wall_s"] = float(open(os.path.join(case, "time.txt")).read())
    return out


def sep_angle(th, cf):
    for a in range(1, len(th)):
        if cf[a - 1] > 0 >= cf[a]:
            return th[a - 1] + (th[a] - th[a - 1]) * cf[a - 1] / (cf[a - 1] - cf[a])
    return 0.0


def recirc(case, t, meta, col):
    C = read_internal(case, t, "C") if os.path.exists(
        os.path.join(case, t, "C")) else None
    if C is None:
        return None
    nT = meta["nTheta"]
    xs = {j: C[k][0] for k, (i, j) in enumerate(meta["cellij"]) if i == nT}
    col.sort()
    x0, u0 = xs[col[0][0]], col[0][1]
    if u0 >= 0:
        return 0.0
    for j, u in col[1:]:
        if u >= 0:
            return x0 + (xs[j] - x0) * (-u0) / (u - u0) - 0.5
        x0, u0 = xs[j], u
    return 0.0


def observables_dns(prefix):
    """cd / cdf (stats), theta_s / cp / cf (surface), L_r (stats lr)."""
    import csv
    rows = list(csv.DictReader(open(prefix + "_stats.csv")))
    cd = np.array([float(r["cd"]) for r in rows])
    out = dict(cd=float(cd[-1]), iters=int(float(rows[-1].get("iter", 0))),
               settle_1e3=settle(cd, 1e-3) * 200, settle_1e4=settle(cd, 1e-4) * 200)
    if "cdf" in rows[-1]:
        out["cdf"] = float(rows[-1]["cdf"])
        out["cdp"] = out["cd"] - out["cdf"]
    if "lr" in rows[-1]:
        out["lr"] = float(rows[-1]["lr"])
    lines = [l for l in open(prefix + "_surface.csv") if not l.startswith("#")]
    srf = list(csv.DictReader(lines))
    out["theta"] = [float(r["theta_deg"]) for r in srf]
    out["cf"] = [float(r["cf"]) for r in srf]
    out["cp"] = [float(r["cp"]) for r in srf]
    out["yplus_max"] = max(float(r["yplus"]) for r in srf)
    out["theta_s"] = sep_angle(out["theta"], out["cf"])
    out["cp_base"] = out["cp"][-1]
    return out


if __name__ == "__main__":
    a = sys.argv[1:]
    if not a:
        sys.exit(__doc__)
    opt = lambda k, d: a[a.index(k) + 1] if k in a else d
    if a[0] == "case":
        make_case(a[1], a[2], a[3], float(a[4]), opt("--ft2", "true") == "true",
                  int(opt("--iters", "8000")), opt("--u-scheme", "linear"))
    elif a[0] == "run":
        sys.exit(run_case(a[1], int(opt("--np", "1"))))
    elif a[0] == "read":
        print(json.dumps(observables_of(a[1])))
    elif a[0] == "dns":
        print(json.dumps(observables_dns(a[1])))
