#!/usr/bin/env python3
"""ocyl1e4: URANS shedding of the 2D cylinder, Re 1e4, M 0.2, dns vs
OpenFOAM on the IDENTICAL body-fitted O-grid (masterplan_verify W12;
docs/validation.md "O-grid cylinder URANS").

    ocyl1e4.py case <dns>_mesh.bin CASEDIR MODEL [--re 1e4] [--mach 0.2]
                    [--trip 0.05] [--end 40] [--co 0.5] [--write 5] [--np 6]
    ocyl1e4.py run  CASEDIR [--timeout S]   # rhoPimpleFoam, from latestTime
    ocyl1e4.py read CASEDIR                 # OpenFOAM window statistics, JSON
    ocyl1e4.py dns  <dns-out-prefix>        # the same statistics from dns

dns side: `dns flow --physics compressible --body cylinder --ogrid --n 256
--ny 96 --ogrid-beta 4 --ogrid-far 20 --re 1e4 --mach 0.2 --ogrid-trip 0.05
--rans sst-2003m|sa-neg --compensated --ft 1` (40 D/U), and once with
`--emit mesh` for the mesh.

OpenFOAM side (tier A, same method): rhoPimpleFoam, SI air (foamcase.py's
gas: gamma 1.4, constant mu from Re, Pr 0.72), D = 1 m, U = M c_inf.
MODEL sst (kOmegaSST) or sa (SpalartAllmaras { ft2 true }); Prt 0.9 for
both. Units of decision 4 (TMR): k = 2.25e-7 U^2, mu_t/mu = 0.009, nuTilda
= 3 nu; wall omega = 60 nu/(beta1 d1^2) per wall face from dns's own d1
(nonuniform fixedValue, as sph1e4.py), k = nuTilda = nut = 0 on the wall,
adiabatic wall. Far row: freestreamVelocity / freestreamPressure /
inletOutlet. Schemes: backward in time, central `Gauss linear` U (the
parity rule), e linearUpwind (tier A), turbulence first-order upwind (as
dns). PIMPLE nOuter 2 / nCorr 2 at maxCo --co. MEASURED (restarts of the SST
case at 30 D/U, 1.5 D/U each): nOuter 1 rings a 10-step cd sawtooth (HF rms
0.34) that biases the mean cd +5% (1.192 vs 1.133); nOuter 2 and 3 agree
(HF 0.063, cd 1.133); maxCo 0.25 alone (0.22) or CrankNicolson 0.9 (0.28)
do not cure it, Euler does (0.011) at first order. --np N > 1 runs mpirun
with wall distance meshWave (correctWalls): exactDistance fails in parallel
(CLAUDE.md); serial exactDistance and parallel meshWave gave the same cd /
cl to 4 digits on that restart.

Start: dns ramps from rest (--ramp 2, a Galilean boost) with the trip
vortex; OpenFOAM starts IMPULSIVELY from the freestream with the SAME
vortex added (u += 2 dy/w e, v -= 2 dx/w e, e = A U exp(-r^2/w^2), w = D/4,
at (D, D/4); dns_main.cpp), evaluated at the same cell centres. The start
differs; only the saturated cycle is compared.

Statistics (both codes, `stats()`): times in D/U, St = f D/U from cl's
upward zero crossings. The window starts at the first period after which
the per-period cl swing changes < 2% between consecutive periods, for good;
if fewer than 4 periods remain, the result is "not saturated" and nothing
is averaged. Mean cd, cl_rms and St are batch means over whole periods;
SE = std(period values)/sqrt(n).
"""
import json
import math
import os
import re
import subprocess
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import dnsmesh  # noqa: E402
import foamcase as fc  # noqa: E402

FOAM = "/opt/OpenFOAM/OpenFOAM-v2606/etc/bashrc"
BETA1 = 0.075
PR = 0.72
PRT = 0.9


def hdr(cls, obj, loc):
    return ("FoamFile\n{\n    version 2.0;\n    format ascii;\n"
            f"    class {cls};\n    location \"{loc}\";\n    object {obj};\n}}\n\n")


def write(case, path, text):
    p = os.path.join(case, path)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    with open(p, "w") as f:
        f.write(text)


def cell_centres(m):
    """dns cell centres (the 4-node average of the 2D cell, which is dns's
    8-node average) in D units, in dnsmesh cellij order."""
    nx, ny = m["nx"], m["ny"]
    x, y = m["x"][0], m["y"][0]
    out = []
    for j in range(1, ny - 1):
        for i in range(nx):
            i1 = (i + 1) % nx
            out.append(((x[j, i] + x[j, i1] + x[j + 1, i] + x[j + 1, i1]) / 4,
                        (y[j, i] + y[j, i1] + y[j + 1, i] + y[j + 1, i1]) / 4))
    return np.array(out) / dnsmesh.D_BOX


def make_case(meshbin, case, model, re_=1e4, mach=0.2, trip=0.05, end=40.0,
              co=0.5, write_every=5.0, np_=6):
    m = dnsmesh.read_mesh(meshbin)
    if not dnsmesh.is_ogrid2d(m):
        sys.exit(f"{meshbin}: not a 2D O-grid mesh")
    cellij, d1 = dnsmesh.write_polymesh(case, m)
    st = fc.tier_a_state(mach, re_)
    U, rho, p, T, mu = st["U"], st["rho"], st["p"], st["T"], st["mu"]
    nu = mu / rho
    tu = 1.0 / U  # D/U in s (D = 1 m)
    json.dump({"model": model, "re": re_, "mach": mach, "trip": trip,
               "U": U, "rho": rho, "p": p, "nu": nu, "tUnit": tu,
               "aref": 1.0, "nx": m["nx"], "ny": m["ny"], "np": np_},
              open(os.path.join(case, "parity.json"), "w"))
    if np_ > 1:
        write(case, "system/decomposeParDict",
              hdr("dictionary", "decomposeParDict", "system") +
              f"numberOfSubdomains {np_};\nmethod scotch;\n")
    wdist = ("meshWave; correctWalls true" if np_ > 1 else "exactDistance")
    write(case, "system/controlDict", hdr("dictionary", "controlDict", "system") + f"""
application rhoPimpleFoam;
startFrom latestTime;
startTime 0;
stopAt endTime;
endTime {end * tu!r};
deltaT {1e-4 * tu!r};
writeControl adjustableRunTime;
writeInterval {write_every * tu!r};
purgeWrite 2;
writeFormat binary;
writePrecision 12;
timeFormat general;
timePrecision 12;
adjustTimeStep yes;
maxCo {co};
runTimeModifiable false;
functions
{{
    forces1
    {{
        type forces; libs (forces); writeControl timeStep; writeInterval 1;
        patches (cylinder); rho rho; rhoInf {rho!r}; pRef {p!r};
        CofR (0 0 0); log false;
    }}
}}
""")
    write(case, "system/fvSchemes", hdr("dictionary", "fvSchemes", "system") + """
ddtSchemes { default backward; }
gradSchemes { default Gauss linear; }
divSchemes
{
    default none;
    div(phi,U) Gauss linear;
    div(phi,e) Gauss linearUpwind grad(e);
    div(phi,K) Gauss linear;
    div(phiv,p) Gauss linear;
    div(phi,nuTilda) bounded Gauss upwind;
    div(phi,k) bounded Gauss upwind;
    div(phi,omega) bounded Gauss upwind;
    div(((rho*nuEff)*dev2(T(grad(U))))) Gauss linear;
}
laplacianSchemes { default Gauss linear corrected; }
interpolationSchemes { default linear; }
snGradSchemes { default corrected; }
wallDist { method @WDIST@; }
""".replace("@WDIST@", wdist))
    write(case, "system/fvSolution", hdr("dictionary", "fvSolution", "system") + """
solvers
{
    "rho.*" { solver diagonal; }
    p { solver GAMG; smoother GaussSeidel; tolerance 1e-9; relTol 0.01; }
    pFinal { $p; relTol 0; }
    "(U|e|k|omega|nuTilda)" { solver smoothSolver; smoother symGaussSeidel; tolerance 1e-10; relTol 0.1; }
    "(U|e|k|omega|nuTilda)Final" { $U; relTol 0; }
}
PIMPLE
{
    momentumPredictor yes;
    transonic no;
    nOuterCorrectors 2;
    nCorrectors 2;
    nNonOrthogonalCorrectors 0;
}
relaxationFactors { equations { ".*" 1; } }
""")
    fc.physics_a(case, st, PR)
    name = "SpalartAllmaras" if model == "sa" else "kOmegaSST"
    coeffs = (f"{name}Coeffs {{ ft2 true; Ct3 1.2; Ct4 0.5; Prt {PRT}; }}"
              if model == "sa" else f"{name}Coeffs {{ Prt {PRT}; }}")
    turb = (f"simulationType RAS;\nRAS {{ RASModel {name}; model {name}; "
            f"turbulence on; printCoeffs on; {coeffs} }}\n")
    for f in ("turbulenceProperties", "momentumTransport"):
        write(case, f"constant/{f}", hdr("dictionary", f, "constant") + turb)

    def field(name, cls, dim, internal, wall, far):
        write(case, f"0/{name}", hdr(cls, name, "0") + f"""
dimensions {dim};
internalField {internal};
boundaryField
{{
    cylinder {{ {wall} }}
    farfield {{ {far} }}
    front {{ type empty; }}
    back {{ type empty; }}
}}
""")

    # The trip: dns's solenoidal vortex at the same cell centres.
    C = cell_centres(m)
    w = 0.25
    dx, dy = C[:, 0] - 1.0, C[:, 1] - 0.25
    e = trip * U * np.exp(-(dx * dx + dy * dy) / (w * w))
    u = U + 2 * dy / w * e
    v = -2 * dx / w * e
    ui = ("nonuniform List<vector>\n" + f"{len(u)}\n(\n" +
          "\n".join(f"({float(a)!r} {float(b)!r} 0)" for a, b in zip(u, v)) + "\n)")
    field("U", "volVectorField", "[0 1 -1 0 0 0 0]", ui, "type noSlip;",
          f"type freestreamVelocity; freestreamValue uniform ({U!r} 0 0); "
          f"value uniform ({U!r} 0 0);")
    field("p", "volScalarField", "[1 -1 -2 0 0 0 0]", f"uniform {p!r}",
          "type zeroGradient;",
          f"type freestreamPressure; freestreamValue uniform {p!r}; value uniform {p!r};")
    field("T", "volScalarField", "[0 0 0 1 0 0 0]", f"uniform {T!r}",
          "type zeroGradient;",
          f"type inletOutlet; inletValue uniform {T!r}; value uniform {T!r};")
    field("alphat", "volScalarField", "[1 -1 -1 0 0 0 0]", "uniform 0",
          "type fixedValue; value uniform 0;", "type calculated; value uniform 0;")
    if model == "sa":
        nt = 3 * nu
        nut_inf = nt * 27 / (27 + 7.1 ** 3)
        field("nuTilda", "volScalarField", "[0 2 -1 0 0 0 0]", f"uniform {nt!r}",
              "type fixedValue; value uniform 0;",
              f"type inletOutlet; inletValue uniform {nt!r}; value uniform {nt!r};")
    else:
        k = 2.25e-7 * U * U
        om = k / (0.009 * nu)
        nut_inf = 0.009 * nu
        # 60 must equal dns's sst::omegaWallFactor * beta1 (src/dns/rans.hpp).
        ww = "\n".join(f"{float(60 * nu / (BETA1 * d * d))!r}" for d in d1)
        field("k", "volScalarField", "[0 2 -2 0 0 0 0]", f"uniform {k!r}",
              "type fixedValue; value uniform 0;",
              f"type inletOutlet; inletValue uniform {k!r}; value uniform {k!r};")
        field("omega", "volScalarField", "[0 0 -1 0 0 0 0]", f"uniform {om!r}",
              f"type fixedValue; value nonuniform List<scalar> {len(d1)}\n(\n{ww}\n);",
              f"type inletOutlet; inletValue uniform {om!r}; value uniform {om!r};")
    field("nut", "volScalarField", "[0 2 -1 0 0 0 0]", f"uniform {nut_inf!r}",
          "type fixedValue; value uniform 0;", "type calculated; value uniform 0;")


def foam(case, cmd, log, timeout=None, append=False):
    full = f"source {FOAM} >/dev/null 2>&1; cd {case} && "
    full += (f"timeout -s INT {int(timeout)} {cmd}" if timeout else cmd)
    with open(os.path.join(case, log), "a" if append else "w") as f:
        return subprocess.run(["bash", "-c", full], stdout=f,
                              stderr=subprocess.STDOUT).returncode


def run_case(case, timeout=None):
    """One chunk: rhoPimpleFoam from latestTime until endTime or timeout
    (SIGINT; the last periodic write is the restart point, forces.dat
    carries every step). Parallel cases decompose once and restart from the
    processor directories."""
    np_ = json.load(open(os.path.join(case, "parity.json"))).get("np", 1)
    if not os.path.exists(os.path.join(case, "log.checkMesh")):
        rc = foam(case, "checkMesh", "log.checkMesh")
        if rc:
            return rc
    cmd = "rhoPimpleFoam"
    if np_ > 1:
        if not os.path.isdir(os.path.join(case, "processor0")):
            rc = foam(case, "decomposePar -force", "log.decompose")
            if rc:
                return rc
        cmd = f"mpirun -np {np_} rhoPimpleFoam -parallel"
    t0 = time.time()
    rc = foam(case, cmd, "log.rhoPimpleFoam", timeout, append=True)
    with open(os.path.join(case, "time.txt"), "a") as f:
        f.write(f"{time.time() - t0} rc={rc}\n")
    return rc


def forces_history(case):
    """(t, Fx, Fy) per step over every restart directory, later restarts
    overriding the overlap."""
    base = os.path.join(case, "postProcessing", "forces1")
    rows = []
    for t in sorted(os.listdir(base), key=float):
        p = os.path.join(base, t, "force.dat")
        if not os.path.exists(p):
            continue
        part = []
        for line in open(p):
            if line.startswith("#"):
                continue
            v = re.findall(r"[-+0-9.eE]+", line)
            if len(v) >= 10:
                part.append([float(v[0]), float(v[1]), float(v[2])])
        if part:
            rows = [r for r in rows if r[0] < part[0][0]] + part
    return np.array(rows)


def upcrossings(t, y):
    """Upward crossings of y through its mean with a 0.2 rms dead band
    (tests/validation_util.hpp strouhal)."""
    mean = float(np.mean(y))
    band = 0.2 * float(np.sqrt(np.mean((y - mean) ** 2)))
    up, armed = [], False
    for i in range(1, len(t)):
        a, b = y[i - 1] - mean, y[i] - mean
        if a < -band:
            armed = True
        if armed and a <= band < b:
            up.append(t[i - 1] + (t[i] - t[i - 1]) * (band - a) / (b - a))
            armed = False
    return np.array(up)


def _wmean(t, y, a, b):
    m = (t >= a) & (t <= b)
    tt, yy = t[m], y[m]
    return float(np.trapezoid(yy, tt) / (tt[-1] - tt[0]))


def stats(t, cd, cl, min_periods=4, tol=0.02):
    """Saturated-window statistics; t in D/U."""
    # Crossings over the second half's mean level (the growth phase biases
    # a whole-run mean of cl only slightly; cl is centred on 0 anyway).
    up = upcrossings(t, cl)
    per = []
    for a, b in zip(up[:-1], up[1:]):
        m = (t >= a) & (t <= b)
        per.append(dict(t0=float(a), t1=float(b), T=float(b - a),
                        swing=float(0.5 * (cl[m].max() - cl[m].min())),
                        cd=_wmean(t, cd, a, b),
                        clrms=math.sqrt(_wmean(t, cl * cl, a, b))))
    out = dict(periods_total=len(per), t_end=float(t[-1]))
    # First period k after which every consecutive swing change is < tol.
    k0 = None
    for k in range(len(per) - 1):
        if all(abs(per[q + 1]["swing"] / per[q]["swing"] - 1) < tol
               for q in range(k, len(per) - 1)):
            k0 = k
            break
    if k0 is None or len(per) - k0 < min_periods:
        out["saturated"] = False
        out["swing_last"] = [p["swing"] for p in per[-6:]]
        return out
    w = per[k0:]
    n = len(w)

    def mse(key):
        v = np.array([p[key] for p in w])
        return float(v.mean()), float(v.std(ddof=1) / math.sqrt(n))

    T, Tse = mse("T")
    out.update(saturated=True, window=[w[0]["t0"], w[-1]["t1"]], periods=n,
               St=1.0 / T, St_se=Tse / T ** 2, cd=mse("cd")[0], cd_se=mse("cd")[1],
               clrms=mse("clrms")[0], clrms_se=mse("clrms")[1],
               swing=mse("swing")[0], swing_se=mse("swing")[1])
    return out


def observables_of(case):
    meta = json.load(open(os.path.join(case, "parity.json")))
    h = forces_history(case)
    q = 0.5 * meta["rho"] * meta["U"] ** 2 * meta["aref"]
    t = h[:, 0] / meta["tUnit"]
    out = stats(t, h[:, 1] / q, h[:, 2] / q)
    out["steps"] = int(len(h))
    if os.path.exists(os.path.join(case, "time.txt")):
        out["wall_s"] = sum(float(l.split()[0]) for l in open(os.path.join(case, "time.txt")))
    return out


def observables_dns(prefix):
    import csv
    rows = list(csv.DictReader(open(prefix + "_stats.csv")))
    D = math.pi / 4  # box units: U = 1, so t / D is D/U
    t = np.array([float(r["t"]) for r in rows]) / D
    cd = np.array([float(r["cd"]) for r in rows])
    cl = np.array([float(r["cl"]) for r in rows])
    out = stats(t, cd, cl)
    js = prefix + ".json"
    if os.path.exists(js):
        out["msPerStep"] = json.load(open(js)).get("msPerStep")
    return out


if __name__ == "__main__":
    a = sys.argv[1:]
    if not a:
        sys.exit(__doc__)
    opt = lambda k, d: a[a.index(k) + 1] if k in a else d
    if a[0] == "case":
        make_case(a[1], a[2], a[3], float(opt("--re", "1e4")),
                  float(opt("--mach", "0.2")), float(opt("--trip", "0.05")),
                  float(opt("--end", "40")), float(opt("--co", "0.5")),
                  float(opt("--write", "5")), int(opt("--np", "6")))
    elif a[0] == "run":
        t = opt("--timeout", None)
        sys.exit(run_case(a[1], float(t) if t else None))
    elif a[0] == "read":
        print(json.dumps(observables_of(a[1])))
    elif a[0] == "dns":
        print(json.dumps(observables_dns(a[1])))
