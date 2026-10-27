#!/usr/bin/env python3
"""othermal: wall heat transfer of the 2D cylinder, Re 40, laminar, steady,
isothermal wall, dns vs OpenFOAM on the IDENTICAL body-fitted O-grid
(plans/verify/exp_thermal.md; tools/parity/DESIGN.md "Thermal parity").

    othermal.py case <dns>_mesh.bin CASEDIR --twall R [--visc suth|const]
                     [--pr P] [--re 40] [--mach 0.2] [--tinf 300]
                     [--iters 4000] [--stock]
    othermal.py run  CASEDIR                 # checkMesh + rhoSimpleFoam
    othermal.py read CASEDIR                 # OpenFOAM observables, JSON
    othermal.py dns  <dns-out-prefix> CASEDIR  # dns observables, same defs
    othermal.py cmp  <dns-out-prefix> CASEDIR  # both + the differences

dns side (one run gives the state AND the mesh):
    dns flow --physics compressible --body cylinder --ogrid --n 128 --ny 64
        --ogrid-far 20 --re 40 --mach 0.2 --wall isothermal --twall R
        --visc sutherland --tinf 300 --prandtl 0.690335
        --steady 1e-5 --steady-scheme implicit
        --emit stats --emit surface --emit mesh --out PREFIX
Constant mu instead: drop --visc/--tinf/--prandtl (Pr 0.72) and pass
`--visc const --pr 0.72` here.

OpenFOAM side: rhoSimpleFoam, laminar, SI air (foamcase.py's gas), D = 1 m,
U = M c(Tinf), p 101325. Wall noSlip / p zeroGradient / T fixedValue Tw;
far row freestreamVelocity / freestreamPressure / T inletOutlet. U, e, Ekp
all `bounded Gauss linear` (central, as dns: the parity rule), laplacian
`Gauss linear corrected` (UNLIMITED: a limited correction is worth +9% of a
force on a non-orthogonal mesh, exp_mrfC). Transport `const` (mu, Pr) or
`sutherland` (As from mu(Tinf), Ts 110.4 K = dns' S).

TRAP 1 -- viscous work. v2606 rhoSimpleFoam / rhoPimpleFoam solve
    div(phi,e) + div(phi,Ekp) - laplacian(alphaEff,e) = fvOptions
with NO div(tau.u): viscous heating is absent from their energy equation.
At Ec = (gamma-1) M^2 Tinf/dT = 0.16 (Tw/Tinf 1.1, M 0.2) that alone reads
Nu 4.4-4.6% HIGH (0.55% at Tw/Tinf 2.0, Ec 0.016). The case therefore adds
div(tau.u) to e as a coded fvOption (VISC_WORK below) unless --stock. The
stock `viscousDissipation` fvOption (tau:grad u) is NOT the fix for the e-K
form: the missing term is the full div(tau.u).

TRAP 2 -- Prandtl number. OpenFOAM's sutherland transport has no Pr entry:
kappa = mu Cv (1.32 + 1.77 R/Cv) (modified Eucken) = mu cp / 0.690335 for
Cv = 2.5 R. A Sutherland dns run must use --prandtl 0.690335 (and this
script --pr 0.690335, the default for --visc suth); at dns' default 0.72
the comparison shows a FALSE 4.3% kappa gap. `const` takes Pr directly.

TRAP 3 -- wall heat flux sign. dns `--emit surface` column qw is the face
Fourier flux per area POSITIVE INTO THE BODY; OpenFOAM `wallHeatFlux`
(alphaEff snGrad(e), laminar alphaEff = kappa/Cv) is positive INTO THE
FLUID. Both are the same two-point gradient (T1 - Tw)/d with kappa(Tw).
For a hot wall: q = -qw_dns = +wallHeatFlux_foam > 0.

Metrics (identical on both sides):
  Nu_i    = q_i D / (kappa(Tinf) (Tw - Tinf)), each code's own kappa(Tinf)
            (identical by the Pr match).
  Nu      = area-weighted face mean of Nu_i (= total heat/(kappa dT pi D)).
  Nu_film = Nu kappa(Tinf) / kappa((Tw + Tinf)/2).
  Nu(theta) at 0/45/90/135/180 deg from the front stagnation point; faces
            matched by index (dnsmesh's wall patch order = dns i).
  cd      = Fx / (0.5 rho U^2 D); foam `forces` with pRef = pInf.
  T1      = first-cell-centre T / Tinf, same cells.

Measured 2026-10-27 (128 x 64, far 20 D; exp_thermal.md), dns vs foam with
div(tau.u): Tw/Tinf 1.1 Sutherland Nu -0.34%, cd +0.43%; Tw/Tinf 2.0
Sutherland Nu -0.14%, cd +0.14%; const mu Pr 0.72 Tw 1.1 Nu -0.35%.
Open: the M 0.5 adiabatic recovery reference (foam LTS unconverged) and
RANS Prt heat transfer -- not codified here.
"""
import csv
import json
import math
import os
import re
import struct
import subprocess
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import dnsmesh  # noqa: E402
import foamcase as fc  # noqa: E402

FOAM = "/opt/OpenFOAM/OpenFOAM-v2606/etc/bashrc"
TS = 110.4           # Sutherland S [K], dns' value
GAM = fc.GAMMA
PR_EUCKEN = 0.690335  # OpenFOAM sutherland's implied Pr (TRAP 2)

# TRAP 1: div(tau.u) on e, Gauss linear (the `div(tauU)` entry in fvSchemes).
VISC_WORK = r"""
viscWork
{
    type            scalarCodedSource;
    selectionMode   all;
    fields          (e);
    name            viscWork;
    codeInclude
    #{
        #include "turbulentFluidThermoModel.H"
        #include "fvcDiv.H"
    #};
    codeOptions
    #{
        -I$(LIB_SRC)/finiteVolume/lnInclude \
        -I$(LIB_SRC)/meshTools/lnInclude \
        -I$(LIB_SRC)/TurbulenceModels/turbulenceModels/lnInclude \
        -I$(LIB_SRC)/TurbulenceModels/compressible/lnInclude \
        -I$(LIB_SRC)/thermophysicalModels/basic/lnInclude \
        -I$(LIB_SRC)/transportModels/compressible/lnInclude
    #};
    codeLibs
    #{
        -lfiniteVolume -lmeshTools -lturbulenceModels -lcompressibleTurbulenceModels -lfluidThermophysicalModels
    #};
    codeAddSup
    #{
        const auto& U = mesh_.lookupObject<volVectorField>("U");
        const auto& turb = mesh_.lookupObject<compressible::turbulenceModel>(turbulenceModel::propertiesName);
        // div(tau . u), tau = -devRhoReff: the total-energy viscous work the
        // e-K energy equation of rhoSimpleFoam/rhoPimpleFoam omits.
        const volVectorField tauU("tauU", (-turb.devRhoReff()) & U);
        eqn += fvc::div(tauU);
    #};
    codeAddSupRho
    #{
        const auto& U = mesh_.lookupObject<volVectorField>("U");
        const auto& turb = mesh_.lookupObject<compressible::turbulenceModel>(turbulenceModel::propertiesName);
        const volVectorField tauU("tauU", (-turb.devRhoReff()) & U);
        eqn += fvc::div(tauU);
    #};
    codeCorrect #{ #};
    codeConstrain #{ #};
}
"""


def hdr(cls, obj):
    return ("FoamFile\n{\n    version 2.0;\n    format ascii;\n"
            f"    class {cls};\n    object {obj};\n}}\n\n")


def write(case, path, text):
    p = os.path.join(case, path)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    with open(p, "w") as f:
        f.write(text)


def gas(p):
    """SI freestream: D = 1 m, U = M c(Tinf), mu from Re."""
    R, T = fc.RGAS, p["tinf"]
    U = p["mach"] * math.sqrt(GAM * R * T)
    rho = fc.P_INF / (R * T)
    return dict(R=R, T=T, U=U, rho=rho, p=fc.P_INF, mu=rho * U / p["re"],
                cp=GAM / (GAM - 1) * R)


def mu_ratio(p, Tr):
    """mu(T)/mu(Tinf) at T = Tr Tinf (1 for const)."""
    if p["visc"] != "suth":
        return 1.0
    s = TS / p["tinf"]
    return Tr ** 1.5 * (1 + s) / (Tr + s)


def kappa_foam(p, g, Tr):
    """OpenFOAM's own conductivity at T = Tr Tinf (SI)."""
    mu = g["mu"] * mu_ratio(p, Tr)
    if p["visc"] == "suth":  # modified Eucken (TRAP 2)
        cv = g["cp"] - g["R"]
        return mu * cv * (1.32 + 1.77 * g["R"] / cv)
    return mu * g["cp"] / p["pr"]


def make_case(meshbin, case, p):
    m = dnsmesh.read_mesh(meshbin)
    if not dnsmesh.is_ogrid2d(m):
        sys.exit(f"{meshbin}: not a 2D O-grid mesh")
    dnsmesh.write_polymesh(case, m)
    g = gas(p)
    p.update(nx=m["nx"], ny=m["ny"])
    json.dump(dict(p=p, g=g), open(os.path.join(case, "thermal.json"), "w"))
    if p["visc"] == "suth":
        As = g["mu"] * (g["T"] + TS) / g["T"] ** 1.5
        tr, tname = f"transport {{ As {As!r}; Ts {TS}; }}", "sutherland"
    else:
        tr, tname = f"transport {{ mu {g['mu']!r}; Pr {p['pr']!r}; }}", "const"
    write(case, "constant/thermophysicalProperties",
          hdr("dictionary", "thermophysicalProperties") + f"""
thermoType
{{
    type heRhoThermo; mixture pureMixture; transport {tname}; thermo hConst;
    equationOfState perfectGas; specie specie; energy sensibleInternalEnergy;
}}
mixture
{{
    specie {{ molWeight {fc.MOLW!r}; }}
    thermodynamics {{ Cp {g['cp']!r}; Hf 0; }}
    {tr}
}}
""")
    for f in ("turbulenceProperties", "momentumTransport"):
        write(case, f"constant/{f}", hdr("dictionary", f) + "simulationType laminar;\n")
    it = p["iters"]
    write(case, "system/controlDict", hdr("dictionary", "controlDict") + f"""
application rhoSimpleFoam;
startFrom latestTime; startTime 0; stopAt endTime; endTime {it}; deltaT 1;
writeControl timeStep; writeInterval {it}; purgeWrite 1;
writeFormat ascii; writePrecision 12; timeFormat general; timePrecision 10;
runTimeModifiable false;
functions
{{
    forces1
    {{
        type forces; libs (forces); writeControl timeStep; writeInterval 50;
        patches (cylinder); rho rho; rhoInf {g['rho']!r}; pRef {g['p']!r};
        CofR (0 0 0); log false;
    }}
    whf
    {{
        type wallHeatFlux; libs (fieldFunctionObjects); patches (cylinder);
        writeControl writeTime;
    }}
}}
""")
    write(case, "system/fvSchemes", hdr("dictionary", "fvSchemes") + """
ddtSchemes { default steadyState; }
gradSchemes { default Gauss linear; }
divSchemes
{
    default none;
    div(phi,U) bounded Gauss linear;
    div(phi,e) bounded Gauss linear;
    div(phi,Ekp) bounded Gauss linear;
    div(tauU) Gauss linear;
    div(((rho*nuEff)*dev2(T(grad(U))))) Gauss linear;
}
laplacianSchemes { default Gauss linear corrected; }
interpolationSchemes { default linear; }
snGradSchemes { default corrected; }
""")
    # Measured: SIMPLE p 0.3 / U, e 0.7 converges to residualControl 1e-11
    # in 1600-1920 iterations (8-19 s serial) at M 0.2.
    write(case, "system/fvSolution", hdr("dictionary", "fvSolution") + """
solvers
{
    p { solver GAMG; smoother GaussSeidel; tolerance 1e-12; relTol 0.01; }
    "(U|e)" { solver smoothSolver; smoother symGaussSeidel; tolerance 1e-14; relTol 0.01; }
}
SIMPLE
{
    nNonOrthogonalCorrectors 0; consistent no; transonic no;
    pMinFactor 0.5; pMaxFactor 2.0;
    residualControl { p 1e-11; U 1e-11; e 1e-11; }
}
relaxationFactors { fields { p 0.3; rho 1; } equations { U 0.7; e 0.7; } }
""")
    if not p["stock"]:
        write(case, "system/fvOptions", hdr("dictionary", "fvOptions") + VISC_WORK)
    U, T, P = g["U"], g["T"], g["p"]
    Tw = p["twall"] * T

    def field(name, cls, dim, internal, wall, far):
        write(case, f"0/{name}", hdr(cls, name) + f"""
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
    field("U", "volVectorField", "[0 1 -1 0 0 0 0]", f"uniform ({U!r} 0 0)",
          "type noSlip;",
          f"type freestreamVelocity; freestreamValue uniform ({U!r} 0 0); "
          f"value uniform ({U!r} 0 0);")
    field("p", "volScalarField", "[1 -1 -2 0 0 0 0]", f"uniform {P!r}",
          "type zeroGradient;",
          f"type freestreamPressure; freestreamValue uniform {P!r}; value uniform {P!r};")
    field("T", "volScalarField", "[0 0 0 1 0 0 0]", f"uniform {T!r}",
          f"type fixedValue; value uniform {Tw!r};",
          f"type inletOutlet; inletValue uniform {T!r}; value uniform {T!r};")


def foam(case, cmd, log):
    full = f"source {FOAM} >/dev/null 2>&1; cd {case} && {cmd}"
    with open(os.path.join(case, log), "w") as f:
        return subprocess.run(["bash", "-c", full], stdout=f,
                              stderr=subprocess.STDOUT).returncode


def run(case):
    rc = foam(case, "checkMesh", "log.checkMesh")
    return rc or foam(case, "rhoSimpleFoam", "log.rhoSimpleFoam")


def latest(case):
    ts = [d for d in os.listdir(case) if re.fullmatch(r"[0-9.e+-]+", d) and d != "0"]
    return max(ts, key=float)


def patch_values(path, patch="cylinder"):
    txt = open(path).read()
    i = txt.index(patch, txt.index("boundaryField"))
    seg = txt[txt.index("value", i):]
    k = seg.index("(")
    n = int(seg[:k].split()[-1])
    return np.array(seg[k + 1:seg.index(")")].split(), float)[:n]


def internal(path, n):
    txt = open(path).read()
    k = txt.index("(", txt.index("internalField"))
    return np.array(txt[k + 1:txt.index(")", k)].split()[:n], float)


def wall_faces(m):
    """Wall face midpoints, lengths and angle from the front (deg), in dns i
    order (= dnsmesh's cylinder patch order)."""
    c = np.array(m["centre"])
    x = (m["x"][0, 1, :] - c[0]) / dnsmesh.D_BOX
    y = (m["y"][0, 1, :] - c[1]) / dnsmesh.D_BOX
    x1, y1 = np.roll(x, -1), np.roll(y, -1)
    xm, ym = 0.5 * (x + x1), 0.5 * (y + y1)
    return np.hypot(x1 - x, y1 - y), np.degrees(np.arctan2(ym, -xm))


def nu_stats(nu, area, theta, film):
    """nu: per-face Nu_i; film: kappa(Tinf)/kappa(Tfilm)."""
    mean = float(np.sum(nu * area) / np.sum(area))
    o = np.argsort(np.abs(theta))
    return dict(Nu=mean, Nu_film=mean * film,
                Nu_theta={f"{a:.0f}": float(np.interp(a, np.abs(theta)[o], nu[o]))
                          for a in (0, 45, 90, 135, 180)})


def meta(case):
    d = json.load(open(os.path.join(case, "thermal.json")))
    return d["p"], d["g"]


def observables_foam(case):
    p, g = meta(case)
    t = latest(case)
    m = dnsmesh.read_mesh(p["mesh"])
    rows = []
    fd = os.path.join(case, "postProcessing", "forces1")
    for d in sorted(os.listdir(fd), key=float):
        for line in open(os.path.join(fd, d, "force.dat")):
            if not line.startswith("#"):
                rows.append([float(x) for x in re.findall(r"[-+0-9.eE]+", line)[:2]])
    qd = 0.5 * g["rho"] * g["U"] ** 2
    area, theta = wall_faces(m)
    # TRAP 3: wallHeatFlux is positive INTO the fluid -- already the hot-wall sign.
    q = patch_values(os.path.join(case, t, "wallHeatFlux"))
    k0 = kappa_foam(p, g, 1.0)
    kf = kappa_foam(p, g, 0.5 * (p["twall"] + 1.0))
    out = dict(iters=int(float(t)), cd=rows[-1][1] / qd,
               cd_hist=[r[1] / qd for r in rows[-5:]],
               T1=(internal(os.path.join(case, t, "T"), p["nx"]) / g["T"]).tolist(),
               theta=theta.tolist())
    out.update(nu_stats(q / (k0 * (p["twall"] - 1.0) * g["T"]), area, theta, k0 / kf))
    return out


def read_final_T(prefix):
    b = open(prefix + "_final.bin", "rb").read()
    if b[:8] != b"DNSFINL1":
        sys.exit(f"{prefix}_final.bin: not a dns final state")
    _, nx, ny, nz, nf = struct.unpack_from("<5I", b, 8)
    off = 28
    names = [b[off + 16 * k: off + 16 * k + 16].rstrip(b"\0").decode() for k in range(nf)]
    off += 16 * nf
    n = nx * ny * nz
    k = names.index("T")
    return np.frombuffer(b, "<f8", n, off + 8 * n * k).reshape(nz, ny, nx)


def observables_dns(prefix, case):
    p, _ = meta(case)
    rows = list(csv.DictReader(l for l in open(prefix + "_surface.csv")
                               if not l.startswith("#")))
    st = list(csv.DictReader(open(prefix + "_stats.csv")))
    # dns units: rho = U = 1, D = D_BOX, Tinf = 1/(gamma M^2) (R = 1).
    D = dnsmesh.D_BOX
    Tinf = 1.0 / (GAM * p["mach"] ** 2)
    k0 = (D / p["re"]) * GAM / (GAM - 1) / p["pr"]
    area, theta = wall_faces(dnsmesh.read_mesh(p["mesh"]))
    # TRAP 3: qw is positive INTO the body; the hot-wall flux is -qw.
    qw = np.array([float(r["qw"]) for r in rows])
    nu = -qw * D / (k0 * (p["twall"] - 1.0) * Tinf)
    out = dict(cd=float(st[-1]["cd"]), iters=st[-1].get("iter"),
               T1=(read_final_T(prefix)[0, 1, :] / Tinf).tolist())
    out.update(nu_stats(nu, area, theta, 1.0 / mu_ratio(p, 0.5 * (p["twall"] + 1.0))))
    return out


def compare(prefix, case):
    d, f = observables_dns(prefix, case), observables_foam(case)
    p, _ = meta(case)
    T1d, T1f = np.array(d["T1"]), np.array(f["T1"])
    pct = lambda a, b: 100 * (a / b - 1)  # noqa: E731
    return dict(case=os.path.basename(os.path.normpath(case)),
                foam_energy="stock (NO viscous work)" if p["stock"] else "+ div(tau.u)",
                cd_dns=d["cd"], cd_foam=f["cd"], cd_pct=pct(d["cd"], f["cd"]),
                Nu_dns=d["Nu"], Nu_foam=f["Nu"], Nu_pct=pct(d["Nu"], f["Nu"]),
                Nu_film_dns=d["Nu_film"], Nu_film_foam=f["Nu_film"],
                Nu_theta_dns=d["Nu_theta"], Nu_theta_foam=f["Nu_theta"],
                Nu_theta_pct={a: pct(d["Nu_theta"][a], f["Nu_theta"][a])
                              for a in d["Nu_theta"]},
                T1_maxabsdiff=float(np.max(np.abs(T1d - T1f))),
                foam_cd_hist=f["cd_hist"], dns_iters=d["iters"],
                foam_iters=f["iters"])


if __name__ == "__main__":
    a = sys.argv[1:]
    if not a:
        sys.exit(__doc__)
    opt = lambda k, dflt: a[a.index(k) + 1] if k in a else dflt  # noqa: E731
    if a[0] == "case":
        visc = opt("--visc", "suth")
        if visc not in ("suth", "const"):
            sys.exit("--visc suth|const")
        if "--twall" not in a:
            sys.exit("--twall R is required (Tw/Tinf, != 1)")
        p = dict(mesh=os.path.abspath(a[1]), re=float(opt("--re", "40")),
                 mach=float(opt("--mach", "0.2")), tinf=float(opt("--tinf", "300")),
                 pr=float(opt("--pr", str(PR_EUCKEN) if visc == "suth" else "0.72")),
                 visc=visc, twall=float(opt("--twall", "1.1")),
                 iters=int(opt("--iters", "4000")), stock="--stock" in a)
        if visc == "suth" and p["pr"] != PR_EUCKEN:
            sys.exit(f"--visc suth: OpenFOAM's Pr is {PR_EUCKEN} (TRAP 2)")
        make_case(a[1], a[2], p)
    elif a[0] == "run":
        sys.exit(run(a[1]))
    elif a[0] == "read":
        print(json.dumps(observables_foam(a[1])))
    elif a[0] == "dns":
        print(json.dumps(observables_dns(a[1], a[2])))
    elif a[0] == "cmp":
        print(json.dumps(compare(a[1], a[2]), indent=1))
    else:
        sys.exit(__doc__)
