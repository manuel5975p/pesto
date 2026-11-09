#!/usr/bin/env python3
"""The 0D adiabatic reactor of model C0 in OpenFOAM v2606 chemFoam against the
closed quadrature (masterplan_combnozzle P0, docs/chem.md "P0: the 0D reactor").

  chemfoam_reactor.py setup [--ode S]    write runs/flames/foam/P0/<S>/<case>/
  chemfoam_reactor.py run [--ode S]      every case through simjob (one CPU job)
  chemfoam_reactor.py compare [--ode S]  T(t) at every chemFoam step vs the quadrature
  chemfoam_reactor.py rk3ref             the 40-digit references of tests/fvref_reactor.hpp

Model (box units, Tu = 1): dY/dt = -Lambda Y e^(-theta/T), T(Y) = T0 + dT (1 - Y),
T0 = 2.5, theta = 50 (sigma 5, beta 8), dT = gamma (sigma - 1) at constant volume
(q/cv), sigma - 1 at constant pressure (q/cp). SI by foamflame.model.si_map at
M 0.01 (Tu 300 K): A = Lambda/tau, Ta = theta Tu.

chemFoam (v2606 source):
- readInitialConditions.H casts the thermo to reactingMixture<gasHThermoPhysics>
  (sutherland + janaf): constant cp is janaf with only a0 = cp/R = 3.5 non-zero in
  both ranges, the enthalpy offset a5 carries the heat of reaction:
  ha_F - ha_P = R (a5_F - a5_P) = cp (sigma - 1) Tu = q, independent of R and Tstd.
- StandardChemistryModel::derivatives integrates T at CONSTANT PRESSURE inside each
  chemistry step (dT = -sum ha dc / (rho cp)); `constantProperty volume` resets h to
  u0 + p_old/rho + int Qdot after it. So chemFoam's constant-volume reactor is an
  O(dt) splitting of the cv ODE (the intra-step T rises at q/cp, p/rho lags a step);
  its constant-pressure reactor is the cp ODE to the ODE tolerance.
Cases: cv (dt t_e/2000 and t_e/8000: first order shown), cp (dt t_p/1000), and
the controls cv_ctl, cp_ctl: Hf_P = 1% of Hf_F (q -1%).
"""
import argparse
import json
import math
import os
import shlex
import subprocess
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from foamflame import writer  # noqa: E402
from foamflame.model import GAMMA, si_map  # noqa: E402

REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
SIMJOB = os.path.join(REPO, "build", "simjob")
FOAM = os.environ.get("DNS_FOAM_V2606", "/opt/OpenFOAM/OpenFOAM-v2606")
CHEMFOAM = os.path.join(FOAM, "platforms", "linux64GccDPInt32Opt", "bin", "chemFoam")
ROOT = os.path.join(REPO, "runs", "flames", "foam", "P0")

MACH, SIGMA, BETA, PR = 0.01, 5.0, 8.0, 0.72
LAMBDA = 3656401.0984806987          # premixedEigen(5, 8), runs/flames/foam/oracle_s5_b8.npz
THETA = BETA * SIGMA * SIGMA / (SIGMA - 1.0)
T0 = 2.5                             # T0 / Tu
DT_CV, DT_CP = GAMMA * (SIGMA - 1.0), SIGMA - 1.0
TE_V, TE_P = 3.416547072079102420614175, 4.843940899982749326036909  # t(1/2), 40 digits
BAND, CTL_MISS = 1e-5, 1e-3          # G0.5 band; the control must miss by more

# name: (constantProperty, dT of its oracle, t(1/2), steps per t(1/2), Hf_P / Hf_F)
CASES = {
    "cv":      ("volume",   DT_CV, TE_V, 2000, 0.0),
    "cv_fine": ("volume",   DT_CV, TE_V, 8000, 0.0),
    "cv_ctl":  ("volume",   DT_CV, TE_V, 2000, 0.01),
    "cp":      ("pressure", DT_CP, TE_P, 1000, 0.0),
    "cp_ctl":  ("pressure", DT_CP, TE_P, 1000, 0.01),
}


# --- the closed quadrature -----------------------------------------------------

class Quadrature:
    """t(Y) = int_{ln Y}^0 ds e^(theta/T(e^s)) / Lambda on panels of h in s (Gauss-
    Legendre 12 each, cumulative), Y(t) by Newton inside the panel. Box units."""

    def __init__(self, dT, h=0.02, smin=-60.0):
        self.dT = dT
        self.x, self.w = np.polynomial.legendre.leggauss(12)
        self.edges = -h * np.arange(int(round(-smin / h)) + 1)
        self.tcum = np.concatenate(([0.0], np.cumsum(self._piece(self.edges[1:], self.edges[:-1]))))

    def g(self, s):
        return np.exp(THETA / (T0 + self.dT * (1.0 - np.exp(s)))) / LAMBDA

    def _piece(self, a, b):
        """int_a^b g, a <= b (arrays), one Gauss-Legendre rule."""
        m, r = 0.5 * (a + b), 0.5 * (b - a)
        return r * sum(wk * self.g(m + r * xk) for xk, wk in zip(self.x, self.w))

    def time_to(self, Y):
        s = np.log(np.asarray(Y, float))
        j = np.clip(np.searchsorted(-self.edges, -s, side="right") - 1, 0, len(self.edges) - 2)
        return self.tcum[j] + self._piece(s, self.edges[j])

    def T(self, t):
        """T(t)/Tu at times t (box units)."""
        t = np.atleast_1d(np.asarray(t, float))
        j = np.clip(np.searchsorted(self.tcum, t, side="right") - 1, 0, len(self.edges) - 2)
        past = t >= self.tcum[-1]
        sj = self.edges[j]
        s = sj - (t - self.tcum[j]) / self.g(sj)
        s = np.clip(s, self.edges[j + 1], sj)
        for _ in range(12):
            F = self.tcum[j] + self._piece(s, sj) - t
            s = np.clip(s + F / self.g(s), self.edges[j + 1] - 1.0, sj)
        Y = np.where(past, 0.0, np.exp(s))
        return T0 + self.dT * (1.0 - Y)


# --- the case -----------------------------------------------------------------

def thermo_janaf(case, si, hf_ratio):
    """Two species F, P of molWeight W: janaf, a0 = cp/R = 3.5, a5 the offset."""
    a0 = GAMMA / (GAMMA - 1.0)
    tstd = 298.15
    dq = a0 * (SIGMA - 1.0) * si["Tu"]            # (a5_F - a5_P) = q / R
    a5P = -a0 * tstd + hf_ratio * dq              # Hf_P = R (a0 Tstd + a5_P) = hf_ratio q
    a5F = -a0 * tstd + dq
    writer.put(case, "constant/thermophysicalProperties", """thermoType
{
    type            hePsiThermo;
    mixture         reactingMixture;
    transport       sutherland;
    thermo          janaf;
    energy          sensibleEnthalpy;
    equationOfState perfectGas;
    specie          specie;
}

inertSpecie         P;
chemistryReader     foamChemistryReader;
foamChemistryFile   "<constant>/reactions";
foamChemistryThermoFile "<constant>/thermo.foam";
""")
    body = ""
    for name, a5 in (("F", a5F), ("P", a5P)):
        cf = f"({writer.g(a0)} 0 0 0 0 {writer.g(a5)} 0)"
        body += f"""{name}
{{
    specie         {{ molWeight {writer.g(si["W"])}; }}
    elements       {{ N 2; }}
    thermodynamics {{ Tlow 100; Thigh 10000; Tcommon 1000; highCpCoeffs {cf}; lowCpCoeffs {cf}; }}
    transport      {{ As 1.67212e-06; Ts 170.672; }}
}}

"""
    writer.put(case, "constant/thermo.foam", body)


def write_case(case, prop, te, n, hf_ratio, si, ode):
    writer.put(case, "constant/initialConditions", f"""constantProperty {prop};
fractionBasis   mass;
fractions       {{ F 1; P 0; }}
p               {writer.g(si["p0"])};
T               {writer.g(T0 * si["Tu"])};
""")
    thermo_janaf(case, si, hf_ratio)
    writer.reactions_file(case, ["F", "P"], [("F = P", si["A"], si["Ta"])])
    writer.put(case, "constant/chemistryProperties", """chemistryType
{
    solver          ode;
    method          standard;
}

chemistry           on;
initialChemicalTimeStep 1e-12;

odeCoeffs
{
    solver          %s;
    absTol          1e-14;
    relTol          1e-10;
}
""" % ode)
    dt = te * si["tau"] / n
    writer.put(case, "system/controlDict", f"""application     chemFoam;
startFrom       startTime;
startTime       0;
stopAt          endTime;
endTime         {writer.g(2 * n * dt)};
deltaT          {writer.g(dt)};
maxDeltaT       1;
adjustTimeStep  no;
writeControl    timeStep;
writeInterval   100000000;
purgeWrite      0;
writeFormat     ascii;
writePrecision  17;
writeCompression off;
timeFormat      general;
timePrecision   17;
runTimeModifiable false;
""")
    writer.put(case, "system/fvSchemes", "ddtSchemes { default Euler; }\ngradSchemes { }\n"
               "divSchemes { }\nlaplacianSchemes { }\n")
    writer.put(case, "system/fvSolution", "solvers\n{\n    Yi { solver diagonal; }\n}\n")


def context():
    si = si_map(MACH, SIGMA, BETA, PR, LAMBDA)
    return si


def cmd_setup(ode):
    si = context()
    root = os.path.join(ROOT, ode)
    os.makedirs(root, exist_ok=True)
    with open(os.path.join(root, "si.json"), "w") as f:
        json.dump(si, f, indent=1)
    for name, (prop, _, te, n, hf) in CASES.items():
        write_case(os.path.join(root, name), prop, te, n, hf, si, ode)
        print(f"case {name}: {prop}, dt t(1/2)/{n}, Hf_P/Hf_F {hf:g}")
    print(f"SI: Tu {si['Tu']} K, tau {si['tau']:.10g} s, A {si['A']:.15g} /s, Ta {si['Ta']:.10g} K")


def cmd_run(ode):
    """Each case's log in CASE/log.chemFoam; a failing case does not stop the others."""
    root = os.path.join(ROOT, ode)
    script = os.path.join(root, "run.sh")
    with open(script, "w") as f:
        f.write("#!/bin/bash\nrc=0\nfor c in " + " ".join(CASES) + "; do\n"
                f"    {CHEMFOAM} -case {root}/$c > {root}/$c/log.chemFoam 2>&1 || {{ echo \"$c failed\"; rc=1; }}\n"
                "done\nexit $rc\n")
    os.chmod(script, 0o755)
    job = {"label": f"P0 chemFoam reactor {ode}", "resource": "cpu", "max_runtime_s": 600,
           "max_mem_mb": 2048, "cwd": root, "steps": [{"exe": "/bin/bash", "args": [script]}]}
    path = os.path.join(root, "job.json")
    with open(path, "w") as f:
        json.dump(job, f, indent=1)
    cmd = (f"source {FOAM}/etc/bashrc >/dev/null 2>&1; "
           f"{shlex.quote(SIMJOB)} submit --spec {shlex.quote(path)} --inherit-env --wait")
    p = subprocess.run(["bash", "-c", cmd], capture_output=True, text=True)
    print(p.stdout + p.stderr, end="")
    return p.returncode


def read_out(case):
    a = np.loadtxt(os.path.join(case, "chemFoam.out"), comments="#")
    return a[:, 0], a[:, 1]


def cmd_compare(ode):
    si = context()
    root = os.path.join(ROOT, ode)
    quad = {DT_CV: Quadrature(DT_CV), DT_CP: Quadrature(DT_CP)}
    for dT, te in ((DT_CV, TE_V), (DT_CP, TE_P)):
        e = abs(float(quad[dT].time_to(0.5)) / te - 1.0)
        print(f"quadrature t(1/2) dT {dT:g}: {float(quad[dT].time_to(0.5)):.16f} vs 40 digits {te:.16f} ({e:.1e})")
        assert e <= 1e-12, "quadrature self-check"
    res = {}
    for name, (prop, dT, te, n, hf) in CASES.items():
        if not os.path.exists(os.path.join(root, name, "chemFoam.out")) or \
                "End" not in open(os.path.join(root, name, "log.chemFoam")).read():
            print(f"{name:8s}: did not finish (see {name}/log.chemFoam)")
            continue
        t, T = read_out(os.path.join(root, name))
        tb = t / si["tau"]
        Tex = quad[dT].T(tb) * si["Tu"]
        rel = np.abs(T / Tex - 1.0)
        k = int(np.argmax(rel))
        # the ignition delay: where T crosses T(Y 1/2), linear between steps, vs t(1/2)
        Th = (T0 + 0.5 * dT) * si["Tu"]
        i = int(np.argmax(T >= Th))
        th = tb[i - 1] + (Th - T[i - 1]) / (T[i] - T[i - 1]) * (tb[i] - tb[i - 1])
        res[name] = {"steps": len(t), "t_end": float(tb[-1]), "max_rel": float(rel.max()),
                     "at_t": float(tb[k]), "T_end": float(T[-1] / si["Tu"]),
                     "T_end_oracle": T0 + dT, "t_half_rel": float(th / te - 1.0)}
        print(f"{name:8s} {prop:8s} dt t(1/2)/{n:<5d} Hf_P/Hf_F {hf:4g}: {len(t)} steps to t {tb[-1]:.4f}, "
              f"max |T/T_ex - 1| {rel.max():.3e} at t {tb[k]:.5f}; T_end/Tu {T[-1] / si['Tu']:.12f} "
              f"(oracle {T0 + dT:g}); t(T(1/2))/t(1/2) - 1 {th / te - 1.0:.3e}")
    live = len(res) == len(CASES) and all(r["steps"] >= 2 * CASES[k][3] - 1 for k, r in res.items())
    print(f"live (every case ran to 2 t(1/2)): {live}")
    if not live:
        return 1
    for base, ctl in (("cv", "cv_ctl"), ("cp", "cp_ctl")):
        m, c = res[base]["max_rel"], res[ctl]["max_rel"]
        print(f"G0.5 {base}: {m:.3e} <= {BAND:g}: {'PASS' if m <= BAND else 'UNMET'}; "
              f"control {ctl} misses by {c:.3e} > {CTL_MISS:g}: {'PASS' if c > CTL_MISS else 'FAIL'}")
    o1 = math.log2(res["cv"]["t_half_rel"] / res["cv_fine"]["t_half_rel"]) / 2.0
    print(f"cv ignition-delay error order in dt (t(1/2)/2000 -> /8000): {o1:.3f}")
    with open(os.path.join(root, "compare.json"), "w") as f:
        json.dump({**res, "cv_order": o1}, f, indent=1)


def cmd_rk3ref():
    """tests/fvref_reactor.hpp kRk3Shu, kRk3Kutta2000, kTeV, kTeP, kYcpAtTe."""
    import mpmath as mp
    mp.mp.dps = 40
    L, th, t0 = mp.mpf("3656401.0984806987"), mp.mpf(THETA), mp.mpf(T0)
    qv, qp = mp.mpf("1.4") * (SIGMA - 1), mp.mpf(SIGMA - 1)

    def tof(Y, dT):
        return mp.quad(lambda s: mp.exp(th / (t0 + dT * (1 - mp.exp(s)))) / L,
                       [mp.log(Y), mp.log(0.9), mp.log(0.99), 0])
    te = tof(mp.mpf("0.5"), qv)
    print("kTeV", mp.nstr(te, 25), "kTeP", mp.nstr(tof(mp.mpf("0.5"), qp), 25))
    print("kYcpAtTe", mp.nstr(mp.findroot(lambda Y: tof(Y, qp) - te, 0.95), 20))
    f = lambda Y: -L * max(Y, 0) * mp.exp(-th / (t0 + qv * (1 - Y)))  # noqa: E731

    def shu(N):
        dt, Y = te / N, mp.mpf(1)
        for _ in range(N):
            Y1 = Y + dt * f(Y)
            Y2 = Y * 3 / 4 + Y1 / 4 + dt / 4 * f(Y1)
            Y = Y / 3 + 2 * Y2 / 3 + 2 * dt / 3 * f(Y2)
        return Y

    def kutta(N):
        dt, Y = te / N, mp.mpf(1)
        for _ in range(N):
            k1 = f(Y)
            k2 = f(Y + dt / 2 * k1)
            k3 = f(Y - dt * k1 + 2 * dt * k2)
            Y = Y + dt / 6 * (k1 + 4 * k2 + k3)
        return Y
    print("kRk3Kutta2000", mp.nstr(kutta(2000) - 0.5, 15))
    for N in (2000, 4000, 8000, 16000, 32000, 64000):
        print("kRk3Shu", N, mp.nstr(shu(N) - 0.5, 15), flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["setup", "run", "compare", "rk3ref"])
    ap.add_argument("--ode", default="seulex", choices=["seulex", "RKDP45"],
                    help="odeCoeffs solver (the plan's seulex; RKDP45 explicit: amendment A3)")
    a = ap.parse_args()
    if a.cmd == "rk3ref":
        return cmd_rk3ref() or 0
    return {"setup": cmd_setup, "run": cmd_run, "compare": cmd_compare}[a.cmd](a.ode) or 0


if __name__ == "__main__":
    sys.exit(main())
