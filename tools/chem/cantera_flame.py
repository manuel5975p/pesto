#!/usr/bin/env python3
"""Cantera reference for the P0 premixed oracle (docs/chem.md). Reference only:
NEVER a build dependency -- its outputs are checked in under tests/data/chem/.

The model is the solver's C0 model in dimensional form: two species F, P of
EQUAL molecular weight (N2 composition, 28.014), constant cp, unity Lewis
number, one irreversible step F => P with A = 1e9 /s and Ea = Ta R_u,
Tu = 300 K, Tb = sigma Tu = 1500 K (sigma 5, beta 8), 1 atm, N2 Lennard-Jones
transport (so lambda(T) is Cantera's own; the oracle reads it as a table).

Writes
  tests/data/chem/cantera_premixed.csv   T [K], lambda [W/m/K] on 4001 points
  tests/data/chem/cantera_premixed.json  parameters, the S_L refinement ladder,
                                         the finest S_L and its O(h) Richardson
                                         extrapolation (Cantera's convection is
                                         first-order upwind)

Usage: cantera_flame.py [OUTDIR]   (default: tests/data/chem next to this repo)

--real mode (masterplan real_bunsen S1, docs/chem.md "Model C2"): the REAL
CH4/air flame the partially premixed burner is calibrated against -- GRI-3.0
(Cantera's gri30.yaml, mixture-averaged transport), Tu 298 K, 1 atm, phi
1.0..1.6. Per phi: S_L (the finest rung of a refinement ladder), T_ad (the
equilibrium HP state), the burnt T of the flame's last point, delta_th =
(T_b - Tu)/max dT/dx, and the unburnt alpha_u = lambda/(rho cp), nu_u =
mu/rho, delta_D = alpha_u/S_L. Writes tests/data/chem/ch4_gri30.csv.

Usage: cantera_flame.py --real [OUTDIR] [--phis 0.6,0.65,...]
(--phis: only those, merged into the existing csv by phi; S1b added the lean
side 0.6..0.95 and the off-node checks the A(phi) calibration is judged on.)
"""
import json
import os
import sys
import time

import cantera as ct
import numpy as np

TU, SIGMA, BETA, A = 300.0, 5.0, 8.0, 1e9
TB = SIGMA * TU
TA = BETA * TB**2 / (TB - TU)
CPM = 1004.5                  # J/kg/K used to set the molar cp
CPMOL = CPM * 29.0 / 1000.0   # J/mol/K (species W is 28.014 from N2)
DH = CPMOL * (TB - TU)        # J/mol: F at Tu burns to P at Tb exactly

YAML = f"""
units: {{length: m, quantity: mol, activation-energy: J/mol}}
phases:
- name: gas
  thermo: ideal-gas
  elements: [N]
  species: [F, P]
  kinetics: gas
  reactions: all
  transport: unity-Lewis-number
  state: {{T: {TU}, P: 1 atm, Y: {{F: 1.0}}}}
species:
- name: F
  composition: {{N: 2}}
  thermo: {{model: constant-cp, T0: 300 K, h0: {DH} J/mol, s0: 190 J/mol/K, cp0: {CPMOL} J/mol/K, T-min: 200, T-max: 5000}}
  transport: {{model: gas, geometry: linear, diameter: 3.621, well-depth: 97.53, polarizability: 1.76, rotational-relaxation: 4.0}}
- name: P
  composition: {{N: 2}}
  thermo: {{model: constant-cp, T0: 300 K, h0: 0 J/mol, s0: 190 J/mol/K, cp0: {CPMOL} J/mol/K, T-min: 200, T-max: 5000}}
  transport: {{model: gas, geometry: linear, diameter: 3.621, well-depth: 97.53, polarizability: 1.76, rotational-relaxation: 4.0}}
reactions:
- equation: F => P
  rate-constant: {{A: {A}, b: 0, Ea: {TA * ct.gas_constant / 1000.0}}}
"""


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    out = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
        here, "..", "..", "tests", "data", "chem")
    os.makedirs(out, exist_ok=True)
    g = ct.Solution(yaml=YAML)
    P = ct.one_atm
    cp = g.cp_mass
    W = g.mean_molecular_weight
    Rs = ct.gas_constant / W

    # lambda(T): an identical-species mixture, so a function of T alone.
    TT = np.linspace(TU - 1, TB + 1, 4001)
    lam = np.empty_like(TT)
    for i, t in enumerate(TT):
        g.TPY = t, P, {"F": 1.0}
        lam[i] = g.thermal_conductivity
    with open(os.path.join(out, "cantera_premixed.csv"), "w") as fh:
        fh.write("T,lambda\n")
        for t, l in zip(TT, lam):
            fh.write(f"{t:.10g},{l:.12g}\n")

    # The refinement ladder: the default criteria (slope 0.01) read 0.6% off
    # the oracle, so the check uses the finest rung and the ladder itself.
    g.TPY = TU, P, {"F": 1.0}
    f = ct.FreeFlame(g, width=0.03)
    f.set_max_grid_points(f.flame, 8000)
    f.set_refine_criteria(ratio=3, slope=0.05, curve=0.1, prune=0)
    f.solve(loglevel=0, auto=True)
    ladder = []
    for crit in (0.02, 0.01, 0.005, 0.0025, 0.00125):
        t0 = time.time()
        f.set_refine_criteria(ratio=2, slope=crit, curve=crit, prune=0)
        f.solve(loglevel=0, refine_grid=True)
        ladder.append({"crit": crit, "points": len(f.grid),
                       "SL": float(f.velocity[0]), "Tb": float(f.T[-1])})
        print(f"crit={crit} SL={f.velocity[0]:.8g} npts={len(f.grid)} "
              f"Tb={f.T[-1]:.6f} {time.time() - t0:.1f}s", flush=True)
    fine, prev = ladder[-1]["SL"], ladder[-2]["SL"]
    doc = {"Tu": TU, "Tb": TB, "sigma": SIGMA, "beta": BETA, "A": A, "Ta": TA,
           "P": P, "cp": cp, "W": W, "Rs": Rs, "rho_u": P / (Rs * TU),
           "cantera": ct.__version__, "ladder": ladder,
           "SL_finest": fine, "SL_richardson": 2 * fine - prev}
    with open(os.path.join(out, "cantera_premixed.json"), "w") as fh:
        json.dump(doc, fh, indent=1)
    print(json.dumps({k: doc[k] for k in ("cp", "W", "SL_finest",
                                          "SL_richardson")}))


REAL_TU, REAL_PHIS = 298.0, (1.0, 1.05, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6)


def real_flame(phi):
    """GRI-3.0 CH4/air at phi, Tu 298 K, 1 atm: one row of ch4_gri30.csv."""
    g = ct.Solution("gri30.yaml")
    P = ct.one_atm
    g.set_equivalence_ratio(phi, "CH4", {"O2": 1.0, "N2": 3.76})
    g.TP = REAL_TU, P
    yF = float(g.Y[g.species_index("CH4")])
    yO = float(g.Y[g.species_index("O2")])
    rho_u, cp_u = g.density, g.cp_mass
    alpha_u = g.thermal_conductivity / (rho_u * cp_u)
    nu_u = g.viscosity / rho_u
    eq = ct.Solution("gri30.yaml")
    eq.TPX = g.TPX
    eq.equilibrate("HP")
    f = ct.FreeFlame(g, width=0.03)
    f.transport_model = "mixture-averaged"
    f.set_max_grid_points(f.flame, 4000)
    f.set_refine_criteria(ratio=3, slope=0.07, curve=0.14, prune=0)
    f.solve(loglevel=0, auto=True)
    ladder = []
    for crit in (0.04, 0.02, 0.01):
        f.set_refine_criteria(ratio=2, slope=crit, curve=crit, prune=0)
        f.solve(loglevel=0, refine_grid=True)
        ladder.append(float(f.velocity[0]))
    T = np.asarray(f.T)
    x = np.asarray(f.grid)
    dth = (T[-1] - T[0]) / np.max(np.diff(T) / np.diff(x))
    SL = ladder[-1]
    row = {"phi": phi, "SL": SL, "SL_prev": ladder[-2], "Tad": eq.T,
           "Tb": float(T[-1]), "sigma_ad": eq.T / REAL_TU,
           "dth": float(dth), "alpha_u": alpha_u, "nu_u": nu_u,
           "dD": alpha_u / SL, "dth_over_dD": float(dth) * SL / alpha_u,
           "yCH4": yF, "yO2": yO, "points": len(x)}
    print(" ".join(f"{k}={v:.6g}" for k, v in row.items()), flush=True)
    return row


def real(out, phis, merge):
    """Rows at phis; merge=True keeps the csv's rows at other phi (one
    simjob per subset: a flame is ~20 s, the lean ones longer)."""
    path = os.path.join(out, "ch4_gri30.csv")
    old = {}
    if merge and os.path.exists(path):
        with open(path) as fh:
            lines = [l.strip() for l in fh if not l.startswith("#")]
        keys = lines[0].split(",")
        for l in lines[1:]:
            vals = [float(v) for v in l.split(",")]
            old[round(vals[0], 6)] = dict(zip(keys, vals))
    for phi in phis:
        old[round(phi, 6)] = real_flame(phi)
    rows = [old[k] for k in sorted(old)]
    keys = list(rows[-1])
    with open(path, "w") as fh:
        fh.write(f"# Cantera {ct.__version__} gri30.yaml, mixture-averaged, "
                 f"Tu {REAL_TU} K, 1 atm (tools/chem/cantera_flame.py --real)\n")
        fh.write(",".join(keys) + "\n")
        for r in rows:
            fh.write(",".join(f"{r[k]:.8g}" for k in keys) + "\n")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--real":
        # --real [OUTDIR] [--phis a,b,c]: --phis runs only those and MERGES
        # them into the existing csv (sorted by phi).
        args = sys.argv[2:]
        phis, merge = REAL_PHIS, False
        if "--phis" in args:
            k = args.index("--phis")
            phis = tuple(float(v) for v in args[k + 1].split(","))
            merge = True
            del args[k:k + 2]
        here = os.path.dirname(os.path.abspath(__file__))
        out = args[0] if args else os.path.join(
            here, "..", "..", "tests", "data", "chem")
        os.makedirs(out, exist_ok=True)
        real(out, phis, merge)
    else:
        main()
