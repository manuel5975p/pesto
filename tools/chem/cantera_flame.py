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


if __name__ == "__main__":
    main()
