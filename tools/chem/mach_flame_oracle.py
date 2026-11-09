#!/usr/bin/env python3
"""Finite-Mach steady 1D premixed C0 flame (issues.md N21, docs/chem.md P1 "At the
nozzle's Mach"): the full compressible Navier-Stokes ODE, box units (rho_u = R = 1,
cp = 3.5, lambda = 3.5, mu = Pr, rho D = 1 (Le 1), Tu = p0 = 1/(gamma M^2), q = cp
(sigma - 1) Tu, omega = Lambda rho Y exp(-theta Tu/T)), with (w = 1: dns' total energy
flux, dns/fvflux.hpp fvMeanFlux) and without (w = 0: stock reactingFoam EEqn.H) the
viscous work d/dx((4/3) mu u u_x). States T, Phi, Y, G, u; the eigenvalue m is the
consumption speed (rho_u Y_u = 1):
  momentum  (4/3) mu u' = m u + p - (m^2 + p0),  p = m R T/u
  energy    Phi = lambda T' + w (4/3) mu u u',  Phi' = m (cp T' + u u') - q omega
  species   G = m Y - rho D Y',  G' = -omega
x = -L .. 0 .. L in two collocation segments (solve_bvp has no interior condition):
fresh (T, Y, u) = (Tu, 1, m) upstream, Phi = 0 and G = m Y downstream, T(0) = (1 +
sigma) Tu/2. Initial guess: the low-Mach oracle (runs/flames/foam/oracle_s5_b8.npz).
Prints S(w = 1), S(w = 0), their difference in units of gamma M^2 (sigma - 1).

  mach_flame_oracle.py M [M ...]        (~1 min per Mach number)
"""
import os
import sys

import numpy as np
from scipy.integrate import solve_bvp

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from foamflame.model import hermite  # noqa: E402

ORACLE = os.path.join(HERE, "..", "..", "runs", "flames", "foam", "oracle_s5_b8.npz")
GAMMA, R, PR = 1.4, 1.0, 0.72
CP = GAMMA / (GAMMA - 1) * R
LAMBDA_K = CP           # lambda = rho_u cp S_L delta_D
MU = PR                 # mu = Pr lambda/cp
RHOD = LAMBDA_K / CP    # Le 1


def solve(M, w, L=25.0, n=4001, tol=1e-6):
    """(S, solve_bvp result) at flame Mach M, viscous work on (w = 1) or off (w = 0)."""
    d = np.load(ORACLE)
    sig, lam = float(d["sigma"]), float(d["Lambda"])
    theta = float(d["beta"]) * sig * sig / (sig - 1)
    Tu = p0 = 1.0 / (GAMMA * M * M)
    q = CP * (sig - 1) * Tu
    Tm = 0.5 * (1 + sig) * Tu

    def f(x, s, m):
        T, Phi, Y, G, u = s
        ux = 0.75 / MU * (m * u + m * R * T / u - (m * m + p0))
        Tx = (Phi - w * (4.0 / 3.0) * MU * u * ux) / LAMBDA_K
        om = lam * (m / u) * np.maximum(Y, 0.0) * np.exp(-theta * Tu / T)
        return np.vstack([Tx, m * (CP * Tx + u * ux) - q * om, (m * Y - G) / RHOD, -om, ux])

    def F(xi, S, p):
        return np.vstack([L * f(-L + L * xi, S[:5], p[0]), L * f(L * xi, S[5:], p[0])])

    def BC(A, B, p):
        m = p[0]
        a0, a1, b0, b1 = A[:5], B[:5], A[5:], B[5:]
        return np.concatenate([[a0[0] - Tu, a0[2] - 1.0, a0[4] - m], a1 - b0,
                               [b1[1], b1[3] - m * b1[2], a1[0] - Tm]])

    def guess(x):
        t = hermite(x, d["x"], d["T"], d["G"])
        tx = np.interp(x, d["x"], d["G"])
        Y = (sig - t) / (sig - 1)
        return np.vstack([t * Tu, LAMBDA_K * tx * Tu, Y, Y + RHOD * tx / (sig - 1), t])

    xi = np.linspace(0, 1, n)
    sol = solve_bvp(F, BC, xi, np.vstack([guess(-L + L * xi), guess(L * xi)]), p=[1.0],
                    tol=tol, max_nodes=300000, bc_tol=1e-10)
    if not sol.success:
        sys.exit(f"M {M} w {w}: {sol.message}")
    return float(sol.p[0]), sol


def main():
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    for M in map(float, sys.argv[1:]):
        unit = GAMMA * M * M * 4.0
        S = {w: solve(M, w)[0] for w in (1, 0)}
        S16 = solve(M, 1, n=16001)[0]
        print(f"M {M}: S(viscous work) {S[1]:.9f} (n 16001: {S16:.9f}), S(none) {S[0]:.9f}; "
              f"difference {S[1] - S[0]:+.4e} = {(S[1] - S[0]) / unit:.4f} gamma M^2 (sigma - 1); "
              f"vs 1 - gamma M^2 (sigma - 1): {S[1] - (1 - unit):+.4e} / {S[0] - (1 - unit):+.4e}",
              flush=True)


if __name__ == "__main__":
    main()
