#!/usr/bin/env python3
"""The reacting quasi-1D nozzle (docs/nozzle.md "Reacting", masterplan_combnozzle P4): a Total fresh inlet, a
constant-area chamber holding a planar C0 flame, a choked cosine nozzle, a riemann exit, on one row.

  nozzle_react.py oracle --gamma G --eps-c E --eps-e E [--mach M] [--sigma S]
      the composition oracle (2.1) in flame units at the static fresh state Tu = 1/(gamma M^2), rho_u 1
  nozzle_react.py design --cells C --mach M --gamma G --sl S [--eps-e E] [--pa P] --out DIR
      solves eps_c for u_u = target S (S: P1/P3's 1D S_L,num at the same M, gamma, h, --shock hll), writes the one-row
      react mesh (y* ~ 50 delta_D, Lc 1, Lcon 6, Ldiv 4 y*, eps_e 2, nx a multiple of 11: the twin's xi grid) and bc.toml
      (total p0, t0, y [1]; riemann exit at P; [nozzle] astar = the contour's; [thrust] pa P, station Lc - 2 delta_th)
  nozzle_react.py twin DIR --p0b X --t0b X --out TDIR
      the inert viscous twin (R5): lc 0, y* 1, the same xi grid; Total at the burnt station's (p0b, T0b) as the --mach 1
      reference (p0, t0 ratios 1), rest at it (no ramp); prints Re* = rho0b c0b y*/mu (flame units, mu = Pr) for --re:
      dns flow --physics compressible --polymesh TDIR --bc TDIR/bc.toml --shock hll --mach 1 --re RE --prandtl 0.72
      --fv-precision fp64 --tt T (the inviscid twin: --inviscid for --re)
  nozzle_react.py argv DIR [--tt T] [--compensated]   the dns flame argv of a design (for simjob)
  nozzle_react.py modes DIR
      the chamber's linear acoustic modes (masterplan_combclose 2.4 step 6): P1, e-fold1, P2, e-fold2, the entropy transit
      t_e (x_f0 -> throat), t_flush (-> exit), and the controls (p-node inlet; compact nozzle at the chamber's end);
      scipy, cached in DIR/modes.json
  nozzle_react.py report DIR PREFIX --sl1d S --p1d P --t1d T [--twin TPREFIX | --twin-cd X --twin-cf Y] [--start SPREFIX]
                         [--from T | --window A B] [--compensated] [--json OUT]
      the G4 gates (masterplan_combnozzle P4, masterplan_combclose 2.5) off PREFIX_stats.csv and PREFIX_thrust.csv over the
      rule window W = [tt - k P, tt] (2.4: t_s = t_e + 10, P the ring fitted on uFresh over [t_s, tt] or the model's, k =
      floor((tt - t_s)/P) >= 1, else exit 1); every rate an endpoint ledger difference; [20, 80], [40, 80] beside as
      legacy. The 1D reference: S_L,num and its fresh state (window means of the 1D --shock hll run at the same M, gamma, h)

q is the SOLVER's heat release cp (sigma - 1) Tu (masterplan_combnozzle R5): T0b = T0u + (sigma - 1) Tu, never sigma T0u.
Python twin of dns/fvnozzle fvReactingNozzleDesign (test_fvref react pins the two to 1e-12).
"""
import argparse
import json
import math
import os
import re
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import nozzle_mesh  # noqa: E402

DTH = 1.601651  # delta_th / delta_D of C0 sigma 5 beta 8 (c.flameEig.dthOverDD)
SIGMA, BETA, PR = 5.0, 8.0, 0.72
# The rig (masterplan_combnozzle A17): y* 50 delta_D, Lc 1 y* (the plan's 50 delta_D chamber), Lcon 6 y*, Ldiv 4 y*,
# eps_e 2. The plan's Lcon 4 y* slopes to 64 deg at eps_c 6.3 and R8's --shock refusal counts 1400 unlined cells (the
# axis face's most antiparallel partner, the wall, at cos -0.43; 54 deg: -0.585); its eps_e 6.25 puts rho_e at 0.052
# rho_0b, where the viscous dt (flame units, mu = Pr) is 2.5e-5 tau: 3.1e6 steps for 80 tau, past the job cap.
LC, LCON, LDIV, YSTAR = 1.0, 6.0, 4.0, 50.0
DEPTH = 0.1  # nozzle_mesh.writePolyMesh's planar depth


def rayleighBurnt(g, R, cp, q, Tu, pu, uu):
    """fvRayleighBurnt: the constant-area complete burn, subsonic root."""
    rhou = pu / (R * Tu)
    m = rhou * uu
    P = pu + m * uu
    H = cp * Tu + uu * uu / 2 + q
    f = lambda ub: cp * (P - m * ub) * ub / (m * R) + ub * ub / 2 - H
    lo, hi = uu, (P / m) * g / (g + 1)
    for _ in range(200):
        mid = (lo + hi) / 2
        if f(mid) < 0:
            lo = mid
        else:
            hi = mid
    ub = (lo + hi) / 2
    pb = P - m * ub
    Tb = pb * ub / (m * R)
    Mb = ub / math.sqrt(g * R * Tb)
    T0b = Tb * (1 + (g - 1) / 2 * Mb * Mb)
    p0b = pb * (T0b / Tb) ** (g / (g - 1))
    return dict(ub=ub, pb=pb, Tb=Tb, Mb=Mb, T0b=T0b, p0b=p0b)


def areaMach(ar, g, sup):
    """fvAreaMach."""
    A = lambda M: (1 / M) * ((2 / (g + 1)) * (1 + (g - 1) / 2 * M * M)) ** ((g + 1) / (2 * (g - 1)))
    if ar <= 1.0:
        return 1.0
    lo, hi = (1.0, 60.0) if sup else (1e-6, 1.0)
    for _ in range(200):
        mid = (lo + hi) / 2
        if (A(mid) < ar) == sup:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def cfIdeal(g, epsE):
    """The ideal vacuum C_F of a gamma gas expanded to eps_e (2.1 step 5)."""
    pe = (1 + (g - 1) / 2 * areaMach(epsE, g, True) ** 2) ** (-g / (g - 1))
    return gammaOf(g) * math.sqrt(2 * g / (g - 1) * (1 - pe ** ((g - 1) / g))) + pe * epsE


def gammaOf(g):
    return math.sqrt(g) * (2 / (g + 1)) ** ((g + 1) / (2 * (g - 1)))


def freshAt(g, R, cp, q, Mu, p0u, T0u):
    Tu = T0u / (1 + (g - 1) / 2 * Mu * Mu)
    pu = p0u * (Tu / T0u) ** (g / (g - 1))
    rhou = pu / (R * Tu)
    uu = Mu * math.sqrt(g * R * Tu)
    return dict(Mu=Mu, Tu=Tu, pu=pu, rhou=rhou, uu=uu, m=rhou * uu, b=rayleighBurnt(g, R, cp, q, Tu, pu, uu))


def design(g, R, cp, q, sigma, epsC, epsE, p0u, T0u):
    """fvReactingNozzleDesign (masterplan_combnozzle 2.1)."""
    if not epsC > 1:
        raise SystemExit(f"eps_c {epsC}: no throat")
    G = gammaOf(g)
    excess = lambda o: o["m"] * epsC - G * o["b"]["p0b"] / math.sqrt(R * o["b"]["T0b"])
    lo, hi = 1e-6, 0.3
    if not (excess(freshAt(g, R, cp, q, lo, p0u, T0u)) < 0 and excess(freshAt(g, R, cp, q, hi, p0u, T0u)) >= 0):
        raise SystemExit(f"no choke root for eps_c {epsC}")
    for _ in range(200):
        Mu = (lo + hi) / 2
        if excess(freshAt(g, R, cp, q, Mu, p0u, T0u)) < 0:
            lo = Mu
        else:
            hi = Mu
    d = freshAt(g, R, cp, q, (lo + hi) / 2, p0u, T0u)
    d["Gamma"] = G
    d["Me"] = areaMach(epsE, g, True)
    d["pe"] = (1 + (g - 1) / 2 * d["Me"] ** 2) ** (-g / (g - 1))
    d["CFvac"] = G * math.sqrt(2 * g / (g - 1) * (1 - d["pe"] ** ((g - 1) / g))) + d["pe"] * epsE
    d["MuLowM"] = G / (math.sqrt(g * sigma) * epsC)
    return d


def flameUnits(g, M, sigma=SIGMA):
    """R 1, Tu = p_u = 1/(gamma M^2) (rho_u 1), cp, q = cp (sigma - 1) Tu."""
    Tu = 1.0 / (g * M * M)
    cp = g / (g - 1)
    return dict(R=1.0, Tu=Tu, cp=cp, q=cp * (sigma - 1) * Tu)


def inverse(g, M, uTarget, sigma=SIGMA):
    """eps_c for a static fresh state (Tu, rho_u 1) at u_u = uTarget: closed (the choke relation solved for eps_c).
    Returns (eps_c, M_u, p0 ratio, t0 ratio)."""
    u = flameUnits(g, M, sigma)
    Mu = uTarget / math.sqrt(g * u["Tu"])
    t0 = 1 + (g - 1) / 2 * Mu * Mu
    p0 = t0 ** (g / (g - 1))
    b = rayleighBurnt(g, 1.0, u["cp"], u["q"], u["Tu"], u["Tu"], uTarget)
    epsC = gammaOf(g) * b["p0b"] / (math.sqrt(b["T0b"]) * uTarget)
    return epsC, Mu, p0, t0


def cmdOracle(a):
    u = flameUnits(a.gamma, a.mach, a.sigma)
    Tu = u["Tu"]
    # p0u, T0u for the static fresh state Tu, rho_u 1 at the oracle's own M_u: iterate the design (its M_u moves them).
    d = design(a.gamma, 1.0, u["cp"], u["q"], a.sigma, a.eps_c, a.eps_e, Tu * a.p0, Tu * a.t0)
    T0u = Tu * a.t0
    out = dict(Mu=d["Mu"], uu=d["uu"], Tu=d["Tu"], Mb=d["b"]["Mb"], T0b=d["b"]["T0b"], p0b=d["b"]["p0b"],
               p0b_p0u=d["b"]["p0b"] / (Tu * a.p0), T0b_oracle=T0u + (a.sigma - 1) * Tu, T0b_A3=a.sigma * T0u, Me=d["Me"],
               CFvac=d["CFvac"], MuLowM=d["MuLowM"])
    print(json.dumps(out, indent=1))


def meshNx(cells):
    """nx a multiple of k = (Lc + Lcon + Ldiv)/Lc (11): the chamber ends on a node, the twin's grid is the rest."""
    h = DTH / cells
    L = LC + LCON + LDIV
    k = round(L / LC)
    nx = k * round(L * YSTAR / (k * h))
    return nx, h, nx * h / L


def cmdDesign(a):
    nx, h, S = meshNx(a.cells)
    epsC, Mu, p0, t0 = inverse(a.gamma, a.mach, a.target * a.sl)
    y, x0, x2 = nozzle_mesh.reactContour(epsC, a.eps_e, LC, LCON, LDIV, S)
    X, Y = nozzle_mesh.quadMesh(nx, 1, y, x0, x2)
    os.makedirs(a.out, exist_ok=True)
    nozzle_mesh.writePolyMesh(a.out, X, Y)
    astar = S * DEPTH
    station = S * LC - 2 * DTH
    with open(os.path.join(a.out, "bc.toml"), "w") as f:
        f.write(f"# tools/fv/nozzle_react.py design --cells {a.cells} --mach {a.mach} --gamma {a.gamma} --sl {a.sl!r} --target {a.target!r}: "
                f"eps_c {epsC!r}, M_u {Mu!r}\n")
        f.write("[reference]\ndirection = [1, 0, 0]\nlift = [0, 1, 0]\n\n")
        f.write(f"[nozzle]\nastar = {astar:.17g}\n\n")
        f.write(f"[thrust]\npa = {a.pa:.17g}\nstation = {station:.17g}\n\n")
        f.write(f'[patch.inlet]\ntype = "total"\np0 = {p0:.17g}\nt0 = {t0:.17g}\ny = [1.0]\n\n')
        f.write(f'[patch.outlet]\ntype = "outflow"\nexit = "riemann"\np = {a.pa:.17g}\ntback = 1\ny = [0.0]\n\n')
        f.write('[patch.wall]\ntype = "slip"\n\n[patch.axis]\ntype = "slip"\n')
    meta = dict(cells=a.cells, mach=a.mach, gamma=a.gamma, sl=a.sl, target=a.target, nx=nx, h=h, scale=S, epsC=epsC, epsE=a.eps_e, Mu=Mu,
                p0=p0, t0=t0, astar=astar, station=station, pa=a.pa, upstream=a.upstream)
    with open(os.path.join(a.out, "design.json"), "w") as f:
        json.dump(meta, f, indent=1)
    print(json.dumps(meta))


def cmdArgv(a):
    m = json.load(open(os.path.join(a.dir, "design.json")))
    argv = ["flame", "--mach", repr(m["mach"]), "--expansion", "5", "--zeldovich", "8", "--prandtl", "0.72", "--polymesh",
            a.dir, "--bc", os.path.join(a.dir, "bc.toml"), "--shock", "hll", "--fv-precision", "fp64", "--upstream",
            repr(a.upstream if a.upstream is not None else m["upstream"]), "--tt", repr(a.tt), "--emit", "stats", "--quiet"]
    if m["gamma"] != 1.4:
        argv += ["--gamma", repr(m["gamma"])]
    if a.compensated:
        argv += ["--compensated"]
    print(" ".join(argv))


def cmdTwin(a):
    m = json.load(open(os.path.join(a.dir, "design.json")))
    g = m["gamma"]
    nxT = m["nx"] - m["nx"] // round((LC + LCON + LDIV) / LC)
    y, x0, x2 = nozzle_mesh.reactContour(m["epsC"], m["epsE"], 0.0, LCON, LDIV, 1.0)
    X, Y = nozzle_mesh.quadMesh(nxT, 1, y, x0, x2)
    os.makedirs(a.out, exist_ok=True)
    nozzle_mesh.writePolyMesh(a.out, X, Y)
    # The flame's burnt station in flame units: R 1, mu = Pr (Transport::air(Pr, Tu)), y* = scale.
    rho0 = a.p0b / a.t0b
    c0 = math.sqrt(g * a.t0b)
    Re = rho0 * c0 * m["scale"] / PR
    pr = m["pa"] / (g * m["mach"] ** 2) / a.p0b  # the flame's pa over p0b: the twin's exit and rest ratio
    with open(os.path.join(a.out, "bc.toml"), "w") as f:
        f.write(f"# tools/fv/nozzle_react.py twin: lc 0, y* 1, {nxT} cells (the flame's xi grid), Re* {Re!r}\n")
        f.write(f"[reference]\nlength = 1\narea = {DEPTH:.17g}\ndirection = [1, 0, 0]\nlift = [0, 1, 0]\n\n")
        # Rest at the stagnation state, no ramp: the start is an isentropic expansion (a rest state at the exit p is a
        # high-entropy slug the contraction holds ~40 t*).
        f.write(f"[nozzle]\np = 1\nt = 1\nastar = {DEPTH:.17g}\n\n")
        f.write(f"[thrust]\npa = {pr:.17g}\n\n")
        f.write('[patch.inlet]\ntype = "total"\np0 = 1\nt0 = 1\n\n')
        f.write(f'[patch.outlet]\ntype = "outflow"\nexit = "riemann"\np = {pr:.17g}\ntback = 1\n\n')
        f.write('[patch.wall]\ntype = "slip"\nforce = true\n\n[patch.axis]\ntype = "slip"\n')
    print(json.dumps(dict(nx=nxT, Re=Re, rho0b=rho0, c0b=c0, ystar=m["scale"], pa=pr)))


# --- modes: the chamber's linear acoustic modes (masterplan_combclose 2.4 step 6) -------------------------------------
# A port of analysis_combclose_3's net.py + compact.py: quasi-1D linearized Euler in (p', u', sigma' = s'/cp), modes
# exp(i w t); a Total inlet (p0' = 0, sigma' = 0), the fresh duct to x_f0, a compact moving flame (flame-frame RH +
# kinematic, S_L constant), the burnt duct and the exact cosine contraction on the design's isentropic base flow,
# integrated to M_end 0.995, where the throat's regularity l.r = 0 is the root condition. Flame units (rho_u 1, R 1).

def _areaMachSub(ar, g):
    """fvAreaMach's subsonic root, numpy (60 halvings of [1e-6, 1]; ar <= 1 gives 1)."""
    ar = np.asarray(ar, float)
    A = lambda M: (1 / M) * ((2 / (g + 1)) * (1 + (g - 1) / 2 * M * M)) ** ((g + 1) / (2 * (g - 1)))
    lo, hi = np.full_like(ar, 1e-6), np.ones_like(ar)
    for _ in range(60):
        mid = (lo + hi) / 2
        big = A(mid) >= ar
        lo, hi = np.where(big, mid, lo), np.where(big, hi, mid)
    return np.where(ar <= 1.0, 1.0, (lo + hi) / 2)


def _areaMachSup(ar, g):
    ar = np.asarray(ar, float)
    A = lambda M: (1 / M) * ((2 / (g + 1)) * (1 + (g - 1) / 2 * M * M)) ** ((g + 1) / (2 * (g - 1)))
    lo, hi = np.ones_like(ar), np.full_like(ar, 60.0)
    for _ in range(80):
        mid = (lo + hi) / 2
        small = A(mid) < ar
        lo, hi = np.where(small, mid, lo), np.where(small, hi, mid)
    return np.where(ar <= 1.0, 1.0, (lo + hi) / 2)


class Rig:
    """A design's inviscid base flow on its one-row rig (design.json), flame units."""

    def __init__(self, m):
        g, M = m["gamma"], m["mach"]
        u = flameUnits(g, M)
        self.g, self.cp, self.q = g, u["cp"], u["q"]
        Tu = u["Tu"]
        self.T0u, self.p0u = m["t0"] * Tu, m["p0"] * Tu
        d = design(g, 1.0, self.cp, self.q, SIGMA, m["epsC"], m["epsE"], self.p0u, self.T0u)
        self.d = d
        self.Tu, self.pu, self.rhou, self.uu = d["Tu"], d["pu"], d["rhou"], d["uu"]
        self.cu = math.sqrt(g * self.Tu)
        b = d["b"]
        self.ub, self.pb, self.Tb, self.T0b, self.p0b = b["ub"], b["pb"], b["Tb"], b["T0b"], b["p0b"]
        self.rhob, self.cb = self.pb / self.Tb, math.sqrt(g * self.Tb)
        S = m["scale"]
        self.Lc, self.Lcon, self.Ldiv = LC * S, LCON * S, LDIV * S
        self.epsc, self.epse = m["epsC"], m["epsE"]
        self.xf = m["upstream"] * DTH
        self.xth, self.xexit = self.Lc + self.Lcon, self.Lc + self.Lcon + self.Ldiv

    def area(self, x):
        """A/A* of the contour (nozzle_mesh.reactContour)."""
        if x <= self.Lc:
            return self.epsc
        if x <= self.xth:
            return 1 + (self.epsc - 1) * (1 + math.cos(math.pi * (x - self.Lc) / self.Lcon)) / 2
        return 1 + (self.epse - 1) * (1 - math.cos(math.pi * min(x - self.xth, self.Ldiv) / self.Ldiv)) / 2

    def dlnA(self, x):
        """d ln A/dx on the chamber and the contraction (the operator's domain ends before the throat)."""
        if x <= self.Lc:
            return 0.0
        dA = -(self.epsc - 1) * math.pi / self.Lcon * math.sin(math.pi * (x - self.Lc) / self.Lcon) / 2
        return dA / self.area(x)

    def base(self, x):
        """(u, p, rho, du/dx, dp/dx): fresh uniform before x_f0, burnt subsonic isentropic at (p0b, T0b) past it."""
        g = self.g
        if x < self.xf:
            return self.uu, self.pu, self.rhou, 0.0, 0.0
        M = float(_areaMachSub(self.area(x), g))
        T = self.T0b / (1 + (g - 1) / 2 * M * M)
        p = self.p0b * (T / self.T0b) ** (g / (g - 1))
        u = M * math.sqrt(g * T)
        L = self.dlnA(x)
        return u, p, p / T, u * (-L / (1 - M * M)), p * g * M * M / (1 - M * M) * L

    def speeds(self, x, sup=False):
        """u(x), c(x) on numpy x past x_f0: the chamber's Rayleigh u_b (design), the contour's area-Mach elsewhere."""
        g = self.g
        x = np.asarray(x, float)
        ar = np.array([self.area(v) for v in x])
        M = _areaMachSup(ar, g) if sup else _areaMachSub(ar, g)
        T = self.T0b / (1 + (g - 1) / 2 * M * M)
        u, c = M * np.sqrt(g * T), np.sqrt(g * T)
        ch = x <= self.Lc
        return np.where(ch, self.ub, u), np.where(ch, self.cb, c)

    def transit(self):
        """t_e (x_f0 -> throat), t_flush (x_f0 -> exit), t_ac (inlet -> throat, downstream acoustic)."""
        n = 200001
        xs = np.linspace(self.Lc, self.xth, n)
        u, c = self.speeds(xs)
        con, conAc = np.trapezoid(1 / u, xs), np.trapezoid(1 / (c + u), xs)
        xd = np.linspace(self.xth, self.xexit, n)
        ud, _ = self.speeds(xd, sup=True)
        ch = (self.Lc - self.xf) / self.ub
        te = ch + con
        return dict(t_e=te, t_e_chamber=ch, t_e_contraction=con, t_flush=te + np.trapezoid(1 / ud, xd),
                    t_ac=self.xf / (self.cu + self.uu) + (self.Lc - self.xf) / (self.cb + self.ub) + conAc,
                    t_ac_chamber=self.xf / (self.cu + self.uu) + (self.Lc - self.xf) / (self.cb + self.ub))

    # The linearized operator (net.py rhs / funcF): A(x) y' = -r.
    def _r(self, x, y, w):
        g = self.g
        p1, u1, s1 = y
        u, p, rho, ux, px = self.base(x)
        L = self.dlnA(x)
        rho1 = rho * (p1 / (g * p) - s1)
        r1 = 1j * w * p1 + u1 * px + g * p1 * (ux + u * L) + g * p * u1 * L
        r2 = 1j * w * u1 + u1 * ux - rho1 / rho ** 2 * px
        return u, p, rho, r1, r2

    def rhs(self, x, y, w):
        u, p, rho, r1, r2 = self._r(x, y, w)
        d = np.linalg.solve(np.array([[u, self.g * p], [1 / rho, u]], complex), -np.array([r1, r2]))
        return [d[0], d[1], -1j * w * y[2] / u]

    def regularity(self, x, y, w):
        u, p, rho, r1, r2 = self._r(x, y, w)
        return (r1 - rho * u * r2) / (self.g * p)

    def inlet(self, kind):
        """u'(0) = 1. total: p0' = 0, sigma' = 0; pnode: p' = 0 (the control)."""
        g = self.g
        if kind == "pnode":
            return [0.0, 1.0, 0.0]
        m2 = (g - 1) / 2 * (self.uu / self.cu) ** 2
        k = g / (g - 1) * m2 / (1 + m2)
        return [-k * 2 * (1.0 / self.uu) / (1 - k * (1 - 1 / g)) * self.pu, 1.0, 0.0]

    def flameJump(self, y, w):
        """The moving compact front: flame-frame RH (mass, momentum, energy, q per fresh mass), S_L' = 0."""
        g, cp = self.g, self.cp
        p1, u1, s1 = y
        rho1 = self.rhou * (p1 / (g * self.pu) - s1)
        T1 = self.Tu * (p1 / self.pu - rho1 / self.rhou)
        wv = u1
        m1 = rho1 * self.uu
        mb = self.rhou * self.uu
        uu, ub, pb, Tb, rhob = self.uu, self.ub, self.pb, self.Tb, self.rhob

        def res(z):
            p2, u2, s2 = z[0] + 1j * z[1], z[2] + 1j * z[3], z[4] + 1j * z[5]
            rho2 = rhob * (p2 / (g * pb) - s2)
            T2 = Tb * (p2 / pb - rho2 / rhob)
            e1 = rho2 * ub + rhob * (u2 - wv) - m1
            e2 = (p1 + m1 * uu + mb * (u1 - wv)) - (p2 + m1 * ub + mb * (u2 - wv))
            e3 = (cp * T1 + uu * (u1 - wv)) - (cp * T2 + ub * (u2 - wv))
            return np.array([e1.real, e1.imag, e2.real, e2.imag, e3.real, e3.imag])

        A, b0 = np.zeros((6, 6)), res(np.zeros(6))
        for i in range(6):
            e = np.zeros(6)
            e[i] = 1
            A[:, i] = res(e) - b0
        z = np.linalg.solve(A, -b0)
        return [z[0] + 1j * z[1], z[2] + 1j * z[3], z[4] + 1j * z[5]]

    def shoot(self, w, inlet="total", Mend=0.995, compact=False):
        from scipy.integrate import solve_ivp
        w = complex(w)
        kw = dict(args=(w,), method="DOP853", rtol=1e-10, atol=1e-14)
        s1 = solve_ivp(self.rhs, (0, self.xf), np.array(self.inlet(inlet), complex), **kw)
        yb = np.array(self.flameJump(s1.y[:, -1], w), complex)
        if compact:  # the control: a compact choked nozzle (M' = 0) at the chamber's end
            s2 = solve_ivp(self.rhs, (self.xf, self.Lc), yb, **kw)
            p1, u1, sg = s2.y[:, -1]
            u, p = self.base(self.Lc)[:2]
            return u1 / u - 0.5 * ((self.g - 1) / self.g * p1 / p + sg)
        g = self.g
        Ae = (1 / Mend) * ((2 / (g + 1)) * (1 + (g - 1) / 2 * Mend ** 2)) ** ((g + 1) / (2 * (g - 1)))
        xe = self.Lc + self.Lcon / math.pi * math.acos(2 * (Ae - 1) / (self.epsc - 1) - 1)
        s2 = solve_ivp(self.rhs, (self.xf, xe), yb, **kw)
        return self.regularity(xe, s2.y[:, -1], w)

    def root(self, w0, **kw):
        from scipy.optimize import fsolve

        def f(z):
            F = self.shoot(complex(z[0], z[1]), **kw)
            return [F.real, F.imag]

        z, info, ok, msg = fsolve(f, [w0.real, w0.imag], xtol=1e-12, full_output=True)
        return complex(z[0], z[1]), ok == 1


def _mode(w):
    return dict(w=[w.real, w.imag], period=2 * math.pi / w.real, efold=1 / w.imag if w.imag else math.inf)


def modesOf(ddir, fresh=False):
    """modes DIR: P1, e-fold1, P2, e-fold2, t_e (+ t_flush, t_ac) and the controls; cached in DIR/modes.json (keyed on
    design.json)."""
    m = json.load(open(os.path.join(ddir, "design.json")))
    path = os.path.join(ddir, "modes.json")
    if not fresh and os.path.exists(path):
        c = json.load(open(path))
        if c.get("design") == m:
            return c
    rig = Rig(m)
    tr = rig.transit()
    w0 = 2 * math.pi / (4 * tr["t_ac"])
    w1, ok1 = rig.root(complex(w0, w0 / 9))
    # Mode 2: the root polished from the next real-axis minimum of |F| (net.py's scan; complex starts near 3 w0 also
    # land on heavily damped entropy-coupled roots, e-fold ~9 tau, that no real-axis resonance shows).
    ws = np.linspace(0.5 * w0, 4 * w0, 60)
    F = [abs(rig.shoot(w)) for w in ws]
    w2, ok2 = complex(math.nan, math.nan), False
    for i in range(1, len(ws) - 1):
        if F[i] < F[i - 1] and F[i] < F[i + 1]:
            w, ok = rig.root(complex(ws[i], 0.0))
            if ok and w.real > 1.1 * w1.real:
                w2, ok2 = w, ok
                break
    wp, okp = rig.root(w1, inlet="pnode")
    wc0 = 2 * math.pi / (4 * tr["t_ac_chamber"])
    wc, okc = rig.root(complex(wc0, wc0 / 20), compact=True)
    o = dict(design=m, transit=tr, mode1=_mode(w1), mode2=_mode(w2), control_pnode=_mode(wp), control_compact=_mode(wc),
             converged=dict(mode1=ok1, mode2=ok2, pnode=okp, compact=okc), w0=w0, x_f0=rig.xf)
    with open(path, "w") as f:
        json.dump(o, f, indent=1)
    return o


def cmdModes(a):
    o = modesOf(a.dir, fresh=True)
    tr, m1, m2 = o["transit"], o["mode1"], o["mode2"]
    print(f"mode 1: period {m1['period']:.3f} tau, e-fold {m1['efold']:.2f} tau")
    print(f"mode 2: period {m2['period']:.3f} tau, e-fold {m2['efold']:.2f} tau")
    print(f"t_e (x_f0 {o['x_f0']:.3f} -> throat) {tr['t_e']:.3f} tau (chamber {tr['t_e_chamber']:.3f}, contraction "
          f"{tr['t_e_contraction']:.3f}); t_flush (-> exit) {tr['t_flush']:.3f}; t_ac (inlet -> throat) {tr['t_ac']:.3f}")
    print(f"control p-node inlet: period {o['control_pnode']['period']:.3f}, e-fold {o['control_pnode']['efold']:.2f} tau")
    print(f"control compact choked nozzle at the chamber's end: period {o['control_compact']['period']:.3f}, e-fold "
          f"{o['control_compact']['efold']:.2f} tau")
    print(f"converged: {o['converged']}")


# --- report: the G4 gates (masterplan_combnozzle P4; window rule and estimators masterplan_combclose 2.4, 2.5) -----

A_T, A_P = 2.33532, -0.5  # Oracle.sens() (masterplan_combnozzle R3, A6 G1.4)


def readCsv(path):
    import csv
    with open(path) as f:
        rows = list(csv.DictReader(l for l in f if not l.startswith("#")))
    return {k: [float(r[k]) for r in rows] for k in rows[0]}


def labelOf(path):
    with open(path) as f:
        for l in f:
            return l[1:].strip() if l.startswith("#") else ""
    return ""


def mean(v):
    return sum(v) / len(v)


def slope(t, x):
    tm, xm = mean(t), mean(x)
    return sum((a - tm) * (b - xm) for a, b in zip(t, x)) / sum((a - tm) ** 2 for a in t)


def snap(t, ta):
    """The first row with t >= ta."""
    return next(i for i, v in enumerate(t) if v >= ta - 1e-12)


def fitRing(t, v, ts, pModel, eModel):
    """2.4 step 3: A e^{-l (t - ts)} cos(w (t - ts) + phi) + c0 + c1 (t - ts) + c2 (t - ts)^2 on [ts, tt]; the least
    residual rms over the starts. Accepted iff resid/|A| <= 0.05 and |P/P_model - 1| <= 0.10."""
    from scipy.optimize import curve_fit
    t, v = np.asarray(t), np.asarray(v)
    if len(t) < 20:
        return dict(accepted=False, why=f"{len(t)} rows in [t_s, tt]: no fit")
    f = lambda t, A, lam, w, ph, c0, c1, c2: A * np.exp(-lam * (t - ts)) * np.cos(w * (t - ts) + ph) + c0 + c1 * (t - ts) \
        + c2 * (t - ts) ** 2
    best = None
    for r in np.arange(0.70, 1.30 + 1e-9, 0.05):
        for ph in (0.0, 1.5, 3.0, 4.5):
            try:
                p, _ = curve_fit(f, t, v, p0=[np.std(v) * 1.4, 1 / eModel, 2 * math.pi / pModel * r, ph, v.mean(), 0, 0],
                                 maxfev=20000)
            except Exception:
                continue
            rr = float(np.sqrt(np.mean((v - f(t, *p)) ** 2)))
            if best is None or rr < best[1]:
                best = (p, rr)
    if best is None:
        return dict(accepted=False, why="no fit converged")
    p, rr = best
    P, e, A = 2 * math.pi / abs(p[2]), (1 / p[1] if p[1] != 0 else math.inf), abs(p[0])
    o = dict(period=P, efold=e, amp=A, amp_rel=A / abs(v.mean()), resid_amp=rr / A if A > 0 else math.inf, rows=len(t))
    why = []
    if not o["resid_amp"] <= 0.05:
        why.append(f"resid/amp {o['resid_amp']:.3f} > 0.05")
    if not abs(P / pModel - 1) <= 0.10:
        why.append(f"P_fit/P_model - 1 {P / pModel - 1:+.3f} beyond 0.10")
    o["accepted"], o["why"] = not why, "; ".join(why)
    return o


def windowGates(m, rig, st, th, ta, tb, sl1d, p1d, t1d, twin):
    """Every windowed gate over [ta, tb] snapped to rows (2.4 step 5, 2.5): i1 = the first row with t >= ta, i2 the last
    with t <= tb, the same rows in _stats.csv and _thrust.csv; every rate an endpoint ledger difference."""
    g, cp, q = rig.g, rig.cp, rig.q
    # One _thrust.csv row per stats row (the stats print 6 digits): the rows are snapped on the ledger's t.
    assert len(th["t"]) == len(st["t"]) and all(abs(a - b) <= 1e-5 * abs(a) for a, b in zip(th["t"], st["t"])), \
        "stats and thrust rows differ"
    i1, i2 = snap(th["t"], ta), max(i for i, v in enumerate(th["t"]) if v <= tb + 1e-12)
    j1, j2 = i1, i2
    dt = th["t"][i2] - th["t"][i1]
    D = lambda k: th[k][i2] - th[k][i1]
    w = {k: v[j1:j2 + 1] for k, v in st.items()}
    wt = {k: v[i1:i2 + 1] for k, v in th.items()}
    o = dict(window=[th["t"][i1], th["t"][i2]], rows=i2 - i1 + 1)
    # G4.2
    Sc, uF, pF, TF = mean(w["SL"]), mean(w["uFresh"]), mean(w["pFresh"]), mean(w["TFresh"])
    dS = A_T * (TF - t1d) / t1d + A_P * (pF - p1d) / p1d
    corr = abs(dS) > 5e-4 / 4
    gap = Sc - sl1d * (1 + dS) if corr else Sc - sl1d
    o["G4.2"] = dict(Sc=Sc, gap=gap, dS_fresh=dS, corrected=corr, band=5e-4, ok=abs(gap) <= 5e-4)
    # G4.3: endpoint kinematics; the regression slope beside
    xk = (w["xf"][-1] - w["xf"][0]) / dt
    sl = slope(w["t"], w["xf"])
    o["G4.3"] = dict(dxf_dt=xk, uu=uF, gap=xk - (uF - Sc), band=2e-3 * sl1d, ok=abs(xk - (uF - Sc)) <= 2e-3 * sl1d,
                     regression_slope=sl, regression_gap=sl - (uF - Sc))
    # G4.4
    T0b, p0b = mean(wt["T0b"]), mean(wt["p0b"])
    T0bO = rig.T0u + (SIGMA - 1) * rig.Tu
    fuel = D("I_exit_ry") / D("Omega")
    o["G4.4"] = dict(T0b=T0b, rel=T0b / T0bO - 1, fuelOut_burnIn=fuel, ok=abs(T0b / T0bO - 1) <= 1e-4 and abs(fuel) <= 1e-8)
    # G4.5: the throat flux mOut + Delta Mth / Delta t (exit and station forms beside)
    mOut, mIn = D("I_exit_mass") / dt, -D("I_in_mass") / dt
    Tvac = D("I_exit_mx") / dt
    ideal = m["astar"] * p0b * math.sqrt(g / T0b) * (2 / (g + 1)) ** ((g + 1) / (2 * (g - 1)))
    hasM = "Mth" in th
    mTh = mOut + D("Mth") / dt if hasM else math.nan
    mSt = mOut + D("Mst") / dt if hasM else math.nan
    CdB = mTh / ideal if hasM else mOut / ideal
    CfB = Tvac / (p0b * m["astar"])
    o["G4.5"] = dict(form="throat flux" if hasM else "exit flux (no Mst/Mth columns)", CdB=CdB, CdB_exit=mOut / ideal,
                     CdB_station=mSt / ideal, contraction_storage=(D("Mst") - D("Mth")) / (dt * mOut) if hasM else math.nan,
                     CfBvac=CfB, mdotIn=mIn, mdotOut=mOut)
    if twin:
        cd, cf = twin
        o["G4.5"].update(twin_Cd=cd, twin_CfVac=cf, dCd=CdB - cd, dCd_exit=mOut / ideal - cd, dCf=CfB - cf, band=1e-4,
                         ok=abs(CdB - cd) <= 1e-4 and abs(CfB - cf) <= 1e-4)
    # G4.6 p0b: A19 unchanged (the flame frame M_u = S_c/c_u, the regression slope's Galilean term)
    orc = freshAt(g, 1.0, cp, q, Sc / math.sqrt(g * TF), rig.p0u, rig.T0u)
    rhobA = orc["b"]["pb"] / orc["b"]["Tb"]
    gal = 0.5 * (rhobA - orc["rhou"]) * sl * sl / rig.p0u
    loss, lossO = p0b / rig.p0u - 1, orc["b"]["p0b"] / rig.p0u - 1
    Ac = m["epsC"] * m["astar"]
    a19 = (mOut - rhobA * sl * Ac) / (orc["m"] * Ac) - 1
    # G4.6 mdot (2.5): the fuel-inventory jump velocity, the lab-frame fresh state
    dF, dM, dOm = D("P_ry") / dt, D("P_mass") / dt, D("Omega") / dt
    Tlab = rig.T0u - uF * uF / (2 * cp)
    plab = rig.p0u * (Tlab / rig.T0u) ** (g / (g - 1))
    rhou = plab / Tlab
    b = rayleighBurnt(g, 1.0, cp, q, Tlab, plab, Sc)
    rhob = b["pb"] / b["Tb"]
    mB = rhou * Sc * Ac
    k = 1 - rhob / rhou
    xF = dF / (rhou * Ac)
    g46 = (mOut - rhob * Ac * xF) / mB - 1
    burn, storage = dOm / mB - 1, -(dM - k * dF) / mB
    o["G4.6"] = dict(p0b_p0u_m1=loss, oracle=lossO, galilean=gal, gap_p0b=loss - lossO - gal, band_p0b=0.1 * abs(lossO),
                     ok_p0b=abs(loss - lossO - gal) <= 0.1 * abs(lossO), mdot=g46, band_mdot=1e-4, ok_mdot=abs(g46) <= 1e-4,
                     burn=burn, storage=storage, ident=g46 - burn - storage, ident_ok=abs(g46 - burn - storage) <= 1e-9,
                     xF=xF, A19_flame_frame=a19, estimator_term=rhob * Ac * (xF - sl) / mB, rho_u_lab=rhou, rho_b_lab=rhob)
    o["spans"] = dict(mdotIn=max(wt["mdotIn"]) - min(wt["mdotIn"]), CfBvac=max(wt["CfBvac"]) - min(wt["CfBvac"]),
                      Sc=max(w["SL"]) - min(w["SL"]))
    o["G4.8"] = dict(shockCells_max=max(w["shockCells"]), ok=max(w["shockCells"]) == 0)
    return o


def ruleWindow(st, mo, tt, ts):
    """2.4 steps 3-5: the ring fit on uFresh over [t_s, tt], k = floor((tt - t_s)/P), W = [tt - k P, tt]."""
    pM, eM = mo["mode1"]["period"], mo["mode1"]["efold"]
    j = [i for i, t in enumerate(st["t"]) if ts - 1e-12 <= t <= tt + 1e-12]
    fit = fitRing([st["t"][i] for i in j], [st["uFresh"][i] for i in j], ts, pM, eM)
    P = fit["period"] if fit["accepted"] else pM
    k = math.floor((tt - ts) / P)
    return fit, P, k


def g412(th, rig, te, mW):
    """G4.12: (a) the first row's T0b vs the design's; (b) the slug's dip in 4-tau ledger bins about t_e."""
    a = th["T0b"][0] / rig.T0b - 1
    dips = []
    s = te - 10
    while s + 4 <= min(te + 10, th["t"][-1]) + 1e-12:
        i, j = snap(th["t"], s), snap(th["t"], s + 4)
        if j > i and s >= th["t"][0]:
            dips.append((s, ((th["I_exit_mass"][j] - th["I_exit_mass"][i]) / (th["t"][j] - th["t"][i])) / mW - 1))
        s += 0.5
    d = min(dips, key=lambda x: x[1]) if dips else (math.nan, math.nan)
    return dict(a=a, band_a=1e-6, ok_a=abs(a) <= 1e-6, T0b_first=th["T0b"][0], T0b_design=rig.T0b, b=d[1], b_at=d[0],
                band_b=3e-4, ok_b=abs(d[1]) <= 3e-4)


def g41(th, q, comp):
    """G4.1 over every row; G4.1f (compensated): max r <= 1e-13 and the last 10% of rows <= 4 x rows 10%-20%."""
    roles = ("in", "exit", "wall", "side")
    n = len(th["t"])
    r = [0.0] * n
    ctrl = 0.0
    for k in ("mass", "mx", "my", "mz", "E", "ry"):
        for i in range(n):
            S = abs(th["P_" + k][i]) + sum(abs(th[f"I_{o}_{k}"][i]) for o in roles)
            if k == "E":
                S += q * abs(th["Omega"][i])
            if k == "ry":
                S += abs(th["Omega"][i])
                if S > 0:
                    ctrl = max(ctrl, abs(th["res_ry"][i] - th["Omega"][i]) / S)
            if S > 0:
                r[i] = max(r[i], abs(th["res_" + k][i]) / S)
    worst = max(r)
    last, early = max(r[int(0.9 * n):]), max(r[int(0.1 * n):max(int(0.2 * n), int(0.1 * n) + 1)])
    return {"G4.1": dict(res=worst, band=1e-12, ok=worst <= 1e-12, control_noOmega=ctrl, control_fails=ctrl >= 1e-6),
            "G4.1f": dict(evaluated=comp, res=worst, band=1e-13, last10=last, rows10_20=early,
                          ratio=last / early if early > 0 else math.inf, ok=worst <= 1e-13 and last <= 4 * early)}


def gates(a):
    """The report (2.4, 2.5): the rule window W (or --from / --window), the legacy windows beside."""
    m = json.load(open(os.path.join(a.dir, "design.json")))
    rig = Rig(m)
    mo = modesOf(a.dir)
    st, th = readCsv(a.prefix + "_stats.csv"), readCsv(a.prefix + "_thrust.csv")
    tt = a.tt or th["t"][-1]
    te = mo["transit"]["t_e"]
    ts = te + 10
    comp = a.compensated or "compensated" in labelOf(a.prefix + "_thrust.csv")
    o = dict(prefix=a.prefix, rows=len(th["t"]), steps=th["step"][-1], tt=tt, compensated=comp)
    fit, P, k = ruleWindow(st, mo, tt, ts)
    rule = dict(t_e=te, t_s=ts, t_flush=mo["transit"]["t_flush"], P_model=mo["mode1"]["period"],
                efold_model=mo["mode1"]["efold"], fit=fit, P=P, P_source="fit" if fit["accepted"] else
                f"model period (fit refused: {fit.get('why', '')})", k=k)
    if a.window:
        W = (a.window[0], a.window[1])
        rule["W_source"] = "--window"
    elif getattr(a, "from") is not None:
        W = (getattr(a, "from"), tt)
        rule["W_source"] = "--from"
    else:
        if k < 1:
            raise SystemExit(f"run too short: tt {tt:g} < t_s + P = {ts + P:.2f}")
        W = (tt - k * P, tt)
        rule["W_source"] = "rule (2.4)"
    rule["W"] = list(W)
    o["rule"] = rule
    twin = None
    if a.twin_cd is not None:
        twin = (a.twin_cd, a.twin_cf)
    elif a.twin:
        tw = readCsv(a.twin + "_thrust.csv")
        i1, i2 = snap(tw["t"], tw["t"][-1] - a.twin_window), len(tw["t"]) - 1
        dtt = tw["t"][i2] - tw["t"][i1]
        g = m["gamma"]
        p0, T0 = 1.0 / g, 1.0 / g  # the --mach 1 reference: p0 = T0 = 1/gamma, R 1
        cd = (tw["I_exit_mass"][i2] - tw["I_exit_mass"][i1]) / dtt / (DEPTH * p0 * math.sqrt(g / T0) * (2 / (g + 1)) ** (
            (g + 1) / (2 * (g - 1))))
        twin = (cd, (tw["I_exit_mx"][i2] - tw["I_exit_mx"][i1]) / dtt / (p0 * DEPTH))
    o["W"] = windowGates(m, rig, st, th, W[0], W[1], a.sl1d, a.p1d, a.t1d, twin)
    o["legacy"] = {f"[{x:g}, {y:g}]": windowGates(m, rig, st, th, x, y, a.sl1d, a.p1d, a.t1d, twin)
                   for x, y in ((20.0, 80.0), (40.0, 80.0)) if th["t"][-1] >= y - 1e-9 and th["t"][0] <= x}
    o.update(g41(th, rig.q, comp))
    o["G4.11"] = dict(fit=fit, P_model=rule["P_model"], efold_model=rule["efold_model"])
    if fit["accepted"]:
        dp, de = fit["period"] / rule["P_model"] - 1, fit["efold"] / rule["efold_model"] - 1
        o["G4.11"].update(dP=dp, defold=de, ok=abs(dp) <= 0.03 and abs(de) <= 0.15 and fit["resid_amp"] <= 0.05)
    else:
        o["G4.11"].update(ok=None, note=f"ring below the fit floor (amplitude {fit.get('amp', math.nan):.3e}): {fit.get('why', '')}")
    mW = o["W"]["G4.5"]["mdotOut"]
    o["G4.12"] = g412(th, rig, te, mW)
    if a.start:
        s = readCsv(a.start + "_stats.csv"), readCsv(a.start + "_thrust.csv")
        sw = windowGates(m, rig, s[0], s[1], W[0], W[1], a.sl1d, a.p1d, a.t1d, None)
        pw = o["W"]
        res = {}
        for key, v, sp in (("mdotIn", (pw["G4.5"]["mdotIn"], sw["G4.5"]["mdotIn"]), pw["spans"]["mdotIn"]),
                           ("CfBvac", (pw["G4.5"]["CfBvac"], sw["G4.5"]["CfBvac"]), pw["spans"]["CfBvac"]),
                           ("Sc", (pw["G4.2"]["Sc"], sw["G4.2"]["Sc"]), pw["spans"]["Sc"])):
            res[key] = dict(primary=v[0], start=v[1], diff=v[1] - v[0], band=2 * sp, ok=abs(v[1] - v[0]) <= 2 * sp)
        res["window"] = sw["window"]
        o["G4.7b"] = res
    return o


def summary(o):
    """The human lines (the JSON carries everything)."""
    L = []
    r = o["rule"]
    f = r["fit"]
    L.append(f"rule: t_e {r['t_e']:.3f}, t_s {r['t_s']:.3f}, t_flush {r['t_flush']:.3f}; model P {r['P_model']:.3f} e-fold "
             f"{r['efold_model']:.2f}; fit " + (f"P {f['period']:.3f} e-fold {f['efold']:.2f} resid/amp {f['resid_amp']:.3f} amp "
                                                f"{f['amp']:.3e}" if 'period' in f else "none") + f" -> P {r['P']:.3f} ({r['P_source']}), "
             f"k {r['k']}; W [{r['W'][0]:.3f}, {r['W'][1]:.3f}] ({r['W_source']})")

    def win(name, w):
        g5, g6 = w["G4.5"], w["G4.6"]
        s = (f"{name} [{w['window'][0]:.3f}, {w['window'][1]:.3f}]: G4.2 {w['G4.2']['gap']:+.2e}; G4.3 {w['G4.3']['gap']:+.2e} "
             f"(regression {w['G4.3']['regression_gap']:+.2e}); G4.4 {w['G4.4']['rel']:+.2e} fuel {w['G4.4']['fuelOut_burnIn']:.1e}; ")
        if "dCd" in g5:
            s += (f"G4.5 dCd {g5['dCd']:+.2e} ({g5['form']}; exit {g5['dCd_exit']:+.2e}) dCf {g5['dCf']:+.2e}; ")
        s += (f"G4.6 p0b {g6['gap_p0b']:+.2e} mdot {g6['mdot']:+.3e} = burn {g6['burn']:+.2e} + storage {g6['storage']:+.2e} "
              f"(ident {g6['ident']:+.1e}; A19 {g6['A19_flame_frame']:+.2e}; regression-estimator term {g6['estimator_term']:+.2e})")
        if g5["form"] == "throat flux":
            s += f"; contraction storage {g5['contraction_storage']:+.2e}"
        return s

    L.append(win("W", o["W"]))
    for k, w in o["legacy"].items():
        L.append(win("legacy", w))
    L.append(f"G4.1 {o['G4.1']['res']:.2e} (ctrl {o['G4.1']['control_noOmega']:.2e}); G4.1f {'evaluated' if o['G4.1f']['evaluated'] else 'reported'}: "
             f"{o['G4.1f']['res']:.2e}, last10/rows10-20 {o['G4.1f']['ratio']:.2f}")
    g11 = o["G4.11"]
    L.append("G4.11 " + (f"dP {g11['dP']:+.3f} de {g11['defold']:+.3f} ok {g11['ok']}" if "dP" in g11 else g11["note"]))
    g = o["G4.12"]
    L.append(f"G4.12 a {g['a']:+.3e} (ok {g['ok_a']}); b dip {g['b']:+.3e} at [{g['b_at']:.1f}, {g['b_at'] + 4:.1f}] (ok {g['ok_b']})")
    if "G4.7b" in o:
        L.append("G4.7b " + "; ".join(f"{k} {v['diff']:+.2e} (band {v['band']:.1e}, ok {v['ok']})" for k, v in o["G4.7b"].items()
                                      if k != "window"))
    return "\n".join(L)


def cmdReport(a):
    o = gates(a)
    txt = json.dumps(o, indent=1, default=float)
    print(summary(o))
    if a.json:
        open(a.json, "w").write(txt)


def cmdDrift(a):
    """combscale S3 (issues.md N20): the ledger residual's drift per step, from PREFIX_thrust.csv. Per row k: slope_k =
    LSQ d res_k / d step, beta_k = slope_k / (u_R |P_k|) (writes' mean rounding bias per step in units of the state's
    unit roundoff, u_R = 2^-24 FP32 / 2^-53 FP64, read off the label), and r_max as G4.1 (first / last 10% printed)."""
    path = a.prefix + "_thrust.csv"
    th, lab = readCsv(path), labelOf(path)
    st = re.search(r", FP(32|64)( compensated)?", lab)
    fp32 = bool(st) and st.group(1) == "32"
    uR = 2.0 ** -24 if fp32 else 2.0 ** -53
    n = len(th["t"])
    o = dict(prefix=a.prefix, precision="FP32" if fp32 else "FP64", compensated=bool(st and st.group(2)), rows=n,
             steps=th["step"][-1], rows_out={})
    for k in ("mass", "mx", "E", "ry"):
        s = slope(th["step"], th["res_" + k])
        P = mean([abs(v) for v in th["P_" + k]])
        o["rows_out"][k] = dict(slope=s, P=P, beta=s / (uR * P) if P > 0 else math.nan, res_last=th["res_" + k][-1])
    q = flameUnits(a.gamma, a.mach, a.sigma)["q"]
    r = g41(th, q, False)["G4.1"]["res"]
    o["r_max"] = r
    print(f"{a.prefix}: {o['precision']}{' compensated' if o['compensated'] else ''}, {n} rows, {o['steps']:.0f} steps")
    for k, v in o["rows_out"].items():
        print(f"  {k:5s} slope {v['slope']:+.4e} /step  |P| {v['P']:.6e}  beta {v['beta']:+.4f}  res_last {v['res_last']:+.4e}")
    print(f"  G4.1 r_max {r:.3e} (q {q:.6g})")
    if a.json:
        open(a.json, "w").write(json.dumps(o, indent=1, default=float))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    s = p.add_subparsers(dest="cmd", required=True)
    o = s.add_parser("oracle")
    o.add_argument("--gamma", type=float, default=1.4)
    o.add_argument("--mach", type=float, default=0.042)
    o.add_argument("--sigma", type=float, default=SIGMA)
    o.add_argument("--eps-c", type=float, required=True)
    o.add_argument("--eps-e", type=float, default=6.25)
    o.add_argument("--p0", type=float, default=1.0, help="Total p0 / Tu")
    o.add_argument("--t0", type=float, default=1.0, help="Total t0 / Tu")
    d = s.add_parser("design")
    d.add_argument("--cells", type=float, required=True)
    d.add_argument("--mach", type=float, default=0.042)
    d.add_argument("--gamma", type=float, default=1.4)
    d.add_argument("--sl", type=float, required=True, help="the 1D S_L,num at the same M, gamma, h, --shock hll")
    d.add_argument("--eps-e", type=float, default=2.0)
    # 1.0, not the open box's sliding 0.98 (masterplan_combnozzle A18): the throat holds the BURNT volume flow, so the
    # flame drifts at sigma (u_d - S_c), 5x the fresh mismatch (0.98: -0.106 S_L, its preheat reached the reference slab).
    d.add_argument("--target", type=float, default=1.0, help="the design's u_u / S_L,num")
    d.add_argument("--pa", type=float, default=0.01, help="riemann p and [thrust] pa, ratio to Tu")
    d.add_argument("--upstream", type=float, default=12.0)
    d.add_argument("--out", required=True)
    v = s.add_parser("argv")
    v.add_argument("dir")
    v.add_argument("--tt", type=float, default=80.0)
    v.add_argument("--upstream", type=float, default=None)
    v.add_argument("--compensated", action="store_true", help="the compensated state (masterplan_combclose R6)")
    md = s.add_parser("modes")
    md.add_argument("dir")
    t = s.add_parser("twin")
    t.add_argument("dir")
    t.add_argument("--p0b", type=float, required=True, help="flame units")
    t.add_argument("--t0b", type=float, required=True, help="flame units")
    t.add_argument("--out", required=True)
    r = s.add_parser("report")
    r.add_argument("dir")
    r.add_argument("prefix")
    r.add_argument("--sl1d", type=float, required=True)
    r.add_argument("--p1d", type=float, required=True)
    r.add_argument("--t1d", type=float, required=True)
    r.add_argument("--tt", type=float, default=None)
    # masterplan_combclose 2.4 (supersedes A23/A24's peak list): W = [tt - k P, tt] from t_s = t_e + 10, k integer ring
    # periods; --from / --window override it (the legacy [20, 80] and [40, 80] are printed beside).
    r.add_argument("--from", type=float, default=None, help="window [T, tt] (overrides the rule)")
    r.add_argument("--window", type=float, nargs=2, default=None, metavar=("A", "B"), help="window [A, B] (overrides)")
    r.add_argument("--twin", default=None)
    r.add_argument("--twin-window", type=float, default=30.0, help="the twin's window, t* before its end")
    r.add_argument("--twin-cd", type=float, default=None, help="the twin's Cd (test_gpu_fv_nozzle reacttwin*)")
    r.add_argument("--twin-cf", type=float, default=None, help="the twin's CfVac")
    r.add_argument("--start", default=None, help="G4.7b: another start's PREFIX, on this run's W")
    r.add_argument("--compensated", action="store_true", help="evaluate G4.1f (else read off the label)")
    r.add_argument("--json", default=None)
    dr = s.add_parser("drift")
    dr.add_argument("prefix")
    dr.add_argument("--gamma", type=float, default=1.4)
    dr.add_argument("--mach", type=float, default=0.042)
    dr.add_argument("--sigma", type=float, default=SIGMA)
    dr.add_argument("--json", default=None)
    a = p.parse_args()
    if a.cmd == "report" and (a.twin_cd is None) != (a.twin_cf is None):
        p.error("--twin-cd and --twin-cf go together")
    {"oracle": cmdOracle, "design": cmdDesign, "argv": cmdArgv, "modes": cmdModes, "twin": cmdTwin, "report": cmdReport,
     "drift": cmdDrift}[a.cmd](a)


if __name__ == "__main__":
    main()
