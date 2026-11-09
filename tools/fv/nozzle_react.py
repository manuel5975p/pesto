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
  nozzle_react.py argv DIR [flags]   the dns flame argv of a design (for simjob)
  nozzle_react.py report DIR PREFIX [--twin TPREFIX] [--start SPREFIX] --sl1d S --p1d P --t1d T [--json OUT]
      the G4 gates (masterplan_combnozzle P4) off PREFIX_stats.csv and PREFIX_thrust.csv over [tt/2, tt]; the 1D
      reference: S_L,num and its fresh state (window means of the 1D --shock hll run at the same M, gamma, h)

q is the SOLVER's heat release cp (sigma - 1) Tu (masterplan_combnozzle R5): T0b = T0u + (sigma - 1) Tu, never sigma T0u.
Python twin of dns/fvnozzle fvReactingNozzleDesign (test_fvref react pins the two to 1e-12).
"""
import argparse
import json
import math
import os
import sys

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


# --- report: the G4 gates ------------------------------------------------------------------------------------------

A_T, A_P = 2.33532, -0.5  # Oracle.sens() (masterplan_combnozzle R3, A6 G1.4)


def readCsv(path):
    import csv
    with open(path) as f:
        rows = list(csv.DictReader(l for l in f if not l.startswith("#")))
    return {k: [float(r[k]) for r in rows] for k in rows[0]}


def window(c, t0, t1):
    idx = [i for i, t in enumerate(c["t"]) if t0 <= t <= t1 + 1e-12]
    return {k: [v[i] for i in idx] for k, v in c.items()}


def mean(v):
    return sum(v) / len(v)


def slope(t, x):
    tm, xm = mean(t), mean(x)
    return sum((a - tm) * (b - xm) for a, b in zip(t, x)) / sum((a - tm) ** 2 for a in t)


def gates(design, prefix, sl1d, p1d, t1d, tt=None):
    m = json.load(open(os.path.join(design, "design.json")))
    g, M = m["gamma"], m["mach"]
    u = flameUnits(g, M)
    Tu, cp, q = u["Tu"], u["cp"], u["q"]
    st, th = readCsv(prefix + "_stats.csv"), readCsv(prefix + "_thrust.csv")
    tt = tt or st["t"][-1]
    w, wt = window(st, tt / 2, tt), window(th, tt / 2, tt)
    o = dict(rows=len(st["t"]), steps=th["step"][-1], window=[wt["t"][0], wt["t"][-1]], tt=tt)
    # G4.1: the ledger per row, S' = |P_k| + sum |I_k| (+ q |Omega| on E, |Omega| on rho Y).
    roles = ("in", "exit", "wall", "side")
    worst, ctrl = 0.0, 0.0
    for k in ("mass", "mx", "my", "mz", "E", "ry"):
        for i in range(len(th["t"])):
            S = abs(th["P_" + k][i]) + sum(abs(th[f"I_{r}_{k}"][i]) for r in roles)
            if k == "E":
                S += q * abs(th["Omega"][i])
            if k == "ry":
                S += abs(th["Omega"][i])
            if S > 0:
                worst = max(worst, abs(th["res_" + k][i]) / S)
            if k == "ry" and S > 0:
                ctrl = max(ctrl, abs(th["res_ry"][i] - th["Omega"][i]) / S)
    o["G4.1"] = dict(res=worst, band=1e-12, ok=worst <= 1e-12, control_noOmega=ctrl, control_fails=ctrl >= 1e-6)
    # G4.2: S_c (A_ref) vs the 1D S_L,num, fresh-state corrected (R3).
    Sc, uF, pF, TF = mean(w["SL"]), mean(w["uFresh"]), mean(w["pFresh"]), mean(w["TFresh"])
    dS = A_T * (TF - t1d) / t1d + A_P * (pF - p1d) / p1d
    band = 5e-4
    corr = abs(dS) > band / 4
    gap = Sc - sl1d * (1 + dS) if corr else Sc - sl1d
    o["G4.2"] = dict(Sc=Sc, sl1d=sl1d, dS_fresh=dS, corrected=corr, gap=gap, band=band, ok=abs(gap) <= band,
                     Sc_span=max(w["SL"]) - min(w["SL"]))
    # G4.3: x_f's regression slope vs u_u - S_c; control: the shipped vol/L area.
    sl = slope(w["t"], w["xf"])
    kin = sl - (uF - Sc)
    o["G4.3"] = dict(dxf_dt=sl, uu=uF, uu_minus_Sc=uF - Sc, gap=kin, band=2e-3 * sl1d, ok=abs(kin) <= 2e-3 * sl1d)
    # G4.4: T0b vs T0u + (sigma - 1) Tu; fuelOut / burnIn over the window (the ledger).
    T0u, p0u = m["t0"] * Tu, m["p0"] * Tu
    T0b, p0b = mean(wt["T0b"]), mean(wt["p0b"])
    T0bO = T0u + (SIGMA - 1) * Tu
    i1, i2 = th["t"].index(wt["t"][0]), th["t"].index(wt["t"][-1])
    dOm = th["Omega"][i2] - th["Omega"][i1]
    fuelOut = (th["I_exit_ry"][i2] - th["I_exit_ry"][i1]) / dOm
    o["G4.4"] = dict(T0b=T0b, T0b_oracle=T0bO, rel=T0b / T0bO - 1, band=1e-4, fuelOut_burnIn=fuelOut, band_fuel=1e-8,
                     ok=abs(T0b / T0bO - 1) <= 1e-4 and abs(fuelOut) <= 1e-8, control_sigmaT0u=SIGMA * T0u / T0b - 1,
                     T0b_span=max(wt["T0b"]) - min(wt["T0b"]))
    # Window means off the ledger.
    dt = th["t"][i2] - th["t"][i1]
    mIn = -(th["I_in_mass"][i2] - th["I_in_mass"][i1]) / dt
    mOut = (th["I_exit_mass"][i2] - th["I_exit_mass"][i1]) / dt
    Tvac = (th["I_exit_mx"][i2] - th["I_exit_mx"][i1]) / dt
    ideal = lambda P0, T0: m["astar"] * P0 * math.sqrt(g / T0) * (2 / (g + 1)) ** ((g + 1) / (2 * (g - 1)))
    o["window"] = dict(mdotIn=mIn, mdotOut=mOut, out_in=mOut / mIn - 1, Tvac=Tvac, p0b=p0b, T0b=T0b, CdB=mOut / ideal(p0b, T0b),
                       CfBvac=Tvac / (p0b * m["astar"]), CdB_rows=mean(wt["CdB"]), CfBvac_rows=mean(wt["CfBvac"]),
                       Cd_T0u=mOut / ideal(p0b, T0u), exitSuper_min=min(wt["exitSuper"]),
                       p_fresh_rms=(mean([(x - pF) ** 2 for x in w["pFresh"]])) ** 0.5 / pF,
                       u_fresh_span=max(w["uFresh"]) - min(w["uFresh"]), Pmy_over_Pmx=max(abs(a / b) for a, b in zip(th["P_my"], th["P_mx"])))
    # G4.6: composition at the measured S_c (the flame-frame fresh speed: M_u = S_c / c_u). A drifting flame (x_f' = sl)
    # adds the Galilean term (rho_b - rho_u) x_f'^2 / 2 to p0b - p0u, and its burnt side carries rho_b x_f' A_c past the
    # station on top of the burning rate rho_u S_c A_c (masterplan_combnozzle A19); the plan-literal stationary numbers
    # are kept beside them.
    Mu = Sc / math.sqrt(g * TF)
    orc = freshAt(g, 1.0, cp, q, Mu, p0u, T0u)
    rhob = orc["b"]["pb"] / orc["b"]["Tb"]
    gal = 0.5 * (rhob - orc["rhou"]) * sl * sl / p0u
    loss, lossO = p0b / p0u - 1, orc["b"]["p0b"] / p0u - 1
    Ac = m["epsC"] * m["astar"]
    mBurn = orc["m"] * Ac
    mFrame = mOut - rhob * sl * Ac
    o["G4.6"] = dict(Mu=Mu, p0b_p0u_m1=loss, oracle=lossO, galilean=gal, gap=loss - lossO - gal, gap_stationary=loss - lossO,
                     band=0.1 * abs(lossO), mdot_out=mOut, mdot_burn_oracle=mBurn, mdot_rel=mFrame / mBurn - 1,
                     mdot_rel_stationary=mOut / mBurn - 1, band_mdot=1e-4,
                     ok=abs(loss - lossO - gal) <= 0.1 * abs(lossO) and abs(mFrame / mBurn - 1) <= 1e-4,
                     control_noHeat=abs(loss), control_fails=abs(loss) > 0.1 * abs(lossO))
    # G4.8: the sensor.
    o["G4.8"] = dict(shockCells_max=max(st["shockCells"]), ok=max(st["shockCells"]) == 0)
    o["series"] = dict(Sc=w["SL"], xf=w["xf"], t=w["t"])
    o["spans"] = {k: max(wt[k]) - min(wt[k]) for k in ("mdotIn", "mdotOut", "CfBvac", "CdB", "T0b", "p0b")}
    return o


def cmdReport(a):
    o = gates(a.dir, a.prefix, a.sl1d, a.p1d, a.t1d, a.tt)
    ser = o.pop("series")
    if a.twin:
        tw = readCsv(a.twin + "_thrust.csv")
        tt = tw["t"][-1]
        t1 = tt - a.twin_window
        i1 = min(i for i, t in enumerate(tw["t"]) if t >= t1)
        i2 = len(tw["t"]) - 1
        dt = tw["t"][i2] - tw["t"][i1]
        m = json.load(open(os.path.join(a.dir, "design.json")))
        g = m["gamma"]
        mOut = (tw["I_exit_mass"][i2] - tw["I_exit_mass"][i1]) / dt
        Tv = (tw["I_exit_mx"][i2] - tw["I_exit_mx"][i1]) / dt
        p0, T0 = 1.0 / g, 1.0 / g  # the --mach 1 reference: p0 = T0 = 1/gamma, R 1
        Cd = mOut / (DEPTH * p0 * math.sqrt(g / T0) * (2 / (g + 1)) ** ((g + 1) / (2 * (g - 1))))
        Cf = Tv / (p0 * DEPTH)
        wd = o["window"]
        o["G4.5"] = dict(twin_Cd=Cd, twin_CfVac=Cf, CdB=wd["CdB"], CfBvac=wd["CfBvac"], dCd=wd["CdB"] - Cd, dCf=wd["CfBvac"] - Cf,
                         band=1e-4, ok=abs(wd["CdB"] - Cd) <= 1e-4 and abs(wd["CfBvac"] - Cf) <= 1e-4,
                         control_Cd_T0u=wd["Cd_T0u"] - Cd, twin_window=[tw["t"][i1], tw["t"][i2]],
                         twin_viscous_loss_Cd=Cd - 1, twin_viscous_loss_Cf=Cf - cfIdeal(g, m["epsE"]))
    if a.start:
        s = gates(a.dir, a.start, a.sl1d, a.p1d, a.t1d, a.tt)
        # The band: 2 x the primary's window span of the per-row series (mdotIn, CfBvac: _thrust.csv; S_c: _stats.csv).
        res = {}
        for k, v, sp in (("mdot", (o["window"]["mdotIn"], s["window"]["mdotIn"]), o["spans"]["mdotIn"]),
                         ("CfBvac", (o["window"]["CfBvac"], s["window"]["CfBvac"]), o["spans"]["CfBvac"]),
                         ("Sc", (o["G4.2"]["Sc"], s["G4.2"]["Sc"]), o["G4.2"]["Sc_span"])):
            res[k] = dict(primary=v[0], start=v[1], diff=v[1] - v[0], band=2 * sp, ok=abs(v[1] - v[0]) <= 2 * sp)
        res["xf_offset_end"] = s["series"]["xf"][-1] - ser["xf"][-1]
        o["G4.7"] = res
    txt = json.dumps(o, indent=1)
    print(txt)
    if a.json:
        open(a.json, "w").write(txt)


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
    r.add_argument("--twin", default=None)
    r.add_argument("--twin-window", type=float, default=30.0, help="the twin's window, t* before its end")
    r.add_argument("--start", default=None)
    r.add_argument("--json", default=None)
    a = p.parse_args()
    {"oracle": cmdOracle, "design": cmdDesign, "argv": cmdArgv, "twin": cmdTwin, "report": cmdReport}[a.cmd](a)


if __name__ == "__main__":
    main()
