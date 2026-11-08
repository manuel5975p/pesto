#!/usr/bin/env python3
"""T2, T3 of the nozzle (docs/nozzle.md): rhoCentralFoam time means vs dns, same mesh, gap per level.

  nozzle_ladder.py prep ROOT CASE N      ROOT/CASE_N: mesh + foam case + ROOT/CASE_N.foam.json (simjob --spec)
                                         + ROOT/CASE_N.dns.sh (the dns run, through simjob --gpu)
  nozzle_ladder.py switch DIR RECON T0 T1  fvSchemes -> RECON, startFrom latestTime, fieldAverage over [T0, T1]
  nozzle_ladder.py fork DIR NEW            cp -r, for the Minmod reference
  nozzle_ladder.py report ROOT CASE N...   the gap table, one row per level, orders, the stop rule

CASE: A (exit supersonic), B (p_b/p0 0.6) inviscid; L (laminar Re* 1e4 adiabatic), I (isothermal T_w/T0 0.5), Lw (L on a
5 deg wedge). foam: upwind to t0 from the q1d IC, then vanLeer + vanLeerV (and Minmod) with fieldAverage. dns: hybrid + D4,
rest + ramp, FP64, --foam-write snapshots, averaged over the window. One post-processor reads both.
"""
import json
import math
import os
import re
import shutil
import sys

import numpy as np

import nozzle_cases as nc
import nozzle_mesh as nm

FOAM = nc.FOAM
HERE = os.path.dirname(os.path.abspath(__file__))
ROOTDIR = os.path.dirname(os.path.dirname(HERE))
C0 = math.sqrt(1.4 * 287.1 * 300.0)
REF = 1e4  # Re* = rho0 c0 y* / mu


def muOf(re):
    return (nm.P0 / (nm.RGAS * nm.T0)) * C0 / re


# case: mesh args, foam stage times (upwind end, mean window), dns t* (ramp, end), cap s, np by level
def spec(case, N):
    ny = N // 5
    d = dict(nx=N, ny=ny, contour="nc", co=0.2)
    if case == "A":
        d.update(exit="extrapolate")
    elif case == "B":
        d.update(exit="fixed", pb=0.6)
    elif case in ("L", "I", "Lw"):
        d.update(ny=N * 3 // 10, beta=2.0, exit="extrapolate", wall="noslip", mu=muOf(REF), Pr=0.72)
        if case == "I":
            d.update(twall=0.5)
        if case == "Lw":
            d.update(mode="wedge", dphi=5.0, co=1.0)
    else:
        raise SystemExit("unknown case " + case)
    return d


def prep(root, case, N):
    out = os.path.join(root, f"{case}_{N}")
    a = nc.args(out=out, end=0.6, write=0.6, **spec(case, N))
    astar, col, X, Y = nm.generate(a)
    wedge = a.mode == "wedge"
    # dns: the production exit. A: riemann against a low p (supersonic: P, foam's zeroGradient).
    # Viscous: extrapolate, foam's own row: a riemann row sets p_b on the layer's subsonic faces and moved the
    # wall p 14% in the last cells.
    exitKind, pb = ("extrapolate", 0.05) if a.mu > 0 else ("riemann", 0.05) if a.exit == "extrapolate" else ("fixed", a.pb)
    nm.writeBc(out, astar, wedge, pb, exitKind, a.wall, a.twall, astar)
    viscous = a.mu > 0
    np_ = 1 if N <= 160 else 12
    me = os.path.abspath(__file__)
    py = "/usr/bin/python3"
    steps = []
    dec = lambda c: [{"exe": f"{FOAM}/decomposePar", "args": ["-case", c, "-force"]}]
    if np_ > 1:
        with open(out + "/system/decomposeParDict", "w") as f:
            f.write(nm.FH.format(cls="dictionary", loc="system", obj="decomposeParDict"))
            f.write(f"numberOfSubdomains {np_};\nmethod simple;\ncoeffs {{ n (6 2 1); }}\n")
        steps += dec(out)

    def run(c):
        if np_ <= 1:
            return [{"exe": f"{FOAM}/rhoCentralFoam", "args": ["-case", c]}]
        return [{"exe": "/usr/bin/mpirun", "args": ["-np", str(np_), "--bind-to", "core", "--map-by", "core", f"{FOAM}/rhoCentralFoam", "-case", c, "-parallel"]}]

    def recon(c):
        return [] if np_ <= 1 else [{"exe": f"{FOAM}/reconstructPar", "args": ["-case", c, "-latestTime"]}]

    # Viscous: L's t 0.4 is 139 t*, the mean window a tenth of a second; Lw, I t 0.25, the budget's.
    tu, t0, t1 = (0.25, 0.3, 0.4) if case == "L" else (0.15, 0.18, 0.25) if viscous else (0.6, 0.65, 0.9)
    if viscous:
        s = open(out + "/system/controlDict").read()
        open(out + "/system/controlDict", "w").write(re.sub(r"endTime [0-9.e-]+;", f"endTime {tu};", s).replace("writeInterval 0.6;", f"writeInterval {tu};"))
    steps += run(out)
    if N <= 320 and not viscous:
        steps.append({"exe": py, "args": [me, "fork", out, out + "_M"]})
    # The 5 deg wedge's vanLeer stage dies at maxCo 1 (sigFpe at its first steps); foam's Co counts the sides.
    co = ["0.3"] if case == "Lw" else []
    steps.append({"exe": py, "args": [me, "switch", out, "vanLeer", str(t0), str(t1)] + co})
    steps += run(out) + recon(out)
    if N <= 320 and not viscous:
        steps.append({"exe": py, "args": [me, "switch", out + "_M", "Minmod", str(t0), str(t1)]})
        steps += run(out + "_M") + recon(out + "_M")
    with open(out + ".foam.json", "w") as f:
        json.dump({"label": f"nozzle-{case}{N}", "resource": "cpu", "max_runtime_s": 1800, "max_mem_mb": 16000, "cwd": root, "steps": steps}, f, indent=1)
    # dns
    ramp, tt, snaps = 40.0, 260.0, 20
    if viscous:
        ramp, tt = 30.0, 110.0
    dtw = (tt / 3) / snaps
    inviscid = "--inviscid" if a.mu == 0 else f"--re {REF:g} --prandtl {a.Pr}"
    # --foam-write writes every DT from t = DT: keep the snapshots past the window's start.
    args = (f"flow --physics compressible --polymesh {out} --bc {out}/bc.toml --shock hll {inviscid} --mach 1 --ramp {ramp} --tt {tt} "
            f"--fv-precision fp64 --foam-write {dtw:.12g} --emit stats --out {out}/dnsrun")
    with open(out + ".dns.args", "w") as f:
        f.write(args + "\n")
    with open(out + ".meta.json", "w") as f:
        json.dump(dict(nx=a.nx, ny=a.ny, wedge=wedge, astar=astar, col=col, pb=a.pb, exit=a.exit, tDns=ramp + tt, win=tt / 3, mu=a.mu, twall=a.twall), f)
    print(f"{case} {N}: {a.nx}x{a.ny}, np {np_}, A* {astar:.10g}; foam spec {out}.foam.json; dns args {out}.dns.args")


def switch(d, recon, t0, t1, co=None):
    rec = {"vanLeer": ("vanLeer", "vanLeerV", "vanLeer"), "Minmod": ("Minmod", "MinmodV", "Minmod")}[recon]
    p = d + "/system/fvSchemes"
    s = open(p).read()
    s = re.sub(r"reconstruct\(rho\) \w+; reconstruct\(U\) \w+; reconstruct\(T\) \w+;", f"reconstruct(rho) {rec[0]}; reconstruct(U) {rec[1]}; reconstruct(T) {rec[2]};", s)
    open(p, "w").write(s)
    p = d + "/system/controlDict"
    s = open(p).read()
    s = re.sub(r"startFrom \w+;", "startFrom latestTime;", s)
    if co:
        s = re.sub(r"maxCo [0-9.]+;", f"maxCo {co};", s)
    s = re.sub(r"endTime [0-9.e-]+;", f"endTime {t1};", s)
    s = re.sub(r"writeInterval [0-9.e-]+;", f"writeInterval {t1 - t0};", s)
    s += ("functions { avg { type fieldAverage; libs (fieldFunctionObjects); writeControl writeTime; timeStart %g; "
          "fields ( p { mean on; prime2Mean off; base time; } T { mean on; prime2Mean off; base time; } U { mean on; prime2Mean off; base time; } ); } }\n" % t0)
    open(p, "w").write(s)


def fork(d, new):
    if os.path.exists(new):
        shutil.rmtree(new)
    shutil.copytree(d, new)


def rdField(path, nC):
    t = open(path).read()
    m = re.search(r"internalField\s+nonuniform\s+List<(\w+)>\s*(\d+)\s*\(", t)
    n = int(m.group(2))
    b = t[m.end():]
    if m.group(1) == "scalar":
        return np.array(b.split(None, n)[:n], float)
    v = re.findall(r"\(([^()]*)\)", b)[:n]
    return np.array([[float(x) for x in a.split()] for a in v])


def lastTime(d):
    ts = [x for x in os.listdir(d) if re.fullmatch(r"[0-9.e-]+", x) and x != "0"]
    return max(ts, key=float)


def foamMean(d, nC):
    t = lastTime(d)
    return rdField(f"{d}/{t}/pMean", nC), rdField(f"{d}/{t}/TMean", nC), rdField(f"{d}/{t}/UMean", nC), t


def dnsMean(d, nC, tDns, win):
    """Mean of the --foam-write snapshots in the window, SI: box p, T, u scale by p0, T0, c0."""
    root = d + "/dnsrun_foam"
    ts = sorted([x for x in os.listdir(root) if float(x) >= tDns - win - 1e-9], key=float)
    p, T, U = 0, 0, 0
    for x in ts:
        p = p + rdField(f"{root}/{x}/p", nC)
        T = T + rdField(f"{root}/{x}/T", nC)
        U = U + rdField(f"{root}/{x}/U", nC)
    n = len(ts)
    g = 1.4
    return p / n * g * nm.P0, T / n * g * nm.T0, U / n * C0, n  # p_inf = 1/g, T_inf = 1/g in box units


def geom(d, meta):
    z = np.load(d + "/mesh.npz")
    X, Y = z["X"], z["Y"]
    return X, Y


def post(p, T, U, meta, X, Y):
    """Quantities on cell-order fields, SI."""
    nx, ny = meta["nx"], meta["ny"]
    R, g = nm.RGAS, 1.4
    pc = p.reshape(ny, nx)
    Tc = T.reshape(ny, nx)
    Ux = U[:, 0].reshape(ny, nx)
    Uy = U[:, 1].reshape(ny, nx)
    M = np.hypot(Ux, Uy) / np.sqrt(g * R * Tc)
    rho = pc / (R * Tc)
    # the exit column: S_x per cell
    y = Y[-1, :]
    if meta["wedge"]:
        tanh = meta["astar"] / Y[:, -1].min() ** 2
        Sx = (y[1:] ** 2 - y[:-1] ** 2) * tanh
    else:
        Sx = (y[1:] - y[:-1]) * 0.1
    ue, re, pe = Ux[:, -1], rho[:, -1], pc[:, -1]
    if meta["exit"] == "fixed":
        pe = meta["pb"] * nm.P0 * np.ones_like(pe)
        re = pe / (R * Tc[:, -1])
    mdot = (re * ue * Sx).sum()
    Fex = ((re * ue * ue + pe) * Sx).sum()
    # The wall row: friction and heat by the same one-sided read in both solvers.
    dn = Y[:-1, -1] * 0.5 + Y[1:, -1] * 0.5 - 0.25 * (Y[:-1, -2] + Y[1:, -2] + Y[:-1, -1] + Y[1:, -1])
    tau = meta["mu"] * np.hypot(Ux[-1, :], Uy[-1, :]) / dn
    qw = Tc[-1, :] / dn
    # delta99 at the exit, in cells: rows whose speed is below 0.99 of the core's.
    sp = np.hypot(Ux[:, -1], Uy[:, -1])
    cells = int((sp < 0.99 * sp[:ny // 2].max()).sum() - (sp[:ny // 2] < 0.99 * sp[:ny // 2].max()).sum())
    return dict(pc=pc, M=M, mdot=mdot, Cf=Fex / (nm.P0 * meta["astar"]), Cd=mdot / idealMdot(meta["astar"]), axisM=M[0, -1], wall=pc[-1, :], axisP=pc[0, :], tau=tau, Twall1=Tc[-1, :], dn=dn, d99=cells)


def idealMdot(astar):
    g = 1.4
    return astar * nm.P0 * math.sqrt(g / (nm.RGAS * nm.T0)) * (2 / (g + 1)) ** ((g + 1) / (2 * (g - 1)))


def shockX(q, xc):
    """The axis shock: largest p rise past the throat, in x."""
    a = q["axisP"]
    i0 = int(np.searchsorted(xc, 0.0))
    i = i0 + int(np.argmax(np.diff(a[i0:])))
    # Sub-cell: where p crosses the middle of the jump, four cells either side.
    lo, hi = max(i - 4, 0), min(i + 5, len(a) - 1)
    mid = 0.5 * (a[lo] + a[hi])
    for k in range(lo, hi):
        if a[k] <= mid <= a[k + 1]:
            return xc[k] + (mid - a[k]) / (a[k + 1] - a[k]) * (xc[k + 1] - xc[k]), xc[1] - xc[0]
    return 0.5 * (xc[i] + xc[i + 1]), xc[1] - xc[0]


def vols(X, Y, meta):
    """Cell weights (ny, nx): planar quad area, wedge area x centroid radius."""
    dx = X[1:, :-1] - X[:-1, :-1]
    h = 0.5 * ((Y[:-1, 1:] - Y[:-1, :-1]) + (Y[1:, 1:] - Y[1:, :-1]))
    w = (dx * h).T
    if meta["wedge"]:
        r = 0.25 * (Y[:-1, :-1] + Y[1:, :-1] + Y[:-1, 1:] + Y[1:, 1:]).T
        w = w * r
    return w


def report(root, case, levels):
    rows = []
    for N in levels:
        d = f"{root}/{case}_{N}"
        meta = json.load(open(d + ".meta.json"))
        nC = meta["nx"] * meta["ny"]
        X, Y = geom(d, meta)
        pf, Tf, Uf, tf = foamMean(d, nC)
        pd, Td, Ud, ns = dnsMean(d, nC, meta["tDns"], meta["win"])
        a, b = post(pf, Tf, Uf, meta, X, Y), post(pd, Td, Ud, meta, X, Y)
        w = vols(X, Y, meta)
        w = w / w.sum()
        L1p = float((np.abs(pd / pf - 1) * w.ravel()).sum())
        wall = float(np.abs(b["wall"] / a["wall"] - 1).mean())
        r = dict(N=N, snaps=ns, L1p=L1p, wall=wall, axisM=abs(b["axisM"] - a["axisM"]), mdot=abs(b["mdot"] / a["mdot"] - 1), Cf=abs(b["Cf"] / a["Cf"] - 1), Cd=abs(b["Cd"] - a["Cd"]),
                 CdF=a["Cd"], CdD=b["Cd"], CfF=a["Cf"], CfD=b["Cf"], MF=a["axisM"], MD=b["axisM"])
        xc = 0.5 * (X[1:, 0] + X[:-1, 0])
        if meta["mu"] > 0:
            r.update(cf=float(np.abs(b["tau"] / np.maximum(a["tau"], 1e-300) - 1).mean()), d99=b["d99"], d99F=a["d99"], tauF=float(a["tau"].mean()), tauD=float(b["tau"].mean()))
            print(f"  wall shear: mean gap {r['cf']:.3e}, mean tau {r['tauD']:.4f}/{r['tauF']:.4f}; delta99 {b['d99']}/{a['d99']} cells")
            if meta.get("twall"):
                Tw = meta["twall"] * nm.T0
                qa, qb = (a["Twall1"] - Tw) / a["dn"], (b["Twall1"] - Tw) / b["dn"]
                r.update(q=float(np.abs(qb / qa - 1).mean()))
                print(f"  wall heat: mean gap {r['q']:.3e}")
        if meta["pb"] > 0 and not meta["mu"] > 0:
            xa, h = shockX(a, xc)
            xb, _ = shockX(b, xc)
            r.update(xsF=xa, xsD=xb, xs=abs(xb - xa) / h)
        if os.path.isdir(d + "_M") and os.path.isdir(f"{d}_M/{lastTime(d + '_M')}"):
            pm, Tm, Um, _ = foamMean(d + "_M", nC)
            m = post(pm, Tm, Um, meta, X, Y)
            r.update(sprL1=float((np.abs(pm / pf - 1) * w.ravel()).sum()), sprWall=float(np.abs(m["wall"] / a["wall"] - 1).mean()), sprM=abs(m["axisM"] - a["axisM"]),
                     sprMdot=abs(m["mdot"] / a["mdot"] - 1), sprCf=abs(m["Cf"] / a["Cf"] - 1))
        rows.append(r)
        print(f"{case} {N}: L1 p {L1p:.3e} wall {wall:.3e} axis M {r['axisM']:.3e} ({r['MD']:.4f} / {r['MF']:.4f}) mdot {r['mdot']:.3e} Cf {r['Cf']:.3e} Cd {r['CdD']:.5f}/{r['CdF']:.5f} snaps {ns}"
              + (f" x_s {r['xs']:.2f} h ({r['xsD']:.4f} / {r['xsF']:.4f})" if "xs" in r else "")
              + (f" | vL vs Mm: L1 {r['sprL1']:.3e} wall {r['sprWall']:.3e} M {r['sprM']:.3e} mdot {r['sprMdot']:.3e} Cf {r['sprCf']:.3e}" if "sprL1" in r else ""))
    for i in range(1, len(rows)):
        A, B = rows[i - 1], rows[i]
        print(f"  {A['N']}->{B['N']}: order L1 p {math.log2(A['L1p'] / B['L1p']):.2f}, wall {math.log2(A['wall'] / B['wall']):.2f}, ratio wall {B['wall'] / A['wall']:.2f}, Cd {B['Cd'] / max(A['Cd'], 1e-300):.2f}, Cf {B['Cf'] / max(A['Cf'], 1e-300):.2f}")
    json.dump(rows, open(f"{root}/{case}.report.json", "w"), indent=1)


def main():
    c = sys.argv[1]
    if c == "prep":
        prep(os.path.abspath(sys.argv[2]), sys.argv[3], int(sys.argv[4]))
    elif c == "switch":
        switch(sys.argv[2], sys.argv[3], float(sys.argv[4]), float(sys.argv[5]), float(sys.argv[6]) if len(sys.argv) > 6 else None)
    elif c == "fork":
        fork(sys.argv[2], sys.argv[3])
    elif c == "report":
        report(os.path.abspath(sys.argv[2]), sys.argv[3], [int(x) for x in sys.argv[4:]])
    else:
        raise SystemExit(__doc__)


if __name__ == "__main__":
    main()
