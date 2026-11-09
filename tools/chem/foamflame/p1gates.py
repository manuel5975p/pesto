"""masterplan_combnozzle P1 gates G1.0-G1.7 from the compare.json of the poly1d
cases (fv1d, fv1d_stockle, fv1d_g12) and G1's blockMesh foam run. Bands are
the plan's, fixed; every control is evaluated and must FAIL its band.

  foam_flame.py gates      -> runs/flames/foam/fv1d/gates.{txt,json}
"""
import json
import math
import os

import numpy as np

from . import cases, observe

# docs/fvcomp.md "Species cross-verification", FV(FP64) vs oracle (the box,
# fromCurvMesh): S_L at 7.5 / 15 / 30 and the window spans (G1.1's regression)
DOCS_FV64 = {7.5: (1.016873, 2.9e-4), 15: (1.003472, 8.0e-5), 30: (1.000437, 2.0e-5)}
G10_BAND = 2e-5                       # G1.0: the window span
G12_CONTROL = 0.1                     # G1.2's control: stock Le gap >= 0.1
ORDER_MIN = 1.8                       # G1.1, G1.5, G1.6
G17_D, G17_TOL = 4.8e-5, 1.2e-5       # G1.7: 4.8e-5 +- 1.2e-5 [D]


def sref(gamma, mach, sigma):
    """The O(M^2) offset of the low-Mach oracle: 1 - gamma M^2 (sigma - 1)."""
    return 1.0 - gamma * mach * mach * (sigma - 1.0)


def _load(root, case):
    with open(os.path.join(root, case, "compare.json")) as f:
        return json.load(f)


def _sl(r):
    return r["SL_series"]["mean"]


def _order(e):
    """Orders of a 3-rung error ladder (ratio 2): log2(e0/e1), log2(e1/e2)."""
    return [math.log2(abs(e[k] / e[k + 1])) if e[k] * e[k + 1] > 0 else float("nan")
            for k in range(2)]


def sens(root, oracle):
    """Oracle.sens() cached next to the oracle (two eigenvalue solves)."""
    path = os.path.join(root, f"oracle_sens_s{oracle.sigma:g}_b{oracle.beta:g}.json")
    if os.path.exists(path):
        with open(path) as f:
            d = json.load(f)
        return d["aT"], d["ap"]
    aT, ap = oracle.sens()
    with open(path, "w") as f:
        json.dump({"aT": aT, "ap": ap, "eps": 1e-4}, f, indent=1)
    return aT, ap


def g1_window(root, si, oracle):
    """G1's blockMesh foam run at 15 cells, re-read over P1's window [7, 14]."""
    spec = {**cases.CASES["G1"], "window": (7.0, 14.0)}
    m = cases.mesh_1d(spec, 15, oracle.dth)
    run = observe.load_foam(os.path.join(root, "G1", "foam_mclin_c15"), si, m, spec["species"])
    return observe._window(run["series"], spec["window"])


def evaluate(root, oracle, si):
    fv, st, g12 = _load(root, "fv1d"), _load(root, "fv1d_stockle"), _load(root, "fv1d_g12")
    R, Rg = fv["runs"], g12["runs"]
    sig, M = 5.0, 0.01
    s14, s12 = sref(1.4, M, sig), sref(1.2, M, sig)
    lad = [7.5, 15, 30]
    out, L = {}, []

    def line(gid, ok, text):
        out[gid] = {"pass": bool(ok), "text": text}
        L.append(f"{gid:6s} {'PASS' if ok else 'UNMET'}  {text}")

    # G1.0: foam poly1d vs foam G1 blockMesh, 15 cells, the same column (G1's)
    g1 = g1_window(root, si, oracle)
    p = R["foam_mclin_c15"]["SL_g1_series"]
    d0 = p["mean"] - g1["mean"]
    line("G1.0", abs(d0) <= G10_BAND,
         f"S_L(poly, G1 column) {p['mean']:.8f} - S_L(blockMesh) {g1['mean']:.8f} = {d0:+.3e} "
         f"(band {G10_BAND:g}; spans {p['span']:.1e} / {g1['span']:.1e})")
    c = R.get("foam_mclin_icwrong_c15") or fv.get("series_only", {}).get("foam_mclin_icwrong_c15")
    dc = (c["SL_g1_series"]["mean"] - g1["mean"]) if c else float("nan")
    out["G1.0 control"] = {"fails": not (abs(dc) <= G10_BAND), "value": dc}
    L.append(f"       control (IC j-fastest): {dc:+.3e} {'fails the band: live' if not abs(dc) <= G10_BAND else 'PASSES: dead control'}"
             + ("" if c else " (run absent or diverged)"))

    # G1.1: dns FV FP64 ladder about the O(M^2) offset; the docs' box rows
    e = [_sl(R[f"dns_c{N:g}"]) - s14 for N in lad]
    o = _order(e)
    reg = [abs(_sl(R[f"dns_c{N:g}"]) - DOCS_FV64[N][0]) <= max(DOCS_FV64[N][1], R[f"dns_c{N:g}"]["SL_series"]["span"])
           for N in lad]
    line("G1.1", min(o) >= ORDER_MIN and all(reg),
         f"S_L - S_ref {e[0]:+.3e} {e[1]:+.3e} {e[2]:+.3e}, orders {o[0]:.3f} {o[1]:.3f}; docs box "
         + " ".join(f"{_sl(R[f'dns_c{N:g}']) - DOCS_FV64[N][0]:+.1e}" for N in lad)
         + f" (within spans: {reg})")

    # G1.4 first (it decides G1.2's form): fresh states at the reference slab
    aT, ap = sens(root, oracle)
    Tu, pu = 1.0 / (1.4 * M * M), 1.0 / (1.4 * M * M)
    fresh, band = {}, {}
    for N in lad:
        dn, fn = R[f"dns_c{N:g}"], R[f"foam_mclin_c{N:g}"]
        dT = dn["TRef_series"]["mean"] - fn["TRef_series"]["mean"]
        dp = dn["pRef_series"]["mean"] - fn["pRef_series"]["mean"]
        fresh[N] = aT * dT / Tu + ap * dp / pu
        band[N] = max(abs(_sl(dn) - s14), abs(_sl(fn) - s14))
    corr = any(abs(fresh[N]) > band[N] / 4 for N in lad)
    line("G1.4", not corr,
         f"a_T {aT:.5f} a_p {ap:.5f}; dS_fresh " + " ".join(f"{fresh[N]:+.2e}" for N in lad)
         + " vs band/4 " + " ".join(f"{band[N] / 4:.2e}" for N in lad)
         + (" -> G1.2 on the fresh-state-corrected S_L" if corr else " -> raw S_L"))

    # G1.2: dns vs foam at equal mesh, shrinking; control stock Le
    gap = {N: _sl(R[f"dns_c{N:g}"]) - _sl(R[f"foam_mclin_c{N:g}"]) - (fresh[N] if corr else 0.0) for N in lad}
    ok = all(abs(gap[N]) <= band[N] for N in lad) and abs(gap[30]) < abs(gap[7.5])
    line("G1.2", ok, "dns - foam " + " ".join(f"{gap[N]:+.3e}" for N in lad) + " vs max own error "
         + " ".join(f"{band[N]:.3e}" for N in lad) + f"; |gap 30| < |gap 7.5|: {abs(gap[30]) < abs(gap[7.5])}")
    sc = _sl(st["runs"]["foam_mclin_c15"]) - _sl(R["dns_c15"])
    out["G1.2 control"] = {"fails": abs(sc) >= G12_CONTROL, "value": sc}
    L.append(f"       control (stock Le 1.39, foam 15): gap {sc:+.4f} "
             f"{'>= 0.1: fails the band, live' if abs(sc) >= G12_CONTROL else '< 0.1: dead control'}")

    # G1.3: profiles aligned at x_f, dns vs foam <= max own oracle Linf
    ok3, t3 = True, []
    for N in lad:
        pr = fv["pairs"][f"dns_c{N:g}~foam_mclin_c{N:g}"]["linf"]
        own = {q: max(R[f"dns_c{N:g}"]["linf_oracle"][q], R[f"foam_mclin_c{N:g}"]["linf_oracle"][q])
               for q in ("theta", "Y")}
        ok3 &= all(pr[q] <= own[q] for q in own)
        t3.append(f"{N:g}: th {pr['theta']:.2e}/{own['theta']:.2e} Y {pr['Y']:.2e}/{own['Y']:.2e}")
    line("G1.3", ok3, "; ".join(t3))

    # G1.5: M_n self-convergence, Richardson reported
    sm = [_sl(R[f"dns_c{N:g}_mn"]) for N in lad]
    r5 = (sm[0] - sm[1]) / (sm[1] - sm[2])
    p5 = math.log2(r5) if r5 > 0 else float("nan")
    rich = sm[2] + (sm[2] - sm[1]) / (2 ** p5 - 1) if p5 == p5 else float("nan")
    line("G1.5", p5 >= ORDER_MIN,
         f"M 0.042: S_L {sm[0]:.6f} {sm[1]:.6f} {sm[2]:.6f}, ratio {r5:.3f} order {p5:.3f}; Richardson "
         f"{rich:.6f} - (1 - gamma M^2 (sigma - 1)) = {rich - sref(1.4, 0.042, sig):+.2e} (reported)")

    # G1.6: gamma 1.2 ladder and foam at 15
    e6 = [_sl(Rg[f"dns_c{N:g}"]) - s12 for N in lad]
    o6 = _order(e6)
    d6 = _sl(Rg["dns_c15"]) - _sl(Rg["foam_mclin_c15"])
    b6 = max(abs(_sl(Rg["dns_c15"]) - s12), abs(_sl(Rg["foam_mclin_c15"]) - s12))
    line("G1.6", min(o6) >= ORDER_MIN and abs(d6) <= b6,
         f"S_L - (1 - 1.2 M^2 (sigma - 1)) {e6[0]:+.3e} {e6[1]:+.3e} {e6[2]:+.3e}, orders {o6[0]:.3f} {o6[1]:.3f}; "
         f"dns - foam (15) {d6:+.3e} vs {b6:.3e}")

    # G1.7: the burnt T offset's gamma dependence
    D = {N: (Rg[f"dns_c{N:g}"]["Tb35"] - 1) - (R[f"dns_c{N:g}"]["Tb35"] - 1) for N in (15, 30)}
    ok7 = all(abs(D[N] - G17_D) <= G17_TOL for N in D)
    line("G1.7", ok7, "D " + " ".join(f"{N:g}: {D[N]:+.3e}" for N in D) + f" (band {G17_D:g} +- {G17_TOL:g}); "
         f"T_b/(sigma Tu) - 1: 1.4 {R['dns_c15']['Tb35'] - 1:+.3e} / {R['dns_c30']['Tb35'] - 1:+.3e}, "
         f"1.2 {Rg['dns_c15']['Tb35'] - 1:+.3e} / {Rg['dns_c30']['Tb35'] - 1:+.3e}")
    mut = Rg.get("dns_c15_qcp14")
    Dm = (mut["Tb35"] - 1) - (R["dns_c15"]["Tb35"] - 1) if mut else float("nan")
    out["G1.7 control"] = {"fails": not (abs(Dm - G17_D) <= G17_TOL), "value": Dm}
    L.append(f"       control (q on cp(1.4) at gamma 1.2, flag form): D {Dm:+.4f} "
             f"{'fails the band: live' if not abs(Dm - G17_D) <= G17_TOL else 'PASSES: dead control'}")
    return out, "\n".join(L)


# masterplan_combclose K1: the archived G1.5 / G3.3 M_n ladder (fv1d_hll dns_c*_mn, window [7, 14])
MN_ARCHIVED = {15: 0.9945059, 30: 0.9918456870897155}


def evaluate_mn(root, oracle, si):
    """masterplan_combclose G1.8 off fv1d_mn / fv1d_mn_vw / fv1d_mn_stockle's compare.json: the nozzle's flame
    (M 0.042, gamma 1.4, --shock hll) vs reactingFoam + the viscous work div(tau . u) (fv1d_mn_vw; stock
    reactingFoam's EEqn.H has none, issues.md N21) at 15 and 30 cells. S_inf = S30 + (S30 - S15)/(2^p - 1),
    p from the dns ladder; |S_dns - S_foam| <= max(|S_dns - S_inf|, |S_foam - S_inf|) at 15 and 30, |gap 30| <
    |gap 15|; the fresh correction in G1.4's form; 1 - gamma M^2 (sigma - 1) printed beside. Stock foam's
    gap (the stated model gap) and the finite-Mach oracle (tools/chem/mach_flame_oracle.py) are printed
    beside, not gated."""
    fv, st, vw = _load(root, "fv1d_mn"), _load(root, "fv1d_mn_stockle"), _load(root, "fv1d_mn_vw")
    R = {**fv["runs"], **{f"vw_{k}": v for k, v in vw["runs"].items()}}
    sig, M, g = 5.0, 0.042, 1.4
    out, L = {}, []

    def line(gid, ok, text):
        out[gid] = {"pass": bool(ok), "text": text}
        L.append(f"{gid:6s} {'PASS' if ok else 'UNMET'}  {text}")

    sd = {N: _sl(R[f"dns_c{N:g}"]) for N in (7.5, 15, 30)}
    r = (sd[7.5] - sd[15]) / (sd[15] - sd[30])
    p = math.log2(r) if r > 0 else float("nan")
    sinf = sd[30] + (sd[30] - sd[15]) / (2 ** p - 1)
    rep = {N: sd[N] - MN_ARCHIVED[N] for N in MN_ARCHIVED}
    spans = {N: R[f"dns_c{N:g}"]["SL_series"]["span"] for N in (7.5, 15, 30)}
    line("G1.8a", all(abs(rep[N]) <= spans[N] for N in rep),
         f"dns ladder {sd[7.5]:.7f} {sd[15]:.7f} {sd[30]:.7f}, ratio {r:.3f} order {p:.3f}, S_inf {sinf:.7f} "
         f"(1 - gamma M^2 (sigma - 1) = {sref(g, M, sig):.7f}, S_inf - it {sinf - sref(g, M, sig):+.2e}); vs archived "
         + " ".join(f"{N:g}: {rep[N]:+.1e} (span {spans[N]:.1e})" for N in rep))
    aT, ap = sens(root, oracle)
    Tu = pu = 1.0 / (g * M * M)
    def gaps(pre):
        fresh, band, raw = {}, {}, {}
        for N in (15, 30):
            dn, fn = R[f"dns_c{N:g}"], R[f"{pre}foam_mclin_c{N:g}"]
            fresh[N] = aT * (dn["TRef_series"]["mean"] - fn["TRef_series"]["mean"]) / Tu \
                + ap * (dn["pRef_series"]["mean"] - fn["pRef_series"]["mean"]) / pu
            band[N] = max(abs(sd[N] - sinf), abs(_sl(fn) - sinf))
            raw[N] = sd[N] - _sl(fn)
        corr = any(abs(fresh[N]) > band[N] / 4 for N in fresh)
        gap = {N: raw[N] - (fresh[N] if corr else 0.0) for N in raw}
        sf = [_sl(R[f"{pre}foam_mclin_c{N:g}"]) for N in (15, 30)]
        rich = sf[1] + (sf[1] - sf[0]) / 3.0
        txt = ("dns - foam " + " ".join(f"{N:g}: {gap[N]:+.3e}" for N in gap) + " vs max own error vs S_inf "
               + " ".join(f"{band[N]:.3e}" for N in band) + f"; |gap 30| < |gap 15|: {abs(gap[30]) < abs(gap[15])}; "
               f"foam S_L {sf[0]:.7f} {sf[1]:.7f}, Richardson (p 2) {rich:.7f} (- S_inf {rich - sinf:+.2e}); dS_fresh "
               + " ".join(f"{fresh[N]:+.2e}" for N in fresh) + (" (corrected)" if corr else " (raw S_L compared, <= band/4)"))
        ok = all(abs(gap[N]) <= band[N] for N in gap) and abs(gap[30]) < abs(gap[15])
        return ok, txt, gap

    ok, txt, gap = gaps("vw_")
    line("G1.8", ok, "foam + viscous work: " + txt)
    oks, txts, gaps_stock = gaps("")
    out["G1.8 stock"] = {"pass": bool(oks), "gap": gaps_stock, "gated": False}
    L.append(f"       stock reactingFoam (no viscous work, stated model gap, not gated): {txts}")
    try:        # the finite-Mach oracle at the exact fresh state (Tu, p0), ~3 s
        import sys as _sys
        _sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
        from mach_flame_oracle import solve as _mach
        o1, o0 = _mach(M, 1)[0], _mach(M, 0)[0]
        out["G1.8 oracle"] = {"S_work": o1, "S_none": o0}
        L.append(f"       finite-Mach oracle (Tu, p0): S {o1:.7f} with viscous work (S_inf - it {sinf - o1:+.2e}), "
                 f"{o0:.7f} without; difference {o1 - o0:+.3e} = {(o1 - o0) / (g * M * M * (sig - 1)):.3f} "
                 f"gamma M^2 (sigma - 1)")
    except (ImportError, SystemExit) as e:
        L.append(f"       finite-Mach oracle: not evaluated ({e})")
    sc = _sl(st["runs"]["foam_mclin_c15"]) - sd[15]
    out["G1.8 control"] = {"fails": abs(sc) >= G12_CONTROL, "value": sc}
    L.append(f"       control (stock Le 1.39, foam 15): gap {sc:+.4f} "
             f"{'>= 0.1: fails the band, live' if abs(sc) >= G12_CONTROL else '< 0.1: dead control'}")
    return out, "\n".join(L)
