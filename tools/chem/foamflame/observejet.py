"""Observables of the two-reactant cases (G3a diffusion2d, G3b candle2d) in
box units, by ONE code path for both solvers (docs/chem.md "Against
OpenFOAM reactingFoam", G3).

measure_jet is dns::measureJet (dns/diffusion.cpp) verbatim; jet_obs adds
the runner's diffusion stats row (gpu/runner_flame.hip measure(): Tmax,
TmaxIn past max(sink, 4) cells, burn = int omega, fuelIn through the first
interior face 0|1 as kSpeciesStage2 forms it (central mdot x central Y - rho
D dY/dx), fuelOut = rho u Y_F of the last row, pMean = box-mean p / p_ref)
and the grounding observables: the flame-sheet temperature on the Z_st
isoline, the Shvab-Zeldovich defect H = t - 1 + (q/cpT0)(Y_F - Y_F1 Z) and,
foam only, the carried tracer Zt against Z(Y_F, Y_O). dns' 2D frames carry
T - T0, Y_F, Y_O, omega (no rho, u): fuelIn, fuelOut, pMean, burnRatio of
dns are its stats rows.
"""
import math
import os

import numpy as np

from . import cases, writer
from .observe2d import _means, csv_cols, dns_frames, field_diff, foam_times

# --- loading ------------------------------------------------------------------


def foam_fields(d, si, nx, ny, sp):
    """Box-unit fields (ny, nx) of one foam time directory: T, rho = p/(R T)
    (both thermos: perfectGas), u, v, p, Y_F, Y_O, Zt, omega = Qdot/q."""
    def rd(n):          # a uniform field reads as a float: broadcast it
        v = writer.read_scalar_field(os.path.join(d, n))
        return np.full(nx * ny, v) if np.ndim(v) == 0 else v
    T = rd("T")
    p = rd("p")
    U = writer.read_vector(os.path.join(d, "U"))
    sh = (ny, nx)
    qf = os.path.join(d, "Qdot")
    Q = rd("Qdot") if os.path.exists(qf) else np.zeros_like(T)
    f = {"T": (T / si["TUnit"]).reshape(sh),
         "rho": (p / (si["R"] * T) / si["rho0"]).reshape(sh),
         "u": (U[:, 0] / si["U"]).reshape(sh), "v": (U[:, 1] / si["U"]).reshape(sh),
         "p": (p / si["pUnit"]).reshape(sh),
         "YF": rd(sp[0]).reshape(sh), "YO": rd(sp[1]).reshape(sh),
         "omega": (Q / si["q"] / si["omegaUnit"]).reshape(sh)}
    if "Zt" in sp and os.path.exists(os.path.join(d, "Zt")):
        f["Zt"] = rd("Zt").reshape(sh) / cases.ZT_EPS
    return f


# --- dns::measureJet ------------------------------------------------------------


def measure_jet(md, YF, YO, omega, h, yc):
    """dns::measureJet verbatim (fields (ny, nx), cell (i, j) at x = (i +
    1/2) h, y = (j + 1/2) h)."""
    YF = np.asarray(YF, float)
    YO = np.asarray(YO, float)
    om = np.asarray(omega, float)
    ny, nx = YF.shape
    jl = int(math.floor(yc / h - 0.5))
    jr = jl + 1
    zs = cases.zst(md)
    Z = cases.mixture_fraction(md, YF, YO)
    out = (Z < -1e-6) | (Z > 1 + 1e-6) | (YF < -1e-6) | (YF > md["yF1"] + 1e-6) | \
        (YO < -1e-6) | (YO > md["yO2"] + 1e-6)
    r = {"zMin": float(Z.min()), "zMax": float(Z.max()), "yfMin": float(YF.min()),
         "yoMin": float(YO.min()), "hits": float(out.sum())}
    wmax = float(om.max())
    axis = 0.5 * (Z[jl] + Z[jr])
    r["xTip"] = math.nan
    for i in range(nx - 1):
        if axis[i] >= zs and axis[i + 1] < zs:
            r["xTip"] = (i + 0.5 + (zs - axis[i]) / (axis[i + 1] - axis[i])) * h
            break
    r["xBase"] = math.nan
    if wmax > 0:
        colmax = om.max(axis=0)
        k = np.nonzero(colmax >= 0.1 * wmax)[0]
        if len(k):
            r["xBase"] = float(k[0]) * h
    r["half"] = math.nan
    if math.isfinite(r["xTip"]):
        i = min(max(int(math.floor(0.5 * r["xTip"] / h)), 0), nx - 1)
        c = Z[:, i]
        yR = yL = math.nan
        for j in range(jr, ny - 1):
            if c[j] >= zs and c[j + 1] < zs:
                yR = (j + 0.5 + (zs - c[j]) / (c[j + 1] - c[j])) * h
                break
        for j in range(jl, 0, -1):
            if c[j] >= zs and c[j - 1] < zs:
                yL = (j + 0.5 - (zs - c[j]) / (c[j - 1] - c[j])) * h
                break
        if math.isfinite(yR) and math.isfinite(yL):
            r["half"] = 0.5 * (yR - yL)
    # (not in dns) the flame's aspect: tip height over the Z_st width at tip/2
    r["aspect"] = r["xTip"] / (2 * r["half"]) if math.isfinite(r["half"]) else math.nan
    return r


def sheet_temperature(md, Z, T, h, yc, x0, x1):
    """Mean T (box) at the Z = Z_st crossings (each side walked outward from
    the axis, linear) over the rows with x in [x0, x1]."""
    zs = cases.zst(md)
    ny, nx = Z.shape
    jl = int(math.floor(yc / h - 0.5))
    jr = jl + 1
    vals = []
    for i in range(nx):
        x = (i + 0.5) * h
        if x < x0 or x > x1:
            continue
        c, t = Z[:, i], T[:, i]
        for rng, step in ((range(jr, ny - 1), 1), (range(jl, 0, -1), -1)):
            for j in rng:
                if c[j] >= zs and c[j + step] < zs:
                    w = (zs - c[j]) / (c[j + step] - c[j])
                    vals.append((1 - w) * t[j] + w * t[j + step])
                    break
    return float(np.mean(vals)) if vals else math.nan


def jet_obs(md, f, b, si, h, nb):
    """The runner's diffusion stats row + the grounding observables from box
    fields f (T, YF, YO, omega; rho, u, p optional; Zt optional)."""
    T0 = si["T0Box"]
    sig = md["sigma"]
    T = np.asarray(f["T"], float)
    ny, nx = T.shape
    r = measure_jet(md, f["YF"], f["YO"], f["omega"], h, b["yc"])
    ib = max(nb, 4)
    r["Tmax"] = float(T.max())
    r["TmaxIn"] = float(T[:, ib:nx - ib].max())
    r["burn"] = float(np.asarray(f["omega"], float).sum() * h * h)
    Z = cases.mixture_fraction(md, f["YF"], f["YO"])
    r["Tst"] = r["TstBase"] = math.nan
    if math.isfinite(r["xTip"]):
        r["Tst"] = sheet_temperature(md, Z, T, h, b["yc"], 0.25 * r["xTip"],
                                     0.75 * r["xTip"]) / (sig * T0)
        r["TstBase"] = sheet_temperature(md, Z, T, h, b["yc"], ib * h,
                                         ib * h + 4.0) / (sig * T0)
    om = np.asarray(f["omega"], float)
    k = np.unravel_index(np.argmax(om), om.shape)
    r["TatOmegaMax"] = float(T[k]) / (sig * T0)
    r["ZatOmegaMax"] = float(Z[k])
    r["xOmegaMax"] = (k[1] + 0.5) * h
    # Shvab-Zeldovich defect: 0 for any mixing at Le 1, both streams at T0
    H = T / T0 - 1 + cases.q_over_cpT0(md) * (np.asarray(f["YF"], float) - md["yF1"] * Z)
    r["Hmax"] = float(np.abs(H[:, ib:nx - ib]).max())
    for i in (0, 1, 4):
        r[f"H_i{i}"] = float(np.abs(H[:, i]).max())
    if "Zt" in f:
        dz = (Z - f["Zt"])[:, ib:nx - ib]
        r["ZtMax"] = float(np.abs(dz).max())
        r["ZtL2"] = float(np.sqrt((dz ** 2).mean()))
    if "rho" in f:
        rho, u, Y = (np.asarray(f[q], float) for q in ("rho", "u", "YF"))
        md_ = 0.25 * (rho[:, 0] + rho[:, 1]) * (u[:, 0] + u[:, 1])
        rdiff = 1.0                    # mu/Pr: constant mu = Pr (box)
        r["fuelIn"] = float(((md_ * 0.5 * (Y[:, 0] + Y[:, 1]) -
                              rdiff * (Y[:, 1] - Y[:, 0]) / h) * h).sum())
        r["fuelOut"] = float((rho[:, -1] * u[:, -1] * Y[:, -1] * h).sum())
        r["burnRatio"] = r["burn"] / r["fuelIn"]
        pref = T0
        if b.get("gravity", 0) > 0:
            pref = float(cases.candle_ambient_p(T0, b["gravity"], T0,
                                                (np.arange(nx) + 0.5) * h).mean())
        r["pMean"] = float(np.asarray(f["p"], float).mean() / pref)
        r["uMax"] = float(np.hypot(u, np.asarray(f["v"], float)).max())
        r["uOutMin"] = float(u[:, -1].min())
    return r


# --- per case -------------------------------------------------------------------

JET_Q = ("xTip", "half", "aspect", "xBase", "TmaxIn", "Tmax", "burn", "fuelIn",
         "fuelOut", "burnRatio", "pMean", "zMin", "zMax", "hits")
PORT_Q = ("xTip", "xBase", "half", "zMin", "zMax", "yfMin", "yoMin", "hits",
          "Tmax", "TmaxIn", "burn")
EXTRA_Q = ("Tst", "TstBase", "TatOmegaMax", "ZatOmegaMax", "xOmegaMax", "Hmax",
           "H_i0", "H_i1", "H_i4", "ZtMax", "ZtL2", "uMax", "uOutMin")


def _foam_inlet_flux(f, spec, m, si):
    """The fuel foam's inlet FACE carries: rho_f u_in Y_in (p_f = p_cell,
    zeroGradient; T_f = T_in) and Fick (rho D = 1) through the face."""
    u, Z, YF, YO, t, _ = cases.inlet_state(spec, m)
    h = m["h"]
    rf = f["p"][:, 0] / t                    # rho = p/(R T), R = 1, T = t T0, p0 = T0
    rf = rf / si["T0Box"]
    conv = float((rf * u * YF).sum() * h)
    diff = float(((YF - f["YF"][:, 0]) / (0.5 * h)).sum() * h)
    return conv, diff


def compare_rest(spec, si, root):
    """--quiescent: the hydrostatic boundary twin. dns: stats uMax, pMean
    (p over the ambient column), hits; foam: max |u| and max |p/p_a(x) - 1|
    per dump, T range."""
    N = spec["ladder"][0]
    m = cases.mesh_diffusion2d(spec, N)
    b = m["geom"]
    T0 = si["T0Box"]
    pa = cases.candle_ambient_p(T0, b["gravity"], T0, m["x"])[None, :]
    st = csv_cols(os.path.join(root, f"dns_c{N:g}", "flame_stats.csv"))
    res = {"case": spec["title"], "nx": m["nx"], "ny": m["ny"],
           "dns": {"t": st["t"].tolist(), "uMax": st["uMax"].tolist(),
                   "pMean": st["pMean"].tolist(), "hits": st["hits"].tolist()},
           "foam": {}}
    for run, _, _ in cases.foam_runs(spec):
        rows = {"t": [], "uMax": [], "dpMax": [], "pMean": [], "Tmin": [], "Tmax": []}
        for t_si, dd in foam_times(os.path.join(root, run)):
            f = foam_fields(dd, si, m["nx"], m["ny"], spec["species"])
            rows["t"].append(t_si / si["tUnit"])
            rows["uMax"].append(float(np.hypot(f["u"], f["v"]).max()))
            rows["dpMax"].append(float(np.abs(f["p"] / pa - 1).max()))
            rows["pMean"].append(float(f["p"].mean() / pa.mean()))
            rows["Tmin"].append(float(f["T"].min() / T0))
            rows["Tmax"].append(float(f["T"].max() / T0))
        res["foam"][run] = rows
        wf = os.path.join(root, run, "wall.txt")
        if os.path.exists(wf):
            res["foam"][run]["wall"] = open(wf).read().strip()
    return res


def table_rest(r):
    d = r["dns"]
    L = [f"{r['case']}  ({r['nx']} x {r['ny']} cells)",
         f"dns: max uMax {max(d['uMax']):.4e} U (at t {d['t'][int(np.argmax(d['uMax']))]:.3g}), "
         f"final {d['uMax'][-1]:.4e}; pMean range [{min(d['pMean']):.7f}, "
         f"{max(d['pMean']):.7f}]; hits max {max(d['hits']):g}"]
    for k, v in r["foam"].items():
        L.append(f"{k}: max |u| {max(v['uMax']):.4e} U (final {v['uMax'][-1]:.4e}); "
                 f"max |p/p_a - 1| {max(v['dpMax']):.3e}; pMean range "
                 f"[{min(v['pMean']):.7f}, {max(v['pMean']):.7f}]; T/T0 in "
                 f"[{min(v['Tmin']):.9f}, {max(v['Tmax']):.9f}]; wall {v.get('wall')}")
    return "\n".join(L)


def compare(spec, si, root):
    if spec["burner"].get("quiescent"):
        return compare_rest(spec, si, root)
    md = spec["model"]
    N = spec["ladder"][0]
    m = cases.mesh_diffusion2d(spec, N)
    b = m["geom"]
    nx, ny, h, nb = m["nx"], m["ny"], m["h"], m["nb"]
    T0 = si["T0Box"]
    win = spec["window"]
    res = {"case": spec["title"], "N": N, "nx": nx, "ny": ny, "h": h,
           "window": win, "tipCold": b["tipCold"], "zst": cases.zst(md),
           "foam": {}, "dns": {}}
    # dns: stats rows, the port on frames, the port check at the last frame
    def dns_side(pre):
        st = csv_cols(pre + "_stats.csv")
        st["aspect"] = st["xTip"] / (2 * st["half"])
        fr, fdt, meta = dns_frames(pre, nx, ny, ("temp", "fuel", "oxid", "omega"))
        nf = fr["temp"].shape[0]
        tf = fdt * np.arange(nf)
        rows = {"t": []}
        dframes = {}
        for k in range(nf):
            f = {"T": fr["temp"][k].astype(float) + T0, "YF": fr["fuel"][k],
                 "YO": fr["oxid"][k], "omega": fr["omega"][k]}
            o = jet_obs(md, f, b, si, h, nb)
            rows["t"].append(float(tf[k]))
            for q, v in o.items():
                rows.setdefault(q, []).append(v)
            dframes[round(tf[k], 3)] = f
        sq = [q for q in JET_Q + ("dtChem", "uMax", "uOutMin") if q in st]
        d = {"stats": _means(st, win, sq),
             "stats_6080": _means(st, (60.0, 80.0), sq)
             if st["t"][-1] >= 80 - 1e-9 else None,
             "frames": _means(rows, win, [q for q in rows if q != "t"]),
             "series": {q: rows[q] for q in ("t", "xTip", "TmaxIn")},
             "stats_series": {q: st[q].tolist()
                              for q in ("t", "xTip", "pMean", "burnRatio")},
             "ms_per_step": meta.get("msPerStep"), "t_end": float(st["t"][-1])}
        last = jet_obs(md, dframes[round(tf[-1], 3)], b, si, h, nb)
        d["port_check"] = {q: {"port": last[q], "dns": float(st[q][-1]),
                               "rel": (last[q] / float(st[q][-1]) - 1)
                               if float(st[q][-1]) != 0 else last[q] - float(st[q][-1])}
                           for q in PORT_Q}
        d["port_check_t"] = float(st["t"][-1])
        return d, dframes

    res["dns"], dframes = dns_side(os.path.join(root, f"dns_c{N:g}", "flame"))
    res["dns_extra"] = {}
    extra_frames = {}
    for lab, rd, flag in spec.get("extra_dns", []):
        pre = os.path.join(root, rd, "flame")
        if os.path.exists(pre + "_stats.csv"):
            res["dns_extra"][lab], extra_frames[lab] = dns_side(pre)
            res["dns_extra"][lab]["flag"] = flag
    # foam: every variant / check
    te = round(win[1], 3)
    for run, Nf, s in cases.foam_runs(spec):
        fd = os.path.join(root, run)
        times = foam_times(fd)
        if not times:
            continue
        frows = {"t": []}
        last_f = None
        for t_si, dd in times:
            t = t_si / si["tUnit"]
            f = foam_fields(dd, si, nx, ny, spec["species"])
            o = jet_obs(md, f, b, si, h, nb)
            o["fuelInFace"], o["fuelInFick"] = _foam_inlet_flux(f, spec, m, si)
            o["burnRatioFace"] = o["burn"] / (o["fuelInFace"] + o["fuelInFick"])
            o["pIn_jet"] = float(f["p"][ny // 2, 0] / T0 - 1)
            o["pIn_co"] = float(f["p"][0, 0] / T0 - 1)
            frows["t"].append(t)
            for q, v in o.items():
                frows.setdefault(q, []).append(v)
            if abs(t - te) < 1e-6:
                last_f = f
        w = win if "restart" not in s else (s["restart"] + 4.0, s["tt"])
        r = {"window": w, "means": _means(frows, w, [q for q in frows if q != "t"]),
             "series": {q: frows[q] for q in ("t", "xTip", "TmaxIn", "pMean")},
             "settings": {k: v for k, v in s.items() if k != "per"}}
        wf = os.path.join(fd, "wall.txt")
        if os.path.exists(wf):
            r["wall"] = open(wf).read().strip()
        if last_f is not None:
            fd_ = dframes.get(te)
            Zd = cases.mixture_fraction(md, fd_["YF"], fd_["YO"])
            Zf = cases.mixture_fraction(md, last_f["YF"], last_f["YO"])
            thd = (fd_["T"] - T0) / ((md["sigma"] - 1) * T0)
            thf = (last_f["T"] - T0) / ((md["sigma"] - 1) * T0)
            r["Z_diff"] = field_diff(Zd, Zf, m["x"], m["y"], nb)
            r["theta_diff"] = field_diff(thd, thf, m["x"], m["y"], nb)
            for lab, ef in extra_frames.items():     # the dns twins at te
                if te not in ef:
                    continue
                fe = ef[te]
                r[f"Z_diff [{lab}]"] = field_diff(
                    cases.mixture_fraction(md, fe["YF"], fe["YO"]), Zf, m["x"],
                    m["y"], nb)
                r[f"theta_diff [{lab}]"] = field_diff(
                    (fe["T"] - T0) / ((md["sigma"] - 1) * T0), thf, m["x"], m["y"], nb)
            # the first cell row: dns' pinned row vs foam's first cell
            r["row0_T"] = {"dns_max": float(fd_["T"][:, 0].max() / (md["sigma"] * T0)),
                           "foam_max": float(last_f["T"][:, 0].max() / (md["sigma"] * T0))}
        res["foam"][run] = r
    # the base variant against its dt check over the check's window
    for run, r in res["foam"].items():
        s = r["settings"]
        if "restart" in s:
            base = res["foam"].get(f"foam_{s['restart_base']}_c{N:g}")
            if base is None:
                continue
            # the base run's means over the check's window
            fdb = os.path.join(root, f"foam_{s['restart_base']}_c{N:g}")
            brows = {"t": []}
            for t_si, dd in foam_times(fdb):
                t = t_si / si["tUnit"]
                if not (r["window"][0] - 1e-6 <= t <= r["window"][1] + 1e-6):
                    continue
                o = jet_obs(md, foam_fields(dd, si, nx, ny, spec["species"]), b, si, h, nb)
                brows["t"].append(t)
                for q, v in o.items():
                    brows.setdefault(q, []).append(v)
            r["base_means"] = _means(brows, r["window"], [q for q in brows if q != "t"])
    return res


def _fmt(v):
    if v is None:
        return "-"
    return f"{v['mean']:.6g} ({v['h1']:.6g}/{v['h2']:.6g})"


def table(r):
    if "uMax" in r["dns"] and "stats" not in r["dns"]:
        return table_rest(r)
    d = r["dns"]
    L = [f"{r['case']}  ({r['nx']} x {r['ny']} cells, h = l/{r['N']:g}), window "
         f"{r['window']}; Z_st {r['zst']:.6g}; constant-density Burke-Schumann tip "
         f"{r['tipCold']:.4f} l (scale)",
         f"port check vs dns stats at t = {d['port_check_t']:g} (rel; hits abs): " +
         " ".join(f"{q} {v['rel']:+.1e}" for q, v in d["port_check"].items()), "",
         "mean (halves) over the window; dns: stats rows | the port on dns frames"]
    runs = [(k, v) for k, v in r["foam"].items() if "restart" not in v["settings"]]
    ex = list(r.get("dns_extra", {}).items())
    cols = [("dns stats", d["stats"]), ("dns frames", d["frames"])] + \
        ([(f"{ex[0][0]} stats", ex[0][1]["stats"]),
          (f"{ex[0][0]} frames", ex[0][1]["frames"])] if ex else []) + \
        [(k, v["means"]) for k, v in runs]
    keys = list(JET_Q) + list(EXTRA_Q) + ["fuelInFace", "fuelInFick",
                                          "burnRatioFace", "pIn_jet", "pIn_co",
                                          "dtChem"]
    for q in keys:
        cells = [_fmt(c.get(q)) for _, c in cols]
        if all(x == "-" for x in cells):
            continue
        L.append(f"{q:13s} " + " | ".join(cells))
    L.append("columns: " + " | ".join(c for c, _ in cols))
    if d.get("stats_6080"):
        L.append("dns stats [60, 80] (gpu_diffusion_jet's window): " + " ".join(
            f"{q} {v['mean']:.6g}" for q, v in d["stats_6080"].items()))
    L.append(f"dns {d['ms_per_step']:.3f} ms/step, t_end {d['t_end']:g}")
    for lab, e in ex:
        L.append(f"{lab} ({e['flag']}) stats over the window: " + " ".join(
            f"{q} {e['stats'][q]['mean']:.6g}" for q in
            ("xTip", "half", "TmaxIn", "burn", "fuelIn", "burnRatio", "pMean")) +
            f"; frames: Tst {e['frames']['Tst']['mean']:.6g} H_i0 "
            f"{e['frames']['H_i0']['mean']:.4g} Hmax {e['frames']['Hmax']['mean']:.4g}"
            f"; {e['ms_per_step']:.3f} ms/step")
    # settling: the tip every 4 l/U
    def every(ts, xs, step=4.0):
        out = []
        for t, x in zip(ts, xs):
            if abs(t / step - round(t / step)) < 1e-6:
                out.append(f"{t:g}:{x:.2f}")
        return " ".join(out)
    L.append("tip(t) dns frames: " + every(d["series"]["t"], d["series"]["xTip"]))
    for k, v in r["foam"].items():
        L.append(f"tip(t) {k}: " + every(v["series"]["t"], v["series"]["xTip"]))
    for k, v in r["foam"].items():
        for q in [q for q in v if q.startswith(("Z_diff", "theta_diff"))]:
            x = v[q]
            L.append(f"  dns - {k} t {r['window'][1]:g} {q:24s} Linf {x['linf']:.3e} "
                     f"L2 {x['l2']:.3e} at x {x['at_x']:.2f} y {x['at_y']:.2f}")
        if "row0_T" in v:
            L.append(f"  first cell row T/T_ad max: dns {v['row0_T']['dns_max']:.5f}, "
                     f"{k} {v['row0_T']['foam_max']:.5f}")
        if "wall" in v:
            L.append(f"  {k} wall: {v['wall']}")
    for k, v in r["foam"].items():
        if "base_means" not in v:
            continue
        L.append(f"time-step check {k} (window {v['window']}) vs its base, same window:")
        for q in ("xTip", "half", "TmaxIn", "burnRatio", "pMean", "Tst", "Hmax", "ZtMax"):
            a, bb = v["means"].get(q), v["base_means"].get(q)
            if a and bb:
                L.append(f"  {q:10s} check {a['mean']:.6g} base {bb['mean']:.6g} "
                         f"rel {a['mean'] / bb['mean'] - 1:+.2e}")
    return "\n".join(L)
