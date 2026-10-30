"""2D observables of both solvers (cases G2a bunsen2d, G2b wrinkle2d), in box
units, by ONE code path (docs/chem.md "Against OpenFOAM reactingFoam").

Bunsen: a faithful numpy port of dns::measureCone / dns::flankMassFlux
(dns/bunsen.cpp) and of the runner's SLflank / burn / fuelIn
(gpu/runner_flame.hip measure()), checked against dns' own stats row at the
final frame (`check_dns`). dns' 2D frames carry T - Tu, Y, omega only (no
rho, u, v): mFlank and fuelIn are therefore ported but read from dns'
stats on the dns side; |u| comes from the z-extruded nz 8 twin (`speed`
frames, byte-identical to 2D by docs/chem.md).

Wrinkle: per frame the mode-1 amplitude of (a) the fresh-mass thickness
x_I(y) = h sum_i Y (tools/chem/wrinkle_rate.py) and (b) the mid-temperature
isoline x_T(y) (first Tmid crossing per row, linear); growth rate = the
least-squares slope of ln a over the fit window. Frame k of dns sits at
t = k fdt (the first step with t >= k fdt: late by < one dt).
"""
import glob
import json
import math
import os

import numpy as np

from . import writer

# --- loading ------------------------------------------------------------------


def dns_frames(prefix, nx, ny, names=("temp", "fuel", "omega")):
    """{name: (nf, ny, nx) float32}, frameDt."""
    meta = json.load(open(prefix + ".json"))
    out = {}
    for n in names:
        p = f"{prefix}_{n}.bin"
        if not os.path.exists(p):
            continue
        a = np.fromfile(p, dtype=np.float32)
        nf = a.size // (nx * ny)
        out[n] = a[: nf * nx * ny].reshape(nf, ny, nx)
    return out, meta["frameDt"], meta


def csv_cols(path):
    import csv
    with open(path) as f:
        r = csv.reader(f)
        head = next(r)
        rows = np.array([[float(v) for v in row] for row in r])
    return {h: rows[:, k] for k, h in enumerate(head)}


def foam_times(rundir):
    ts = []
    for d in glob.glob(os.path.join(rundir, "[0-9]*")):
        try:
            t = float(os.path.basename(d))
        except ValueError:
            continue
        if t > 0 and os.path.exists(os.path.join(d, "T")):
            ts.append((t, d))
    return sorted(ts)


def foam_fields(d, si, nx, ny, sp0="F"):
    """Box-unit fields (ny, nx) of one foam time directory: T, rho (= p/(R
    T), hePsiThermo), u, v, p, Y, omega (Qdot / q)."""
    T = writer.read_scalar_field(os.path.join(d, "T"))
    p = writer.read_scalar_field(os.path.join(d, "p"))
    U = writer.read_vector(os.path.join(d, "U"))
    Y = writer.read_scalar_field(os.path.join(d, sp0))
    qf = os.path.join(d, "Qdot")
    Q = writer.read_scalar_field(qf) if os.path.exists(qf) else np.zeros_like(T)
    sh = (ny, nx)
    return {"T": (T / si["TUnit"]).reshape(sh),
            "rho": (p / (si["R"] * T) / si["rho_u"]).reshape(sh),
            "u": (U[:, 0] / si["SL"]).reshape(sh),
            "v": (U[:, 1] / si["SL"]).reshape(sh),
            "p": (p / si["pUnit"]).reshape(sh), "Y": Y.reshape(sh),
            "omega": (Q / si["q"] / si["omegaUnit"]).reshape(sh)}


# --- dns::measureCone / flankMassFlux ------------------------------------------


def _fit(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    mx, my = x.mean(), y.mean()
    b = ((x - mx) * (y - my)).sum() / ((x - mx) ** 2).sum()
    a = my - b * mx
    rms = math.sqrt(((y - (a + b * x)) ** 2).sum() / len(x))
    return a, b, rms


def measure_cone(T, h, Tmid, yc, omega=None, win=4):
    """dns::measureCone verbatim (T, omega (ny, nx); cell (i, j) at x = (i +
    1/2) h, y = (j + 1/2) h)."""
    ny, nx = T.shape
    T = np.asarray(T, float)
    jl = int(math.floor(yc / h - 0.5))
    jr = jl + 1
    axis = 0.5 * (T[jl] + T[jr])
    m = {"ok": False, "xTip": math.nan, "xBase": math.nan, "rows": 0}
    m["axisHot"] = not axis[0] < Tmid
    if m["axisHot"]:
        return m
    for i in range(nx - 1):
        if axis[i] < Tmid and axis[i + 1] >= Tmid:
            m["xTip"] = (i + 0.5 + (Tmid - axis[i]) / (axis[i + 1] - axis[i])) * h
            break
    if not math.isfinite(m["xTip"]):
        return m
    wcut = 0.0 if omega is None else 0.1 * float(np.max(omega))

    def burns(i, j):
        if omega is None:
            return True
        lo, hi = max(0, j - win), min(ny - 1, j + 1 + win)
        return float(np.max(omega[lo:hi + 1, i])) >= wcut

    def flank(i):
        if not axis[i] < Tmid:
            return None
        yR = yL = math.nan
        col = T[:, i]
        for j in range(jr, ny - 1):
            if col[j] < Tmid and col[j + 1] >= Tmid:
                if burns(i, j):
                    yR = (j + 0.5 + (Tmid - col[j]) / (col[j + 1] - col[j])) * h
                break
        for j in range(jl, 0, -1):
            if col[j] < Tmid and col[j - 1] >= Tmid:
                if burns(i, j - 1):
                    yL = (j + 0.5 - (Tmid - col[j]) / (col[j - 1] - col[j])) * h
                break
        if not (math.isfinite(yR) and math.isfinite(yL)):
            return None
        return yL, yR

    for i in range(nx):
        if (i + 0.5) * h < m["xTip"] and flank(i):
            m["xBase"] = (i + 0.5) * h
            break
    if not math.isfinite(m["xBase"]):
        return m
    lo = m["xBase"] + 0.25 * (m["xTip"] - m["xBase"])
    hi = m["xBase"] + 0.75 * (m["xTip"] - m["xBase"])
    xs, hw, sr, sl, fi, fyL, fyR = [], [], [], [], [], [], []
    for i in range(nx):
        x = (i + 0.5) * h
        if x < lo or x > hi:
            continue
        f = flank(i)
        if not f:
            continue
        fi.append(i)
        fyL.append(f[0])
        fyR.append(f[1])
        xs.append(x)
        hw.append(0.5 * (f[1] - f[0]))
        sr.append(f[1] - yc)
        sl.append(yc - f[0])
    m.update({"rows": len(fi), "fi": fi, "fyL": fyL, "fyR": fyR})
    if len(fi) < 4:
        return m
    a, b, rms = _fit(xs, hw)
    m.update({"tanA": -b, "tanR": -_fit(xs, sr)[1], "tanL": -_fit(xs, sl)[1],
              "rms": rms, "ok": True, "iLo": fi[0], "iHi": fi[-1],
              "hw0": a})   # (not in dns) the fit's half-width at x = 0
    return m


def flank_mass_flux(m, rho, u, v, h):
    """dns::flankMassFlux verbatim."""
    if not m["ok"]:
        return math.nan
    nR, nL = math.hypot(1.0, m["tanR"]), math.hypot(1.0, m["tanL"])
    nxR, nyR = m["tanR"] / nR, 1.0 / nR
    nxL, nyL = m["tanL"] / nL, -1.0 / nL

    def lerp(f, i, y):
        s = y / h - 0.5
        j = int(math.floor(s))
        w = s - j
        return (1 - w) * float(f[j, i]) + w * float(f[j + 1, i])

    tot = 0.0
    for k, i in enumerate(m["fi"]):
        for y, ex, ey in ((m["fyR"][k], nxR, nyR), (m["fyL"][k], nxL, nyL)):
            tot += lerp(rho, i, y) * (lerp(u, i, y) * ex + lerp(v, i, y) * ey)
    return tot / (2.0 * len(m["fi"]))


def bunsen_obs(T, omega, h, Tu, sigma, yc, uJet, dth, rho=None, u=None,
               v=None, rowY=None):
    """The runner's bunsen stats row from fields (2D, hz = 1): cone, sinRatio,
    SLflank, burn, and with rho, u, v: mFlank, fuelIn (the inflow row's
    rho u Y_row h), burnRatio."""
    Tmid = 0.5 * (Tu + sigma * Tu)
    c = measure_cone(T, h, Tmid, yc, omega, int(math.ceil(0.5 * dth / h)))
    dV = h * h
    colw = np.asarray(omega, float).sum(axis=0)       # col(i, 4): sum over j
    o = {"xTip": c["xTip"], "xBase": c["xBase"], "rows": c["rows"],
         "burn": float(colw.sum() * dV), "Tmax": float(np.max(T))}
    if c["ok"]:
        sinA = c["tanA"] / math.hypot(1.0, c["tanA"])
        cosA = 1.0 / math.hypot(1.0, c["tanA"])
        w = colw[c["iLo"]:c["iHi"] + 1].sum()
        L = 2.0 * (c["iHi"] - c["iLo"] + 1) * h / cosA
        o.update({"tanA": c["tanA"], "tanL": c["tanL"], "tanR": c["tanR"],
                  "rms": c["rms"], "sinRatio": sinA * uJet,
                  "SLflank": float(w * dV / L),
                  "alpha_deg": math.degrees(math.atan(c["tanA"])),
                  # extra (not dns stats): the straight flank's half-width at
                  # the exit plane and the tip's offset from its apex
                  "hw0": c["hw0"], "tipOff": c["xTip"] - c["hw0"] / c["tanA"]})
    if rho is not None:
        o["mFlank"] = flank_mass_flux(c, rho, u, v, h)
        if rowY is not None:
            o["fuelIn"] = float((np.asarray(rho[:, 0], float) * u[:, 0] *
                                 rowY).sum() * h)
            o["burnRatio"] = o["burn"] / o["fuelIn"]
    return o, c


# --- wrinkle amplitude ---------------------------------------------------------


def mode1(f):
    """Mode-1 amplitude 2 |DFT_1| / n of a periodic row profile, and its
    phase."""
    z = np.fft.rfft(np.asarray(f, float))[1] * 2 / len(f)
    return abs(z), math.atan2(z.imag, z.real)


def front_positions(T, Y, h, Tmid):
    """x_I(y) = h sum_i Y and x_T(y) = the first Tmid crossing per row."""
    xI = np.asarray(Y, float).sum(axis=1) * h
    T = np.asarray(T, float)
    ny, nx = T.shape
    xT = np.full(ny, math.nan)
    for j in range(ny):
        r = T[j]
        k = np.nonzero((r[:-1] < Tmid) & (r[1:] >= Tmid))[0]
        if len(k):
            i = k[0]
            xT[j] = (i + 0.5 + (Tmid - r[i]) / (r[i + 1] - r[i])) * h
    return xI, xT


def growth(t, a, win):
    t, a = np.asarray(t), np.asarray(a)
    k = (t >= win[0] - 1e-9) & (t <= win[1] + 1e-9)
    s, c = np.polyfit(t[k], np.log(a[k]), 1)
    res = np.log(a[k]) - (s * t[k] + c)
    return float(s), float(np.sqrt((res ** 2).mean())), int(k.sum())


# --- theory: Matalon & Matkowsky 1982 ------------------------------------------


def dl_theory(sigma, k):
    """Linear growth rate of a planar Le = 1 flame with CONSTANT transport
    (lambda, mu, rho D const) in units S_L, l_f = D_th,u/S_L (= delta_D):
    omega = omega_DL k - B1 k^2 (Pr B3 = 0 for constant viscosity, B2
    multiplies Le_eff - 1 = 0). Matalon, Fluid Dyn. Res. 50 (2018) 051412,
    eqs (12), (19) and the appendix (= Matalon & Matkowsky 1982, JFM 124);
    sigma = rho_u/rho_b. O(k l_f), large Zeldovich number."""
    s = sigma
    w = (math.sqrt(s ** 3 + s ** 2 - s) - s) / (s + 1)
    B1 = 0.5 * s * ((s + 1 + 2 * w) * s * math.log(s) / (s - 1) + s - 1) / (
        s + (s + 1) * w)
    return {"omega_DL": w, "B1": B1, "lambda_c": 2 * math.pi * B1 / w,
            "inviscid": w * k, "mm": w * k - B1 * k * k}


# --- field differences ---------------------------------------------------------


def field_diff(a, b, x, y, skip=0):
    """L-inf, L2 (rms) of a - b over cells i in [skip, nx - skip), and where
    the max sits (x, y in delta_D)."""
    d = np.asarray(a, float) - np.asarray(b, float)
    if skip:
        d = d[:, skip:-skip]
        x = x[skip:-skip]
    k = np.unravel_index(np.argmax(np.abs(d)), d.shape)
    return {"linf": float(np.abs(d).max()), "l2": float(np.sqrt((d ** 2).mean())),
            "at_x": float(x[k[1]]), "at_y": float(y[k[0]])}


# --- per case -------------------------------------------------------------------

BUNSEN_Q = ("xTip", "tanA", "sinRatio", "SLflank", "mFlank", "burnRatio",
            "rms", "Tmax")


def _means(rows, win, keys):
    """Window mean and halves of {key: series} sampled at rows['t']."""
    t = np.asarray(rows["t"])
    a, b = win
    mid = 0.5 * (a + b)
    k = (t >= a - 1e-6) & (t <= b + 1e-6)
    k1 = k & (t <= mid + 1e-6)
    k2 = k & (t > mid + 1e-6)
    out = {}
    for q in keys:
        v = np.asarray(rows[q], float)
        out[q] = {"mean": float(np.nanmean(v[k])), "h1": float(np.nanmean(v[k1])),
                  "h2": float(np.nanmean(v[k2])), "n": int(k.sum())}
    return out


def _dns_frame_rows(fr, fdt, win, h, Tu, sig, b, dth):
    """The port on dns frames inside the window (integer tau: like foam)."""
    tf = fdt * np.arange(fr["temp"].shape[0])
    rows = {"t": []}
    for k, t in enumerate(tf):
        if not (win[0] - 1e-6 <= t <= win[1] + 1e-6):
            continue
        ob, _ = bunsen_obs(fr["temp"][k].astype(float) + Tu, fr["omega"][k], h,
                           Tu, sig, b["yc"], b["uJet"], dth)
        rows["t"].append(t)
        for q in ("xTip", "tanA", "sinRatio", "SLflank", "rms", "Tmax", "hw0",
                  "tipOff"):
            rows.setdefault(q, []).append(ob.get(q, math.nan))
    return _means(rows, win, [q for q in rows if q != "t"])


def compare_bunsen(spec, o, si, root, dns_runs=(("dns", None),
                                                ("dns_tinflow", "_tinflow"))):
    """dns_runs: (label, suffix of dns_c<N>) -- None = the case's own run
    (job 659, rho-target inflow), "_tinflow" = --nscbc-tinflow (job 668)."""
    from . import cases
    N = spec["ladder"][0]
    m = cases.mesh_bunsen2d(spec, N, o.dth)
    nx, ny, h = m["nx"], m["ny"], m["h"]
    Tu, sig = si["TuBox"], o.sigma
    b = m["geom"]
    uin, tin, rowY, _ = cases.bunsen_inflow(b, m["y"])
    win = spec["window"]
    te = round(win[1], 3)
    res = {"case": spec["title"], "N": N, "nx": nx, "ny": ny, "h": h,
           "window": win, "dns": {}}
    # foam
    fd = os.path.join(root, f"foam_mclin_c{N}")
    rows = {"t": []}
    fields = {}
    for t_si, d in foam_times(fd):
        t = t_si / si["tau"]
        if not (win[0] - 1e-6 <= t <= win[1] + 1e-6):
            continue
        f = foam_fields(d, si, nx, ny)
        ob, _ = bunsen_obs(f["T"], f["omega"], h, Tu, sig, b["yc"], b["uJet"],
                           o.dth, f["rho"], f["u"], f["v"], rowY)
        # the inlet face as foam carries it: rho_f u_in Y_in (p_f = p_c) and
        # Fick through the face (dns' open face carries no diffusion)
        rf = f["p"][:, 0] / (tin * Tu)
        conv = float((rf * uin * rowY).sum() * h)
        diff = float(((rowY - f["Y"][:, 0]) / (0.5 * h)).sum() * h)  # rho D = 1
        ob.update({"fuelInFace": conv, "fuelInDiff": diff,
                   "burnRatioFace": ob["burn"] / (conv + diff),
                   "pIn_jet": float(f["p"][ny // 2, 0] / Tu - 1),
                   "pIn_coflow": float(f["p"][0, 0] / Tu - 1)})
        rows["t"].append(t)
        for q, v in ob.items():
            rows.setdefault(q, []).append(v)
        fields[round(t, 3)] = f
    res["foam"] = _means(rows, win, [q for q in rows if q not in ("t", "rows")])
    bf = os.path.join(fd, "postProcessing/burn/0/volFieldValue.dat")
    if os.path.exists(bf):          # volIntegrate(Qdot) every 10 steps
        bb = np.loadtxt(bf)
        burn = bb[:, 1] / si["q"] / si["omegaUnit"] / (h * si["dD"]) ** 3 * h * h
        res["foam_burn_series"] = _means({"t": bb[:, 0] / si["tau"], "burn": burn},
                                         win, ["burn"])
    wf = os.path.join(fd, "wall.txt")
    if os.path.exists(wf):
        res["foam_wall"] = open(wf).read().strip()
    ff = fields.get(te)
    th_f = None if ff is None else (ff["T"] - Tu) / ((sig - 1) * Tu)
    for label, suf in dns_runs:
        pre = os.path.join(root, f"dns_c{N}{suf or ''}", "flame")
        if not os.path.exists(pre + "_stats.csv"):
            continue
        st = csv_cols(pre + "_stats.csv")
        fr, fdt, meta = dns_frames(pre, nx, ny)
        r = {"stats": _means(st, win, BUNSEN_Q),
             "frames": _dns_frame_rows(fr, fdt, win, h, Tu, sig, b, o.dth),
             "ms_per_step": meta.get("msPerStep"), "t_end": float(st["t"][-1])}
        if suf is None:
            r["stats_2030"] = _means(st, (20.0, 30.0), BUNSEN_Q)
            # the port against dns' own stats row at the final frame
            obs, _ = bunsen_obs(fr["temp"][-1].astype(float) + Tu,
                                fr["omega"][-1], h, Tu, sig, b["yc"], b["uJet"],
                                o.dth)
            r["port_check"] = {q: {"port": obs[q], "dns": float(st[q][-1]),
                                   "rel": obs[q] / float(st[q][-1]) - 1}
                               for q in ("xTip", "xBase", "tanA", "tanL", "tanR",
                                         "rms", "sinRatio", "SLflank", "burn",
                                         "Tmax")}
            r["port_check_t"] = float(st["t"][-1])
        # fields at te: the run's frame at exactly te (a tt = te run), else
        # the nz 8 extruded twin's (tt 25, byte-identical to 2D through t 20)
        Td, spd = None, None
        if abs(r["t_end"] - te) < 1e-9:
            Td = fr["temp"][int(round(te / fdt))].astype(float)
        p8 = os.path.join(root, f"dns_c{N}{suf or ''}_nz8", "flame")
        if os.path.exists(p8 + "_speed.bin"):
            f8, fdt8, _ = dns_frames(p8, nx, ny, ("temp", "speed"))
            k8 = int(round(te / fdt8))
            k20 = int(round(20.0 / fdt8))
            r["nz8_vs_2d_t20_maxabs"] = float(np.abs(
                f8["temp"][k20].astype(float) - fr["temp"][int(round(20.0 / fdt))]).max())
            Td = f8["temp"][k8].astype(float)
            spd = f8["speed"][k8].astype(float)
        if Td is not None and th_f is not None:
            thd = Td / ((sig - 1) * Tu)
            r["theta"] = field_diff(thd, th_f, m["x"], m["y"])
            r["theta_nosink"] = field_diff(thd, th_f, m["x"], m["y"], m["nb"])
            r["jetT_x20"] = float((Td[ny // 2, int(20 / h)] + Tu) / Tu - 1)
            r["inflowT_jet"] = float((Td[ny // 2, 0] + Tu) / (tin[ny // 2] * Tu) - 1)
            r["inflowT_coflow"] = float((Td[0, 0] + Tu) / (tin[0] * Tu) - 1)
        if spd is not None and ff is not None:
            spf = np.hypot(ff["u"], ff["v"])
            r["speed"] = field_diff(spd, spf, m["x"], m["y"])
            r["speed_nosink"] = field_diff(spd, spf, m["x"], m["y"], m["nb"])
            r["speed_max"] = float(spd.max())
        res["dns"][label] = r
    if ff is not None:
        res["foam_jetT_x20"] = float(ff["T"][ny // 2, int(20 / h)] / Tu - 1)
        res["foam_inflowT_jet"] = float(ff["T"][ny // 2, 0] / (tin[ny // 2] * Tu) - 1)
    return res


def table_bunsen(r):
    d0 = r["dns"]["dns"]
    L = [f"{r['case']}  ({r['nx']} x {r['ny']} cells, h = delta_th/{r['N']:g}), "
         f"window {r['window']}",
         f"port check vs dns stats at t = {d0['port_check_t']:g} (rel): " +
         " ".join(f"{q} {v['rel']:+.1e}" for q, v in d0["port_check"].items()),
         "", "mean (halves); dns: stats rows | the port on frames at integer tau"]
    cols = [("dns stats", d0["stats"]), ("dns frames", d0["frames"])]
    for lab, rr in r["dns"].items():
        if lab != "dns":
            cols += [(f"{lab} stats", rr["stats"]), (f"{lab} frames", rr["frames"])]
    cols.append(("foam", r["foam"]))
    for q in BUNSEN_Q + ("hw0", "tipOff"):
        cells = []
        for _, d in cols:
            v = d.get(q)
            cells.append("-" if v is None else
                         f"{v['mean']:.5f} ({v['h1']:.5f}/{v['h2']:.5f})")
        L.append(f"{q:9s} " + " | ".join(cells))
    L.append("columns: " + " | ".join(c for c, _ in cols))
    L.append(f"dns [20,30] stats: " + " ".join(
        f"{q} {v['mean']:.5f}" for q, v in d0["stats_2030"].items()))
    for q in ("fuelIn", "fuelInFace", "fuelInDiff", "burn", "burnRatioFace",
              "pIn_jet", "pIn_coflow"):
        if q in r["foam"]:
            L.append(f"  foam {q:14s} {r['foam'][q]['mean']:.6g}")
    if "foam_burn_series" in r:
        L.append(f"  foam burn (volIntegrate series) "
                 f"{r['foam_burn_series']['burn']['mean']:.6f}")
    L.append(f"  foam: jet T at x 20 delta_D /Tu - 1 {r.get('foam_jetT_x20', math.nan):+.3e}"
             f", inflow-row jet T/T_row - 1 {r.get('foam_inflowT_jet', math.nan):+.3e}")
    for lab, rr in r["dns"].items():
        L.append(f"  {lab}: jet T x 20 /Tu - 1 {rr.get('jetT_x20', math.nan):+.3e}, "
                 f"inflow-row T/T_row - 1 jet {rr.get('inflowT_jet', math.nan):+.3e} "
                 f"coflow {rr.get('inflowT_coflow', math.nan):+.3e}; "
                 f"{rr['ms_per_step']:.3f} ms/step")
        if "nz8_vs_2d_t20_maxabs" in rr:
            L.append(f"    nz 8 twin vs 2D temp frame t 20: max |diff| "
                     f"{rr['nz8_vs_2d_t20_maxabs']:g}; dns |u| max {rr['speed_max']:.4f}")
        for k in ("theta", "theta_nosink", "speed", "speed_nosink"):
            if k in rr:
                v = rr[k]
                L.append(f"    {lab} - foam t {r['window'][1]:g} {k:13s} Linf "
                         f"{v['linf']:.3e} L2 {v['l2']:.3e} at x {v['at_x']:.2f} "
                         f"y {v['at_y']:.2f}")
    L.append(f"  foam wall: {r.get('foam_wall')}")
    return "\n".join(L)


def wrinkle_series(T, Y, t, h, Tmid):
    aI, aT, pI = [], [], []
    for k in range(len(t)):
        xI, xT = front_positions(T[k], Y[k], h, Tmid)
        a, ph = mode1(xI)
        aI.append(a)
        pI.append(ph)
        aT.append(mode1(xT)[0])
    return np.array(aI), np.array(aT), np.array(pI)


def compare_wrinkle(spec, o, si, root, extra_dns=()):
    """extra_dns: [(name, rundir, N, ny, sigma, dth)] dns-only runs."""
    from . import cases
    N = spec["ladder"][0]
    m = cases.mesh_wrinkle2d(spec, N, o.dth)
    nx, ny, h = m["nx"], m["ny"], m["h"]
    Tu = si["TuBox"]
    Tmid = 0.5 * (1 + o.sigma) * Tu
    lam = ny * h
    th = dl_theory(o.sigma, 2 * math.pi / lam)
    res = {"case": spec["title"], "N": N, "nx": nx, "ny": ny,
           "lambda_dD": lam, "theory": th, "runs": {}}
    fit = spec["fit"]
    wins = [fit, (4.0, 16.0)]
    pre = os.path.join(root, f"dns_c{N}", "flame")
    fr, fdt, _ = dns_frames(pre, nx, ny, ("temp", "fuel"))
    t = fdt * np.arange(fr["temp"].shape[0])
    aI, aT, pI = wrinkle_series(fr["temp"].astype(float) + Tu, fr["fuel"], t, h, Tmid)
    res["runs"]["dns"] = {"t": t.tolist(), "aI": aI.tolist(), "aT": aT.tolist(),
                          "phase": pI.tolist()}
    fd = os.path.join(root, f"foam_mclin_c{N}")
    ts, TT, YY = [0.0], [], []
    ic = cases.ic_wrinkle2d(spec, N, o, si)
    TT.append(ic["T"].reshape(ny, nx) / si["TUnit"])
    YY.append(ic["F"].reshape(ny, nx))
    for t_si, d in foam_times(fd):
        f = foam_fields(d, si, nx, ny)
        ts.append(t_si / si["tau"])
        TT.append(f["T"])
        YY.append(f["Y"])
    if len(ts) > 1:
        fI, fT, fp = wrinkle_series(TT, YY, ts, h, Tmid)
        res["runs"]["foam"] = {"t": ts, "aI": fI.tolist(), "aT": fT.tolist(),
                               "phase": fp.tolist()}
        # theta field difference at the fit end
        te = fit[1]
        kf = int(np.argmin(np.abs(np.array(ts) - te)))
        kd = int(round(te / fdt))
        thd = (fr["temp"][kd].astype(float)) / ((o.sigma - 1) * Tu)
        thf = (TT[kf] - Tu) / ((o.sigma - 1) * Tu)
        res["theta"] = field_diff(thd, thf, m["x"], m["y"])
        res["theta_t"] = ts[kf]
        wf = os.path.join(fd, "wall.txt")
        if os.path.exists(wf):
            res["foam_wall"] = open(wf).read().strip()
    for name, rd, Ne, nye, sg, dth in extra_dns:
        he = dth / Ne
        nxe = int(math.ceil((spec["box"]["upstream"] + spec["box"]["downstream"])
                            * Ne - 1e-9)) + 2 * spec["box"]["sink"]
        f2, fdt2, _ = dns_frames(os.path.join(rd, "flame"), nxe, nye, ("temp", "fuel"))
        TuE = Tu
        t2 = fdt2 * np.arange(f2["temp"].shape[0])
        a1, a2, p2 = wrinkle_series(f2["temp"].astype(float) + TuE, f2["fuel"], t2,
                                    he, 0.5 * (1 + sg) * TuE)
        res["runs"][name] = {"t": t2.tolist(), "aI": a1.tolist(), "aT": a2.tolist(),
                             "sigma": sg, "lambda_dD": nye * he,
                             "theory": dl_theory(sg, 2 * math.pi / (nye * he))}
    for name, rr in res["runs"].items():
        tt = np.array(rr["t"])
        tmax = tt.max()
        ws = wins + ([(fit[0], tmax)] if tmax > fit[1] + 1 else [])
        if "sigma" in rr:
            ws = [(8.0, tmax), (16.0, tmax)]
        rr["rates"] = {f"{w[0]:g}-{w[1]:g}": {"aI": growth(tt, rr["aI"], w),
                                              "aT": growth(tt, rr["aT"], w)}
                       for w in ws}
        a = np.array(rr["aI"])
        rr["local"] = [(float(tt[k]), float(math.log(a[k + 4] / a[k]) / (tt[k + 4] - tt[k])))
                       for k in range(0, len(tt) - 4, 4)]
    return res


def table_wrinkle(r):
    th = r["theory"]
    L = [f"{r['case']}  ({r['nx']} x {r['ny']} cells, h = delta_th/{r['N']:g}, "
         f"lambda = {r['lambda_dD']:.3f} delta_D)",
         f"theory sigma 5: omega_DL {th['omega_DL']:.5f}, B1 {th['B1']:.5f}, "
         f"lambda_c {th['lambda_c']:.3f} delta_D; inviscid DL {th['inviscid']:.5f}, "
         f"Matalon-Matkowsky {th['mm']:.5f} /tau", ""]
    for name, rr in r["runs"].items():
        extra = ""
        if "theory" in rr:
            t2 = rr["theory"]
            extra = (f"  [sigma {rr['sigma']:g}, lambda {rr['lambda_dD']:.2f}: inviscid "
                     f"{t2['inviscid']:.5f}, MM {t2['mm']:.5f}, lambda_c {t2['lambda_c']:.2f}]")
        L.append(f"{name}{extra}")
        for w, v in rr["rates"].items():
            L.append(f"   fit [{w}]  s(x_I) {v['aI'][0]:+.5f} (rms ln {v['aI'][1]:.3f}, "
                     f"n {v['aI'][2]})  s(x_T) {v['aT'][0]:+.5f} (rms {v['aT'][1]:.3f})")
        L.append("   local s (x_I, per 2 tau from t): " + " ".join(
            f"{t:g}:{s:.3f}" for t, s in rr["local"]))
        L.append("   a_I(t) every 2 tau: " + " ".join(
            f"{a:.4f}" for a in rr["aI"][::4]))
    if "theta" in r:
        v = r["theta"]
        L.append(f"theta dns - foam at t {r['theta_t']:g}: Linf {v['linf']:.3e} "
                 f"L2 {v['l2']:.3e} at x {v['at_x']:.2f} y {v['at_y']:.2f}")
    if "foam_wall" in r:
        L.append(f"foam wall: {r['foam_wall']}")
    return "\n".join(L)
