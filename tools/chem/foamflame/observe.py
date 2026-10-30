"""Observables of both solvers from their outputs, in box units, by ONE code
path (the dns runner's own definitions, gpu/runner_flame.hip measure()):

  S_L    consumption speed: integral(omega dx) / (rho Y) at the fresh
         reference column iRef = sink + round(2 delta_th/h); time series:
         dns <out>_stats.csv SL, foam volIntegrate(Qdot)/q over the probe's
         rho Y; mean over the window, and its two halves
  dth    (Tb - Tu)/max dT/dx, parabolic peak of the face slopes (nominal
         Tb - Tu = (sigma - 1) Tu); x_f the first mid-T crossing (linear)
  Tb     mean T over x - x_f in [2.5, 3.5] delta_th, / (sigma Tu); Tb_far
         over [8, 9] (exact energy: both -(sigma^2 - 1) S^2/(2 cp sigma Tu)
         = -9.6e-5 at M 0.01)
  H      total-enthalpy gain (H_b - H_u)/q in the flame frame between the
         ref column and the burnt window (near / far): 0 for a conservative
         scheme
  dp     p(iRef) - mean p over the near burnt window; against low-Mach RH
         (sigma - 1) S_L^2 (oracle units) and against the measured-state RH
         (rho_ref S_c)^2 (1/rho_b - 1/rho_ref)
  Linf   profiles aligned at x_f, |x - x_f| <= 5 delta_th: theta = (T -
         Tu)/(Tb - Tu), Y, (u - u_in)/(sigma - 1) (oracle: theta), omega /
         max omega_oracle; against the oracle at the solver's own cells, and
         dns vs foam (foam cubic-interpolated onto the dns offsets; the
         interpolation floor is the same operation on the oracle)
"""
import csv
import glob
import json
import math
import os

import numpy as np

from . import cases, writer

PROFILE_HALF = 5.0      # delta_th, the L-infinity window
BURNT = (2.5, 3.5)      # delta_th behind x_f: the burnt state the flame makes
BURNT_FAR = (8.0, 9.0)  # ... and 6 delta_th further (dns heats there, below)


# --- loading ------------------------------------------------------------------

def _csv(path):
    with open(path) as f:
        r = csv.reader(f)
        head = next(r)
        rows = np.array([[float(v) for v in row] for row in r])
    return {h: rows[:, k] for k, h in enumerate(head)}


def load_dns(rundir):
    st = _csv(os.path.join(rundir, "flame_stats.csv"))
    pr = _csv(os.path.join(rundir, "flame_profiles.csv"))
    dumps = []
    for t in np.unique(pr["t"]):
        k = pr["t"] == t
        dumps.append((t, {q: pr[q][k] for q in ("x", "T", "rho", "u", "p", "Y",
                                                "omega")}))
    return {"series": {"t": st["t"], "SL": st["SL"]}, "dumps": dumps,
            "stats": st}


def load_foam(rundir, si, m, species):
    tau, dD = si["tau"], si["dD"]
    dumps = []
    for d in glob.glob(os.path.join(rundir, "[0-9]*")):
        t = float(os.path.basename(d))
        if t <= 0 or not os.path.exists(os.path.join(d, "Qdot")):
            continue
        T = writer.read_scalar_field(os.path.join(d, "T"))
        p = writer.read_scalar_field(os.path.join(d, "p"))
        u = writer.read_vector_x(os.path.join(d, "U"))
        Y = writer.read_scalar_field(os.path.join(d, species[0]))
        Q = writer.read_scalar_field(os.path.join(d, "Qdot"))
        rho = p / (si["R"] * T)            # hePsiThermo: rho = psi p
        dumps.append((t / tau, {
            "x": m["x"], "T": T / si["TUnit"], "rho": rho / si["rho_u"],
            "u": u / si["SL"], "p": p / si["pUnit"], "Y": Y,
            "omega": Q / si["q"] / si["omegaUnit"]}))
    dumps.sort(key=lambda d: d[0])
    # burn integral and the reference probe, every 10 steps
    b = np.loadtxt(os.path.join(rundir, "postProcessing/burn/0/volFieldValue.dat"))
    pr = {}
    for f in ("rho", "F"):
        a = np.loadtxt(os.path.join(rundir, "postProcessing/ref/0", f))
        pr[f] = a
    n = min(len(b), len(pr["rho"]), len(pr["F"]))
    area = (m["h"] * dD) ** 2
    ry = pr["rho"][:n, 1] * pr["F"][:n, 1]
    SL = b[:n, 1] / si["q"] / area / ry / si["SL"]
    wall = None
    wf = os.path.join(rundir, "wall.txt")
    if os.path.exists(wf):
        wall = float(open(wf).read().split()[0])
    clock = None
    log = os.path.join(rundir, "log.reactingFoam")
    if os.path.exists(log):
        for line in open(log):
            if line.startswith("ExecutionTime"):
                clock = (float(line.split()[2]), float(line.split()[6]))
    return {"series": {"t": b[:n, 0] / tau, "SL": SL}, "dumps": dumps,
            "wall": wall, "exec_clock": clock}


# --- per-profile measures (gpu/runner_flame.hip measure()) ---------------------

def measure(prof, spec, o, m):
    sigma = o.sigma
    Tu = 1.0 / (1.4 * spec["model"]["mach"] ** 2)
    Tmid = 0.5 * (Tu + sigma * Tu)
    dTf = (sigma - 1) * Tu
    T, h = prof["T"], m["h"]
    k = np.nonzero((T[:-1] < Tmid) & (T[1:] >= Tmid))[0]
    xf = (k[0] + 0.5 + (Tmid - T[k[0]]) / (T[k[0] + 1] - T[k[0]])) * h
    sl = np.diff(T) / h
    km = int(np.argmax(sl))
    peak = sl[km]
    if 0 < km < len(sl) - 1:
        a, b, c = sl[km - 1], sl[km], sl[km + 1]
        den = a - 2 * b + c
        if den < 0:
            peak = b - (c - a) ** 2 / (8 * den)
    i = m["iRef"]
    ry = prof["rho"][i] * prof["Y"][i]
    xi = (prof["x"] - xf) / o.dth
    bw = (xi >= BURNT[0]) & (xi <= BURNT[1])
    fw = (xi >= BURNT_FAR[0]) & (xi <= BURNT_FAR[1])
    pb, rb = prof["p"][bw].mean(), prof["rho"][bw].mean()
    SLp = prof["omega"].sum() * h / ry
    dp = prof["p"][i] - pb
    rr = prof["rho"][i]
    rh = (rr * SLp) ** 2 * (1 / rb - 1 / rr)
    return {"xf": xf, "dth": dTf / peak, "SL": SLp,
            "Tb": prof["T"][bw].mean() / (sigma * Tu),
            "Tb_far": prof["T"][fw].mean() / (sigma * Tu),
            "Tmax": T.max() / (sigma * Tu),
            "dp_lowMach": dp / (sigma - 1), "dp_RH": dp / rh,
            "uRef": prof["u"][i],
            # plateau states for the total-enthalpy balance (summarize)
            "_ref": (prof["T"][i], prof["u"][i], prof["Y"][i]),
            "_b": tuple(prof[q][bw].mean() for q in ("T", "u", "Y")),
            "_far": tuple(prof[q][fw].mean() for q in ("T", "u", "Y"))}


def enthalpy_defect(mm, uf, sigma, Tu, key="_b"):
    """(H_b - H_u)/q in the frame of the flame (lab speed uf), H = cp T +
    u_rel^2/2 + q Y, cp = 3.5: exactly 0 between plateaus for a conservative
    scheme on a steadily moving flame (heat, viscous and diffusive fluxes
    vanish there; viscous work drops out too)."""
    cp = 3.5
    q = cp * (sigma - 1) * Tu
    (Tu_, uu, Yu), (Tb, ub, Yb) = mm["_ref"], mm[key]
    Hu = cp * Tu_ + 0.5 * (uu - uf) ** 2 + q * Yu
    Hb = cp * Tb + 0.5 * (ub - uf) ** 2 + q * Yb
    return (Hb - Hu) / q


def _oracle_cols(o, xi_dD, uin):
    pr = o.profile(xi_dD)
    th = (pr["T"] - 1) / (o.sigma - 1)
    return {"theta": th, "Y": pr["Y"], "u": th, "omega": pr["omega"]}


def _solver_cols(prof, o, spec, uin):
    Tu = 1.0 / (1.4 * spec["model"]["mach"] ** 2)
    th = (prof["T"] - Tu) / ((o.sigma - 1) * Tu)
    return {"theta": th, "Y": prof["Y"], "u": (prof["u"] - uin) / (o.sigma - 1),
            "omega": prof["omega"]}


def linf_oracle(prof, xf, o, spec):
    uin = spec["inflow"]
    xi = prof["x"] - xf                       # delta_D
    w = np.abs(xi) <= PROFILE_HALF * o.dth
    oc = _oracle_cols(o, xi[w], uin)
    sc = _solver_cols(prof, o, spec, uin)
    wmax = o.profile(np.linspace(-1, 1, 4001))["omega"].max()
    return {q: float(np.abs(sc[q][w] - oc[q]).max() / (wmax if q == "omega" else 1))
            for q in oc}


def _cubic(xs, fs, x):
    """4-point Lagrange interpolation on a uniform grid xs."""
    h = xs[1] - xs[0]
    j = np.clip(np.floor((x - xs[0]) / h).astype(int) - 1, 0, len(xs) - 4)
    s = (x - xs[j]) / h
    w0 = -(s - 1) * (s - 2) * (s - 3) / 6
    w1 = s * (s - 2) * (s - 3) / 2
    w2 = -s * (s - 1) * (s - 3) / 2
    w3 = s * (s - 1) * (s - 2) / 6
    return w0 * fs[j] + w1 * fs[j + 1] + w2 * fs[j + 2] + w3 * fs[j + 3]


def linf_pair(pa, xa, pb, xb, o, spec):
    """dns (a) vs foam (b): b interpolated onto a's offsets; and the floor."""
    uin = spec["inflow"]
    xia, xib = pa["x"] - xa, pb["x"] - xb
    w = np.abs(xia) <= PROFILE_HALF * o.dth
    ca, cb = _solver_cols(pa, o, spec, uin), _solver_cols(pb, o, spec, uin)
    ob = _oracle_cols(o, xib, uin)
    oa = _oracle_cols(o, xia[w], uin)
    wmax = o.profile(np.linspace(-1, 1, 4001))["omega"].max()
    d, fl = {}, {}
    for q in ca:
        s = wmax if q == "omega" else 1
        d[q] = float(np.abs(ca[q][w] - _cubic(xib, cb[q], xia[w])).max() / s)
        fl[q] = float(np.abs(oa[q] - _cubic(xib, ob[q], xia[w])).max() / s)
    return d, fl


# --- per run ------------------------------------------------------------------

def _window(ser, win):
    t, v = ser["t"], ser["SL"]
    a, b = win
    mid = 0.5 * (a + b)
    k = (t >= a) & (t <= b)
    return {"mean": float(v[k].mean()),
            "half1": float(v[(t >= a) & (t <= mid)].mean()),
            "half2": float(v[(t > mid) & (t <= b)].mean()), "n": int(k.sum())}


def summarize(run, spec, o, m):
    win = spec["window"]
    ds = [(t, p) for t, p in run["dumps"] if win[0] - 1e-6 <= t <= win[1] + 1e-3]
    ms = [measure(p, spec, o, m) for _, p in ds]
    li = [linf_oracle(p, mm["xf"], o, spec) for (_, p), mm in zip(ds, ms)]
    out = {"SL_series": _window(run["series"], win), "ndumps": len(ds)}
    for key in ms[0]:
        if not key.startswith("_"):
            out[key] = float(np.mean([mm[key] for mm in ms]))
    for key in ("dp_RH", "Tb", "uRef"):
        out[key + "_std"] = float(np.std([mm[key] for mm in ms]))
    out["linf_oracle"] = {q: float(np.mean([l[q] for l in li])) for q in li[0]}
    out["linf_oracle_max"] = {q: float(np.max([l[q] for l in li])) for q in li[0]}
    uf = (ms[-1]["xf"] - ms[0]["xf"]) / (ds[-1][0] - ds[0][0])
    out["xf_drift"] = float(uf)
    Tu = 1.0 / (1.4 * spec["model"]["mach"] ** 2)
    hd = [enthalpy_defect(mm, uf, o.sigma, Tu) for mm in ms]
    out["H_defect"] = float(np.mean(hd))
    out["H_defect_std"] = float(np.std(hd))
    out["H_defect_far"] = float(np.mean(
        [enthalpy_defect(mm, uf, o.sigma, Tu, "_far") for mm in ms]))
    return out, ds, ms


def order(q):
    """Observed order and Richardson value of a 3-rung ladder (ratio 2)."""
    a, b, c = q
    if (a - b) * (b - c) <= 0:
        return {"p": None, "rich": None, "rich2": c + (c - b) / 3}
    p = math.log2((a - b) / (b - c))
    return {"p": p, "rich": c + (c - b) / (2 ** p - 1), "rich2": c + (c - b) / 3}


def compare(spec, o, si, root, dns_names, foam_runs):
    res = {"case": spec["title"], "Lambda": o.Lambda, "dth_oracle": o.dth,
           "si": si, "runs": {}, "pairs": {}}
    raw = {}
    geo = cases.GEOMETRY[spec["geometry"]][0]
    for name, N in zip(dns_names, spec["ladder"]):
        d = os.path.join(root, name)
        if not os.path.exists(os.path.join(d, "flame_stats.csv")):
            continue
        m = geo(spec, N, o.dth)
        run = load_dns(d)
        s, ds, ms = summarize(run, spec, o, m)
        s["N"] = N
        s["solver"] = "dns"
        s["dth_stats"] = float(np.mean(run["stats"]["dth"][
            (run["stats"]["t"] >= spec["window"][0])]))
        js = os.path.join(d, "flame.json")
        if os.path.exists(js):
            with open(js) as f:
                s["ms_per_step"] = json.load(f).get("msPerStep")
        s["steps"] = int(len(run["stats"]["t"]) and
                         round(run["stats"]["t"][-1] / np.median(run["stats"]["dt"])))
        rec = os.path.join(root, "job_dns.record.json")
        if os.path.exists(rec):
            with open(rec) as f:
                jr = json.load(f)
            for st_, step in zip(jr["spec"]["steps"], jr["steps"]):
                if d + os.sep in " ".join(st_["args"]) + os.sep:
                    s["wall"] = step["wall_s"]
        res["runs"][name] = s
        raw[name] = (ds, ms)
    for name, N, st in foam_runs:
        d = os.path.join(root, name)
        if not os.path.exists(os.path.join(d, "postProcessing")):
            continue
        m = geo(spec, N, o.dth)
        run = load_foam(d, si, m, spec["species"])
        if not run["dumps"]:
            continue
        s, ds, ms = summarize(run, spec, o, m)
        s.update({"N": N, "solver": "foam", "variant": st["variant"],
                  "dt": st["dt"], "n_outer": st["n_outer"],
                  "scheme": f"Y {st['yscheme']}, h {st['hscheme']}, U linear",
                  "wall": run["wall"], "exec_clock": run["exec_clock"]})
        res["runs"][name] = s
        raw[name] = (ds, ms)
        dn = f"dns_c{N:g}"
        if dn in raw:
            dd, dm = raw[dn]
            tf = {round(t, 6): (p, mm) for (t, p), mm in zip(ds, ms)}
            li, fl = [], []
            for (t, p), mm in zip(dd, dm):
                hit = [v for k, v in tf.items() if abs(k - t) < 0.02]
                if not hit:
                    continue
                a, b = linf_pair(p, mm["xf"], hit[0][0], hit[0][1]["xf"], o, spec)
                li.append(a)
                fl.append(b)
            if li:
                res["pairs"][f"{dn}~{name}"] = {
                    "linf": {q: float(np.mean([x[q] for x in li])) for q in li[0]},
                    "floor": {q: float(np.mean([x[q] for x in fl])) for q in fl[0]},
                    "n": len(li)}
    # ladders
    lad = {}
    groups = {"dns": [f"dns_c{N:g}" for N in spec["ladder"]]}
    for v in spec["foam"]["variants"]:
        groups[f"foam_{v}"] = [f"foam_{v}_c{N:g}" for N in spec["ladder"]]
    for gname, names in groups.items():
        if all(n in res["runs"] for n in names):
            r = [res["runs"][n] for n in names]
            lad[gname] = {
                "SL": order([x["SL_series"]["mean"] for x in r]),
                "dth": order([x["dth"] for x in r]),
                "Tb": order([x["Tb"] for x in r]),
                "linf_theta": order([x["linf_oracle"]["theta"] for x in r])}
    res["ladders"] = lad
    return res


def table(res):
    L = []
    th = res["dth_oracle"]
    L.append(f"{res['case']}  (oracle S_L 1, dth/dD {th:.5f}, Lambda {res['Lambda']:.10g})")
    L.append(f"{'run':24s} {'S_L':>9s} {'halves':>19s} {'dth/or-1':>10s} "
             f"{'Tb/sTu-1':>10s} {'dp/(s-1)':>9s} {'dp/RH-1':>9s} "
             f"{'Linf th':>8s} {'Y':>8s} {'u':>8s} {'om':>8s} {'drift':>8s}")
    for name, r in res["runs"].items():
        s = r["SL_series"]
        li = r["linf_oracle"]
        L.append(f"{name:24s} {s['mean']:9.6f} {s['half1']:9.6f}/{s['half2']:9.6f} "
                 f"{r['dth'] / th - 1:+10.3e} {r['Tb'] - 1:+10.3e} "
                 f"{r['dp_lowMach']:9.5f} {r['dp_RH'] - 1:+9.2e} "
                 f"{li['theta']:8.2e} {li['Y']:8.2e} {li['u']:8.2e} {li['omega']:8.2e} "
                 f"{r['xf_drift']:+8.4f}")
    L.append("")
    L.append(f"{'run':24s} {'H_defect':>10s} {'+-':>8s} {'H_far':>10s} "
             f"{'Tbfar-1':>10s} {'dp/RH sd':>9s} "
             f"{'Tb sd':>8s} {'u_ref sd':>9s} {'scheme / cost':s}")
    for name, r in res["runs"].items():
        if r["solver"] == "dns":
            cost = (f"dns central + Y MC-MUSCL, ~{r.get('steps', 0)} steps, "
                    f"{r.get('ms_per_step', float('nan')):.3f} ms/step, "
                    f"wall {r.get('wall', float('nan')):.1f} s (GPU)")
        else:
            ec = r.get("exec_clock") or (float("nan"), float("nan"))
            cost = (f"{r['scheme']}, dt {r['dt']:g} tau, nOuter {r['n_outer']}, "
                    f"exec {ec[0]:.1f} s / clock {ec[1]:.0f} s (1 core)")
        L.append(f"{name:24s} {r['H_defect']:+10.3e} {r['H_defect_std']:8.1e} "
                 f"{r['H_defect_far']:+10.3e} {r['Tb_far'] - 1:+10.3e} "
                 f"{r['dp_RH_std']:9.2e} {r['Tb_std']:8.1e} {r['uRef_std']:9.2e} {cost}")
    L.append("")
    L.append("dns vs foam at equal mesh: Linf (theta, Y, u, omega) [interp floor]")
    for k, p in res["pairs"].items():
        a, f = p["linf"], p["floor"]
        L.append(f"  {k:36s} " + " ".join(
            f"{a[q]:8.2e}[{f[q]:7.1e}]" for q in ("theta", "Y", "u", "omega")))
    L.append("")
    L.append("ladders: observed order p, Richardson (p) and (p = 2)")
    for gname, d in res["ladders"].items():
        for q, v in d.items():
            ps = "  n/a " if v["p"] is None else f"{v['p']:6.3f}"
            rs = "     n/a    " if v["rich"] is None else f"{v['rich']:.6g}"
            L.append(f"  {gname:12s} {q:10s} p {ps}  R {rs:>12s}  R2 {v['rich2']:.6g}")
    return "\n".join(L)
