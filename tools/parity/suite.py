#!/usr/bin/env python3
"""dns vs OpenFOAM parity suite driver. Spec: tools/parity/DESIGN.md.

  suite.py list
  suite.py calibrate CASE [--tier A|B] [--solver dns|foam]
  suite.py run CASE --tier A|B|Bc [--solver dns|foam] [--level k] [--ref] [--np N] [--force]
               [--timeout S] [--tag T [--t-cap|--mach|--dns-ny|--zone-r ...]]
  suite.py adopt CASE --tier A --level k --from DIR --dry FILE [--ft F] [--rev R]
  suite.py stale CASE --tier T --run NAME --reason TEXT
  suite.py report

Every timed command runs as `flock LOCK sh -c 'echo __T0__; exec timeout T cmd'`:
one timed job on the machine at a time, a hard timeout, and the clock starts
when the lock is held (__T0__), not when the job was queued. Each stdout line
is stamped with monotonic seconds since __T0__; dns progress lines (t=...) and
foam `Time =` lines then map simulation time to wall time.
PARITY_LOCK_HELD=1 in the environment: the caller already holds that lock
(e.g. the session's `gpurun`, which flocks the same file), so the suite does
not take it again -- a nested flock on the same file deadlocks.
"""
import argparse
import csv
import datetime
import glob
import json
import math
import os
import re
import shlex
import shutil
import subprocess
import sys
import threading
import time
import tomllib

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE)
import foamcase as fc  # noqa: E402
import geom  # noqa: E402

RUNS = os.path.join(ROOT, "runs", "parity")
DNS = os.path.join(ROOT, "build", "dns")
D_BOX = math.pi / 4          # dns compressible D (box units, U = 1)
CFG_PATH = os.path.join(HERE, "cases.toml")
# foam application per tier: A same method, B incompressible steady (the M -> 0
# leg), Bc compressible steady at the dns Mach
FOAM_APP = {"A": "rhoPimpleFoam", "B": "simpleFoam", "Bc": "rhoSimpleFoam"}
TIERS = tuple(FOAM_APP)


def die(msg, rc=1):
    print(f"suite: {msg}", file=sys.stderr)
    sys.exit(rc)


def load_cfg():
    with open(CFG_PATH, "rb") as f:
        return tomllib.load(f)


def now():
    return datetime.datetime.now().isoformat(timespec="seconds")


# --- provenance ---------------------------------------------------------------

_FOAM_ENV = None


def foam_env(cfg):
    """The OpenFOAM environment, captured once (no per-command sourcing cost)."""
    global _FOAM_ENV
    if _FOAM_ENV is None:
        out = subprocess.run(["bash", "-c", f"source {cfg['suite']['foam_bashrc']} "
                              ">/dev/null 2>&1; env -0"], capture_output=True, check=True)
        _FOAM_ENV = dict(kv.split("=", 1) for kv in out.stdout.decode().split("\0") if "=" in kv)
    return _FOAM_ENV


def provenance(cfg):
    st = os.stat(DNS)
    rev = subprocess.run(["git", "-C", ROOT, "rev-parse", "--short", "HEAD"],
                         capture_output=True, text=True).stdout.strip()
    dirty = subprocess.run(["git", "-C", ROOT, "status", "--porcelain", "--", "src",
                            "CMakeLists.txt"], capture_output=True, text=True).stdout.strip()
    env = foam_env(cfg)
    return dict(dns_binary=DNS,
                dns_mtime=datetime.datetime.fromtimestamp(st.st_mtime).isoformat(timespec="seconds"),
                git_rev=rev, src_dirty=bool(dirty),
                openfoam=f"OpenFOAM-{env.get('WM_PROJECT_VERSION', '?')}",
                host_cpu="i5-14600KF", gpu="RX 7900 XTX", recorded=now())


# --- timed execution ----------------------------------------------------------

def gpu_power_w():
    try:
        out = subprocess.run(["rocm-smi", "--showpower", "--json"], capture_output=True,
                             text=True, timeout=5).stdout
        d = json.loads(out[out.index("{"):])
        for card in d.values():
            for k, v in card.items():
                if "Power" in k:
                    return float(v)
    except Exception:
        pass
    return float("nan")


def timed(cfg, argv, log_path, timeout_s, cwd, env=None, power=False):
    """Run argv under flock + timeout; stamp every output line. Returns
    dict(rc, wall, lines=[(t, text)], power=[(t, W)])."""
    lock = [] if os.environ.get("PARITY_LOCK_HELD") == "1" else ["flock", cfg["suite"]["lock"]]
    wrapped = lock + ["sh", "-c", f'echo __T0__; exec timeout {int(timeout_s)} "$@"', "sh", *argv]
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    lines, samples = [], []
    t0 = [None]
    stop = threading.Event()

    def sampler():
        while not stop.is_set():
            if t0[0] is not None:
                samples.append((time.monotonic() - t0[0], gpu_power_w()))
            stop.wait(1.0)

    th = threading.Thread(target=sampler, daemon=True) if power else None
    with open(log_path, "w") as log:
        log.write(f"# {now()} cwd {cwd}\n# argv {shlex.join(wrapped)}\n")
        p = subprocess.Popen(wrapped, cwd=cwd, env=env, stdout=subprocess.PIPE,
                             stderr=subprocess.STDOUT, text=True, bufsize=1)
        if th:
            th.start()
        for line in p.stdout:
            tnow = time.monotonic()
            if t0[0] is None and line.startswith("__T0__"):
                t0[0] = tnow
                log.write(f"0.000 __T0__ {now()}\n")
                continue
            rel = tnow - t0[0] if t0[0] is not None else -1.0
            lines.append((rel, line.rstrip("\n")))
            log.write(f"{rel:.3f} {line}")
        rc = p.wait()
        wall = time.monotonic() - t0[0] if t0[0] is not None else 0.0
        stop.set()
        if th:
            th.join()
        log.write(f"# rc={rc} wall={wall:.3f}\n")
    return dict(rc=rc, wall=wall, lines=lines, power=samples)


def energy(power, t1):
    """Trapezoid of (t, W) over [0, t1]; J."""
    pts = [(t, w) for t, w in power if math.isfinite(w)]
    if len(pts) < 2:
        return float("nan")
    ts = np.array([p[0] for p in pts])
    ws = np.array([p[1] for p in pts])
    t1 = min(t1, ts[-1])
    m = ts <= t1
    tt, ww = list(ts[m]), list(ws[m])
    if tt and tt[-1] < t1:
        tt.append(t1)
        ww.append(float(np.interp(t1, ts, ws)))
    return float(np.trapezoid(ww, tt)) if len(tt) > 1 else 0.0


# --- simulation-time -> wall-time ---------------------------------------------

RE_DNS_PROG = re.compile(r"^t=([0-9.eE+-]+)/")
RE_FOAM_TIME = re.compile(r"^Time = ([0-9.eE+-]+)\s*$")


def wall_map(lines, kind):
    """(sim, wall) pairs from stamped lines; dns gets a t=0 point at the
    extrapolated start of the time loop (in-process setup)."""
    pat = RE_DNS_PROG if kind == "dns" else RE_FOAM_TIME
    pts = []
    for w, s in lines:
        m = pat.match(s)
        if m:
            pts.append((float(m.group(1)), w))
    if not pts:
        return [], float("nan")
    setup = float("nan")
    if kind == "dns":
        if len(pts) >= 2:
            (s1, w1), (s2, w2) = pts[0], pts[1]
            setup = max(w1 - s1 * (w2 - w1) / (s2 - s1), 0.0)
        else:
            setup = 0.0
        pts.insert(0, (0.0, setup))
    else:
        setup = pts[0][1]
    return pts, setup


def wall_at(pts, sim):
    s = np.array([p[0] for p in pts])
    w = np.array([p[1] for p in pts])
    if sim >= s[-1]:
        return float(w[-1])
    return float(np.interp(sim, s, w))


# --- settle rules ---------------------------------------------------------------

def settle_steady(series, tol, scales):
    """series: {name: (t array, q array)}; scales: {name: abs scale}. t_settle =
    first time after which |q - q_final| <= tol * scale for every observable,
    for good. q_final = mean over the last 5 % of the record (>= 1 sample), so
    a sampled residual wobble does not make the last sample the arbiter.
    Returns (t_settle, finals)."""
    finals, ts = {}, []
    for k, (t, q) in series.items():
        tail = t >= t[-1] - 0.05 * (t[-1] - t[0])
        qf = float(q[tail].mean())
        finals[k] = qf
        sc = scales.get(k) or abs(qf)
        bad = np.nonzero(np.abs(q - qf) > tol * sc)[0]
        ts.append(float(t[0]) if len(bad) == 0 else
                  float(t[min(bad[-1] + 1, len(t) - 1)]))
    return max(ts), finals


def crossings(t, x):
    """Upward zero crossings, linear interpolation."""
    i = np.nonzero((x[:-1] < 0) & (x[1:] >= 0))[0]
    return t[i] - x[i] * (t[i + 1] - t[i]) / (x[i + 1] - x[i])


def _harmonic_basis(tt, w, H):
    cols = [np.ones_like(tt)]
    for h in range(1, H + 1):
        cols += [np.cos(h * w * tt), np.sin(h * w * tt)]
    return np.array(cols).T


def periodic_series(t, cd, cl, nwin=3):
    """Window = nwin shedding periods, delimited by upward crossings of
    (cl - final mean). In each window a least-squares harmonic fit (mean + 3
    harmonics, frequency free) of cl gives St = f L/U, the cl rms, and, with
    that frequency fixed, a 4-harmonic fit of cd gives the windowed cd mean.
    Zero-crossing periods alias against dns' 200-step stats cadence (~8
    samples per period at the coarse level: a 5-period beat of 0.16 % in a
    3-period St); the fit uses every sample and does not.
    Returns (tc, St, cdm, clrms) at the window ends."""
    from scipy.optimize import least_squares
    t, iu = np.unique(t, return_index=True)
    cd, cl = cd[iu], cl[iu]
    tail = t >= t[0] + 0.7 * (t[-1] - t[0])
    c = crossings(t, cl - cl[tail].mean())
    out = []
    for k in range(nwin, len(c)):
        a, b = c[k - nwin], c[k]
        m = (t >= a) & (t <= b)
        if m.sum() < 12:
            continue
        tt, y = t[m] - a, cl[m]

        def res(p):
            A = _harmonic_basis(tt, p[0], 3)
            co = np.linalg.lstsq(A, y, rcond=None)[0]
            return A @ co - y
        w0 = 2 * math.pi * nwin / (b - a)
        w = least_squares(res, [w0], x_scale=[0.01 * w0]).x[0]
        A = _harmonic_basis(tt, w, 3)
        co = np.linalg.lstsq(A, y, rcond=None)[0]
        clr = math.sqrt(0.5 * float(np.sum(co[1:] ** 2)))
        A4 = _harmonic_basis(tt, w, 4)
        cdm = float(np.linalg.lstsq(A4, cd[m], rcond=None)[0][0])
        out.append((b, w / (2 * math.pi), cdm, clr))
    if not out:
        return None
    return tuple(np.array(v) for v in zip(*out))


# --- case model -----------------------------------------------------------------

class Case:
    def __init__(self, cfg, name):
        if name not in cfg["cases"]:
            die(f"unknown case '{name}' (have: {', '.join(cfg['cases'])})")
        self.cfg, self.name, self.c = cfg, name, dict(cfg["cases"][name])
        self.kind = self.c["kind"]
        self.tag = ""          # calibration runs: separate run dir suffix

    # convective unit (L/U) in dns box time: L = D (cylinder, rotor), c = D/2 (section)
    def conv_box(self):
        return D_BOX / 2 if self.kind == "section" else D_BOX

    def mach(self, tier):
        return self.c.get("b_mach", self.c["mach"]) if tier == "B" else self.c["mach"]

    def u_scheme(self):
        return self.c.get("u_scheme", "linearUpwind")

    def zone_r(self):
        return self.c.get("zone_r")

    def refine_r(self):
        """Refinement-cylinder radius: the case's own zone_r unless a zone
        study moved the zone (then the mesh keeps the case's refinement)."""
        return self.c.get("refine_r", self.c.get("zone_r"))

    def levels(self, solver, tier, ref):
        key = {"dns": "dns", "foam": "foam"}[solver]
        if ref:
            v = self.c.get(f"{key}_ref_{tier}") or self.c.get(f"{key}_ref")
            if v is None:
                die(f"{self.name}: no {key} reference level")
            return [v]
        return self.c[f"{key}_levels_{tier}"] if f"{key}_levels_{tier}" in self.c \
            else self.c[f"{key}_levels"]

    def run_dir(self, tier, solver, label, ref, np_=None):
        tag = f"{solver}_{label}" + (f"_np{np_}" if np_ and np_ != self.cfg["suite"]["np"] else "")
        if self.tag:
            return os.path.join(RUNS, "calib", tier, self.name, f"{tag}_{self.tag}")
        base = os.path.join(RUNS, "ref", tier) if ref else os.path.join(RUNS, tier)
        return os.path.join(base, self.name, tag)

    # ---- dns ----
    def dns_ny(self, lvl):
        return int(lvl[0] if isinstance(lvl, list) else lvl)

    def dns_argv(self, tier, lvl, rundir):
        c = self.c
        ny = self.dns_ny(lvl)
        n = int(round(ny * c.get("aspect", 1.0)))
        L = c.get("aspect", 1.0) * c["dr"] * D_BOX
        t_cap_box = (c.get("b_t_cap", c["t_cap"]) if tier == "B" else c["t_cap"]) * self.conv_box()
        ft = max((t_cap_box - c.get("ramp", 2.0)) / L, 1e-3)
        M = self.mach(tier)
        if self.kind == "cylinder":
            a = ["flow", "--body", "cylinder", "--physics", "compressible",
                 "--dr", str(c["dr"]), "--re", str(c["re"]), "--mach", str(M),
                 "--n", str(n), "--ny", str(ny), "--ft", f"{ft:.6f}"]
            if "Lw" in c["observables"]:
                a += ["--fdt", str(c["lw_every"] * D_BOX), "--emit", "stats", "--emit", "bin"]
            else:
                a += ["--emit", "stats"]
        elif self.kind == "section":
            stl = os.path.join(rundir, "body.stl")
            geom.write_section_stl(stl, geom.section_polygon(c["aoa"], geom.section_unit(
                c["thickness"], c["crest"])))
            a = ["flow", "--physics", "compressible", "--body", stl,
                 "--scale", f"{math.pi / 8:.10f}", "--n", str(n), "--ny", str(ny),
                 "--nz", "4", "--dr", str(c["dr"]), "--re", f"{2 * c['re']:.6f}",
                 "--mach", str(M), "--ft", f"{ft:.6f}", "--emit", "stats"]
        elif self.kind == "rotor":
            stl = os.path.join(rundir, "body.stl")
            geom.write_rotor_stl(stl, c["blade"])
            minf = c["tip_mach"] * c["J"] / math.pi
            a = ["flow", "--physics", "compressible", "--body", stl, "--robust",
                 "--n", str(n), "--ny", str(ny), "--nz", str(ny), "--dr", str(c["dr"]),
                 "--focal", "auto", "--focal-ratio", str(lvl[1]), "--focal-blend", str(lvl[2]),
                 "--rotor-mach", str(c["tip_mach"]), "--rotor-axis", "x",
                 "--rotor-frame", "mrf", "--mach", f"{minf:.6f}", "--re", str(c["re"]),
                 "--kick", "0", "--ft", f"{ft:.6f}", "--emit", "stats"]
        else:
            die(f"unknown kind {self.kind}")
        return [DNS] + a + ["--out", "run"]

    # ---- foam ----
    def foam_box(self, placement):
        """Box faces relative to the body origin, in L (= 1 m) units."""
        c = self.c
        Ly = c["dr"] * D_BOX
        Lx = c.get("aspect", 1.0) * Ly
        s = 1.0 / (D_BOX / 2 if self.kind == "section" else D_BOX)   # box units -> L
        cx, cy, cz = placement["cx"], placement["cy"], placement["cz"]
        box = dict(xu=cx * s, xd=(Lx - cx) * s, yb=cy * s, yt=(Ly - cy) * s)
        if self.kind == "rotor":
            box.update(zb=cz * s, zt=(Ly - cz) * s)
        return box

    def ramp_s(self, U):
        """dns --ramp (box time) in foam seconds: dns D <-> foam length
        (cylinder/rotor D = 1 m, section 2 c = 2 m)."""
        dlen = 2.0 if self.kind == "section" else 1.0
        return self.c.get("ramp", 2.0) / D_BOX * dlen / U

    def kick(self, U):
        """dns --kick vortex, foam frame: centre (D, D/4) from the placement
        point, width D/4, amplitude kick U (D = dns D in foam metres)."""
        dlen = 2.0 if self.kind == "section" else 1.0
        return dict(centre=(dlen, 0.25 * dlen), kw=0.25 * dlen,
                    amp=self.c.get("kick", 0.05) * U)


# --- dns dry run / placement ---------------------------------------------------

RE_GRID = re.compile(r"grid\s+(\d+) x (\d+) x (\d+)\s+h=([0-9.eE+-]+)")
RE_PLACE = re.compile(r"placement\s+cx=([0-9.eE+-]+) cy=([0-9.eE+-]+) cz=([0-9.eE+-]+)")
RE_FOCAL = re.compile(r"^\s+([xyz]) physical .*fine ([0-9.eE+-]+) coarse ([0-9.eE+-]+)", re.M)


def dry_run(argv, cwd):
    r = subprocess.run(argv + ["--dry-run"], cwd=cwd, capture_output=True, text=True,
                       timeout=120)
    txt = r.stdout + r.stderr
    if r.returncode != 0:
        die(f"dns --dry-run failed (rc {r.returncode}):\n{txt}", r.returncode)
    g = RE_GRID.search(txt)
    p = RE_PLACE.search(txt)
    info = dict(text=txt, nx=int(g[1]), ny=int(g[2]), nz=int(g[3]), h=float(g[4]),
                cx=float(p[1]), cy=float(p[2]), cz=float(p[3]))
    fine = {m[1]: float(m[2]) for m in RE_FOCAL.finditer(txt)}
    if fine:
        info["h_fine"] = fine
    return info


def place_key(argv):
    """Cache key of a placement dry run: the argv without --ft (the placement
    does not depend on the run length)."""
    a = list(argv)
    if "--ft" in a:
        i = a.index("--ft")
        del a[i:i + 2]
    return shlex.join(a)


def placement(case, tier="A"):
    """The dns placement the foam box is built around: the finest routine dns
    level's dry run (positions differ between levels by < 1 cell). Cached in
    runs/parity/tmp_place/<case>/placement.json under the dry-run argv, so a
    foam run does not need the dns binary once the placement is known."""
    lv = case.levels("dns", tier, False)[-1]
    tmp = os.path.join(RUNS, "tmp_place", case.name)
    os.makedirs(tmp, exist_ok=True)
    argv = case.dns_argv(tier, lv, tmp)
    key = place_key(argv)
    cp = os.path.join(tmp, "placement.json")
    if os.path.exists(cp):
        with open(cp) as f:
            d = json.load(f)
        if d.get("key") == key:
            return d["info"]
    info = dry_run(argv, tmp)
    with open(cp, "w") as f:
        json.dump(dict(key=key, source=f"dns --dry-run {now()}",
                       info={k: v for k, v in info.items() if k != "text"}), f, indent=1)
    return info


# --- dns observables ------------------------------------------------------------

def read_csv(path):
    with open(path) as f:
        rows = list(csv.DictReader(f))
    return {k: np.array([float(r[k]) for r in rows]) for k in rows[0]} if rows else {}


def lw_from_speed(sp, ic, jc, h, D):
    """Recirculation length (rear surface -> closure) from |u| on the wake
    axis: first max after the body, then the following minimum, V-fit
    (runs/verify/L1_ghostwall/analyze.py)."""
    j0 = int(math.floor(jc))
    a = jc - j0
    line = (1 - a) * sp[j0, :] + a * sp[j0 + 1, :]
    xs = (np.arange(sp.shape[1]) - ic) * h / D
    start = int(np.searchsorted(xs, 0.5)) + 1
    s, x = line[start:], xs[start:]
    k = 1
    while k < len(s) - 1 and not (s[k] >= s[k - 1] and s[k] >= s[k + 1]):
        k += 1
    while k < len(s) - 1 and not (s[k] <= s[k - 1] and s[k] <= s[k + 1]):
        k += 1
    if k < 4 or k > len(s) - 4 or x[k] > 0.5 * x[-1]:
        return float("nan")
    pl = np.polyfit(x[k - 4:k - 1], -s[k - 4:k - 1], 1)
    pr = np.polyfit(x[k + 1:k + 4], s[k + 1:k + 4], 1)
    return 0.5 * (-pl[1] / pl[0] - pr[1] / pr[0]) - 0.5


def dns_observables(case, tier, rundir, dry):
    """{name: (t_conv array, q array)} from the run's stats (+ frames)."""
    c = case.c
    st = read_csv(os.path.join(rundir, "run_stats.csv"))
    t = st["t"] / case.conv_box()
    out = {}
    if case.kind == "cylinder":
        out["cd"], out["cl"] = (t, st["cd"]), (t, st["cl"])
        if "Lw" in c["observables"]:
            nx, ny = dry["nx"], dry["ny"]
            sp = np.fromfile(os.path.join(rundir, "run_speed.bin"), dtype=np.float32)
            nfr = sp.size // (nx * ny)
            sp = sp[:nfr * nx * ny].reshape(nfr, ny, nx)
            ic, jc = dry["cx"] / dry["h"] - 0.5, dry["cy"] / dry["h"] - 0.5
            dD = D_BOX / dry["h"]
            lw = np.array([lw_from_speed(sp[k], ic, jc, dry["h"], D_BOX) for k in range(nfr)])
            tf = np.arange(nfr) * c["lw_every"]
            ok = np.isfinite(lw)
            out["Lw"] = (tf[ok], lw[ok])
            del dD
    elif case.kind == "section":
        poly = geom.section_polygon(c["aoa"], geom.section_unit(c["thickness"], c["crest"]))
        A = geom.section_frontal_area(poly, math.pi / 8, dry["cx"], dry["cy"], dry["h"])
        k = A / (math.pi / 8)            # cd_chord = cd_dns * A / c
        out["cd"], out["cl"] = (t, st["cd"] * k), (t, st["cl"] * k)
    elif case.kind == "rotor":
        out["CT"], out["CQ"] = (t, -st["ct"]), (t, -st["cq"])
    return out


# --- foam observables -----------------------------------------------------------

def foam_series(case_dir, name):
    rows = {}
    for f in sorted(glob.glob(os.path.join(case_dir, "postProcessing/forces/*", name))):
        cols = None
        with open(f) as fh:
            for line in fh:
                if line.startswith("#"):
                    h = line[1:].split()
                    cols = h if h and h[0] == "Time" else cols
                    continue
                v = line.replace("(", " ").replace(")", " ").split()
                if v and cols is not None and len(v) == len(cols):
                    rows[float(v[0])] = [float(x) for x in v]
    if not rows:
        return None, None
    ks = sorted(rows)
    return cols, np.array([rows[k] for k in ks])


def foam_lw(case_dir):
    """Lw/D history from the sampled wake line: first Ux zero crossing (- -> +)."""
    out = []
    for d in glob.glob(os.path.join(case_dir, "postProcessing/wake/*")):
        f = glob.glob(os.path.join(d, "line*U*.xy"))
        if not f:
            continue
        a = np.loadtxt(f[0])
        x, ux = a[:, 0], a[:, 1]
        i = np.nonzero((ux[:-1] < 0) & (ux[1:] >= 0))[0]
        lw = (x[i[0]] - ux[i[0]] * (x[i[0] + 1] - x[i[0]]) / (ux[i[0] + 1] - ux[i[0]]) - 0.5
              if len(i) else 0.0)
        out.append((float(os.path.basename(d)), lw))
    out.sort()
    return np.array([o[0] for o in out]), np.array([o[1] for o in out])


def foam_observables(case, tier, case_dir, meta):
    """{name: (t_conv or iteration, q)}."""
    cols, F = foam_series(case_dir, "force.dat")
    if F is None:
        return {}
    ix, iy = cols.index("total_x"), cols.index("total_y")
    t = F[:, 0]
    out = {}
    if tier == "A":
        U, rho, L = meta["U"], meta["rho"], 1.0
        tconv = t * U / L
    elif tier == "Bc":
        U, rho, L = meta["U"], meta["rho"], 1.0
        tconv = t                     # iterations
    else:
        U, rho, L = meta["U"], 1.0, 1.0
        tconv = t
    if case.kind in ("cylinder", "section"):
        q = 0.5 * rho * U * U * L * 1.0
        out["cd"], out["cl"] = (tconv, F[:, ix] / q), (tconv, F[:, iy] / q)
        if "Lw" in case.c["observables"]:
            tl, lw = foam_lw(case_dir)
            if len(tl):
                out["Lw"] = ((tl * U / L) if tier == "A" else tl, lw)
    else:
        mc, M = foam_series(case_dir, "moment.dat")
        n = meta["n"]
        k = min(len(F), len(M))
        tt = tconv[:k]
        out["CT"] = (tt, -F[:k, ix] / (rho * n * n))
        out["CQ"] = (tt, -M[:k, mc.index("total_x")] / (rho * n * n))
    return out


# --- evaluation -----------------------------------------------------------------

def evaluate(case, obs, tol):
    """(t_settle, finals) under the case's stop rule."""
    c = case.c
    if c["regime"] == "periodic":
        t, cd = obs["cd"]
        _, cl = obs["cl"]
        ps = periodic_series(t, cd, cl)
        if ps is None:
            return float("nan"), {}
        tc, st, cdm, clr = ps
        ts, fin = settle_steady({"St": (tc, st), "cd": (tc, cdm)}, tol, {})
        fin["cl_rms"] = float(clr[-1])
        fin["periods"] = int(len(tc))
        return ts, fin
    names = c["observables"]
    scales = {}
    for k in names:
        if k in c.get("scale_by", {}):
            scales[k] = abs(float(obs[c["scale_by"][k]][1][-1]))
    ts, fin = settle_steady({k: obs[k] for k in names}, tol, scales)
    return ts, fin


# --- run: dns -------------------------------------------------------------------

def budget_check(case, solver, tier, idx, ref, force):
    key = f"est_{solver}_{tier}" + ("_ref" if ref else "")
    est = case.c.get(key)
    cap = case.cfg["suite"]["ref_cap_s" if ref else "run_cap_s"]
    if est is None:
        print(f"  budget: no calibrated estimate ({key}); cap {cap} s")
        return cap
    e = est if ref else est[idx]
    print(f"  budget: estimate {e:.0f} s, cap {cap} s")
    if e > cap and not force:
        die(f"{case.name} {solver} tier {tier} level {idx + 1}: estimate {e:.0f} s exceeds "
            f"the {cap} s cap (run `calibrate`, or --force)")
    return max(cap, e)


def run_dns(case, tier, idx, ref, force):
    cfg = case.cfg
    lvl = case.levels("dns", tier, ref)[0 if ref else idx]
    label = "ref" if ref else f"L{idx + 1}"
    rd = case.run_dir(tier, "dns", label, ref)
    if os.path.isdir(rd):
        shutil.rmtree(rd)
    os.makedirs(rd)
    argv = case.dns_argv(tier, lvl, rd)
    print(f"[{now()}] {case.name} tier {tier} dns {label}\n  {shlex.join(argv)}")
    dry = dry_run(argv, rd)
    with open(os.path.join(rd, "dry.txt"), "w") as f:
        f.write(dry["text"])
    print("  " + "\n  ".join(l for l in dry["text"].splitlines() if "grid" in l or "fine" in l))
    tmo = budget_check(case, "dns", tier, idx, ref, force) * 1.5
    r = timed(cfg, argv, os.path.join(rd, "log.txt"), tmo, rd, power=True)
    pts, setup = wall_map(r["lines"], "dns")
    res = dict(case=case.name, tier=tier, solver="dns", level=label, argv=argv,
               rc=r["rc"], run_wall_s=r["wall"], cells=dry["nx"] * dry["ny"] * dry["nz"],
               grid=[dry["nx"], dry["ny"], dry["nz"]], h=dry["h"],
               resolution=resolution_dns(case, dry), mach=case.mach(tier),
               provenance=provenance(cfg))
    if r["rc"] not in (0, 124):
        res["error"] = f"dns rc {r['rc']}"
        return finish(rd, res)
    obs = dns_observables(case, tier, rd, dry)
    ts, fin = evaluate(case, obs, cfg["suite"]["settle_tol"])
    w = wall_at(pts, ts * case.conv_box()) if math.isfinite(ts) else float("nan")
    steps = done_steps(r["lines"])
    res.update(observables=fin, t_settle=ts, t_end=float(obs[next(iter(obs))][0][-1]),
               setup_s=setup, solve_s=w - setup, total_s=w,
               gpu_energy_settle_J=energy(r["power"], w),
               gpu_energy_total_J=energy(r["power"], r["wall"]),
               gpu_power_avg_W=float(np.nanmean([p for _, p in r["power"]]))
               if r["power"] else float("nan"),
               cpu_energy_J="n/a (RAPL root-only)", steps=steps,
               throughput=dict(cell_updates_per_s=res["cells"] * steps / max(r["wall"] - setup, 1e-9)),
               placement=dict(cx=dry["cx"], cy=dry["cy"], cz=dry["cz"]))
    return finish(rd, res)


def done_steps(lines):
    for _, s in lines:
        m = re.match(r"^done: (\d+) steps", s)
        if m:
            return int(m.group(1))
    return 0


def resolution_dns(case, dry):
    c = case.c
    if case.kind == "cylinder":
        return dict(D_h=D_BOX / dry["h"], cells_per_BL=D_BOX / dry["h"] / math.sqrt(c["re"]))
    if case.kind == "section":
        ch = (math.pi / 8) / dry["h"]
        return dict(c_h=ch, t_h=c["thickness"] * ch)
    hf = max(dry["h_fine"].values())
    dh = D_BOX / hf
    cmin, tmin, c75, t75 = geom.rotor_resolution(c["blade"], 1.0 / dh)
    return dict(D_hfine=dh, hfine_axes=dry["h_fine"], c_h_min=cmin, t_h_min=tmin,
                c_h_75=c75, t_h_75=t75)


def finish(rd, res):
    with open(os.path.join(rd, "result.json"), "w") as f:
        json.dump(res, f, indent=1, default=str)
    o = res.get("observables", {})
    print(f"  -> rc {res['rc']} " + " ".join(f"{k} {v:.6g}" for k, v in o.items()
                                             if isinstance(v, float))
          + (f"  t_settle {res['t_settle']:.4g}  setup {res['setup_s']:.1f} s  "
             f"total {res['total_s']:.1f} s  run {res['run_wall_s']:.1f} s"
             if "t_settle" in res else f"  {res.get('error', '')}"))
    return res


# --- run: foam ------------------------------------------------------------------

def mpirun(np_, app):
    if np_ == 1:
        return [app]
    return ["mpirun", "-np", str(np_), "--bind-to", "core", "--map-by", "core", app, "-parallel"]


def foam_cmd(cfg, case_dir, log, argv, timeout_s, setup_log):
    env = foam_env(cfg)
    r = timed(cfg, argv, os.path.join(case_dir, log), timeout_s, case_dir, env=env)
    setup_log.append(dict(cmd=" ".join(argv), wall=r["wall"], rc=r["rc"]))
    if r["rc"] != 0:
        die(f"'{' '.join(argv)}' failed rc {r['rc']} (see {os.path.join(case_dir, log)})")
    return r


def rotor_foam_mesh(case, lvl, idx_label, placement_info, setup_log):
    """Mesh once per level (shared by tier A and B): blockMesh, snappy in
    parallel, reconstruct. Cached under runs/parity/meshes/."""
    cfg = case.cfg
    c = case.c
    zr, rr = case.zone_r(), case.refine_r()
    study = rr != zr
    md = os.path.join(RUNS, "meshes", f"{case.name}_{idx_label}"
                      + (f"_zr{zr:g}_rr{rr:g}" if study else ""))
    meta_p = os.path.join(md, "mesh.json")
    if os.path.exists(meta_p):
        meta = json.load(open(meta_p))
        if meta.get("lvl") == lvl and meta.get("zone_r") == zr \
                and meta.get("refine_r", meta.get("zone_r")) == rr:
            setup_log.extend(meta["setup"])
            return md, meta
    if os.path.isdir(md):
        shutil.rmtree(md)
    os.makedirs(md)
    stl = os.path.join(md, "body.stl")
    info = geom.write_rotor_stl(stl, c["blade"])
    sw = info["swept"]
    a = info["a"]
    lo, hi = info["bounds"]
    zx = (lo[0] / sw - c["zone_margin"], hi[0] / sw + c["zone_margin"])
    box = case.foam_box(placement_info)
    cpd, lmin, lmax, lzone, lwake = lvl
    hub = dict(r=a.hub_radius * 0.5 / sw, half=0.5 * a.hub_length / sw, sides=a.hub_sides)
    np_ = cfg["suite"]["np"]
    m = fc.rotor_mesh(md, stl, box, cpd, lmin, lmax, lzone, lwake, zr, zx,
                      c["wake_len"], hub, np_, refine_r=rr if study else None)
    for d in ("0",):
        os.makedirs(os.path.join(md, d), exist_ok=True)
    # minimal controlDict for the utilities
    fc.control(md, "simpleFoam", 1, 1, 1, 1, "")
    fc.schemes_b(md, True)
    fc.solution_b(md, 1, "yes", 1, 0.7)
    steps = []
    T = cfg["suite"]["ref_cap_s"]
    foam_cmd(cfg, md, "log.blockMesh", ["blockMesh"], T, steps)
    foam_cmd(cfg, md, "log.surfaceFeatureExtract", ["surfaceFeatureExtract"], T, steps)
    foam_cmd(cfg, md, "log.decomposePar0", ["decomposePar", "-force", "-no-fields"], T, steps)
    foam_cmd(cfg, md, "log.snappyHexMesh", mpirun(np_, "snappyHexMesh") + ["-overwrite"], T, steps)
    foam_cmd(cfg, md, "log.reconstructParMesh",
             ["reconstructParMesh", "-constant", "-mergeTol", "1e-6"], T, steps)
    for p in glob.glob(os.path.join(md, "processor*")):
        shutil.rmtree(p)
    r = foam_cmd(cfg, md, "log.checkMesh", ["checkMesh", "-constant", "-noZero"], T, steps)
    txt = "\n".join(s for _, s in r["lines"])
    cells = int(re.search(r"^\s*cells:\s+(\d+)", txt, re.M).group(1))
    ok = "Mesh OK" in txt
    hc = foam_cmd(cfg, md, "log.topoSet", ["topoSet", "-constant", "-noZero", "-dict",
                                            "system/topoSetDict.hubcheck"], T, steps)
    mt = re.search(r"cellSet hubCore now size (\d+)", "\n".join(s for _, s in hc["lines"]))
    hubcore = int(mt.group(1)) if mt else -1
    meta = dict(lvl=lvl, cells=cells, mesh_ok=ok, hub_core_cells=hubcore, swept=sw,
                zone_x=zx, zone_r=zr, refine_r=rr, box=box, setup=steps, **m)
    with open(meta_p, "w") as f:
        json.dump(meta, f, indent=1)
    setup_log.extend(steps)
    if hubcore != 0:
        die(f"rotor mesh {md}: {hubcore} cells inside the hub core")
    return md, meta


def run_foam(case, tier, idx, ref, force, np_, timeout_s=None):
    cfg = case.cfg
    c = case.c
    np_ = np_ or cfg["suite"]["np"]
    lvl = case.levels("foam", tier, ref)[0 if ref else idx]
    label = "ref" if ref else f"L{idx + 1}"
    rd = case.run_dir(tier, "foam", label, ref, np_)
    if os.path.isdir(rd):
        shutil.rmtree(rd)
    os.makedirs(rd)
    app = FOAM_APP[tier]
    sa = tier != "B"                  # SI gas (A, Bc) vs kinematic (B)
    print(f"[{now()}] {case.name} tier {tier} foam {label} ({app}, np {np_}, "
          f"U {case.u_scheme()})\n  case {rd}")
    cap_s = budget_check(case, "foam", tier, idx, ref, force)
    pl = placement(case, "A")
    box = case.foam_box(pl)
    setup = []
    T = cfg["suite"]["ref_cap_s"]
    meta = dict(app=app, np=np_, box=box, level=lvl, u_scheme=case.u_scheme())
    if tier == "Bc":
        meta["e_scheme"] = fc.E_DIV_BC
    pr = cfg["suite"]["prandtl"]
    t0 = time.monotonic()
    if sa:
        st = fc.tier_a_state(c["mach"], c["re"])
        meta.update(U=st["U"], rho=st["rho"], mu=st["mu"], p=st["p"], T=st["T"])
        end = c["t_cap"] / st["U"]
    else:
        U = c["J"] if case.kind == "rotor" else 1.0
        meta.update(U=U, nu=U * 1.0 / c["re"])
    if case.kind == "rotor":
        L = 1.0
        if sa:
            n = st["U"] / (c["J"] * L)
        else:
            n = 1.0
        meta.update(n=n, omega=2 * math.pi * n, zone_r=case.zone_r(),
                    refine_r=case.refine_r())
        md, mmeta = rotor_foam_mesh(case, lvl, f"L{idx + 1}" if not ref else "ref", pl, setup)
        shutil.copytree(os.path.join(md, "constant", "polyMesh"),
                        os.path.join(rd, "constant", "polyMesh"))
        meta.update(cells=mmeta["cells"], mesh=mmeta)
        patches = dict(inlet="inlet", outlet="outlet", sides="lateral", propeller="wall")
        nonorth = 1
        limited = True
        body = "propeller"
    elif case.kind == "cylinder":
        nq, nr = lvl
        m = fc.cylinder_mesh(rd, box, nq, nr)
        meta.update(cells=m["cells"], mesh=m)
        patches = dict(inlet="inlet", outlet="outlet", bottom="symmetryPlane",
                       top="symmetryPlane", body="wall", frontAndBack="empty")
        nonorth, limited, body = 1, False, "body"
    else:
        nt, nr, d1 = lvl
        poly = geom.section_polygon(c["aoa"], geom.section_unit(c["thickness"], c["crest"]))
        mx, my = geom.bbox_centre(poly)
        ring = [(x - mx, y - my) for x, y in poly]
        m = fc.section_mesh(rd, ring, box, nt, nr, d1)
        meta.update(cells=m["cells"], mesh=m)
        patches = dict(body="wall", inlet="inlet", outlet="outlet", sides="lateral",
                       frontAndBack="empty")
        nonorth, limited, body = 2, True, "body"
    # physics + control
    fos = fc.fo_forces(body, "rho" if sa else "rhoInf", rho_inf=meta.get("rho", 1.0),
                       p_ref=meta["p"] if tier == "Bc" else None)
    if "Lw" in c["observables"]:
        if tier == "A":
            fos += fc.fo_line("wake", (0.5, 0, 0.5), (8, 0, 0.5), 3001,
                              c["lw_every"] / meta["U"], app)
        else:
            fos += fc.fo_line("wake", (0.5, 0, 0.5), (8, 0, 0.5), 3001, c["b_lw_every"], app)
    mrf = meta.get("omega") if case.kind == "rotor" else None
    if tier == "A":
        fc.physics_a(rd, st, pr, mrf=mrf)
        fc.boost_sources(rd, st["U"], case.ramp_s(st["U"]))
        fc.schemes_a(rd, limited, case.u_scheme())
        fc.solution_a(rd, nonorth, c.get("foam_nouter", 1))
        dt0 = 1e-3 / st["U"]
        fc.control(rd, app, end, dt0, c.get("foam_maxco", 1.0), end, fos,
                   max_dt=c.get("foam_max_dt", 0.05) / st["U"])
    elif tier == "Bc":
        fc.physics_a(rd, st, pr, mrf=mrf)
        fc.schemes_bc(rd, limited, case.u_scheme())
        b = c["bc_simple"]
        fc.solution_bc(rd, nonorth, b["consistent"], b["p"], b["U"], b["e"])
        iters = (c["bc_iters_ref"] if ref else c["bc_iters"][idx])
        fc.control(rd, app, iters, 1, 1, iters, fos)
        meta["iters_cap"] = iters
    else:
        fc.physics_b(rd, meta["nu"], mrf=mrf)
        fc.schemes_b(rd, limited, case.u_scheme())
        b = c["b_simple"]
        fc.solution_b(rd, nonorth, b["consistent"], b["p"], b["U"])
        iters = (c["b_iters_ref"] if ref else c["b_iters"][idx])
        fc.control(rd, app, iters, 1, 1, iters, fos)
        meta["iters_cap"] = iters
    fc.decompose(rd, np_)
    py_setup = time.monotonic() - t0
    # meshing + fields
    if case.kind != "rotor":
        foam_cmd(cfg, rd, "log.blockMesh", ["blockMesh"], T, setup)
    if tier == "A":
        fc.fields_a(rd, st, patches, case.ramp_s(st["U"]), box["xu"] + box["xd"],
                    mrf=case.kind == "rotor")
        if case.kind != "rotor" and c.get("kick", 0.05) > 0:
            foam_cmd(cfg, rd, "log.writeCellCentres",
                     ["postProcess", "-func", "writeCellCentres", "-time", "0"], T, setup)
            t1 = time.monotonic()
            C = fc.read_cell_centres(rd)
            k = case.kick(st["U"])
            fc.fields_a(rd, st, patches, case.ramp_s(st["U"]), box["xu"] + box["xd"],
                        internal_u=fc.kick_field(C, k["centre"], k["kw"], k["amp"]))
            for f in glob.glob(os.path.join(rd, "0", "C*")):
                os.remove(f)
            py_setup += time.monotonic() - t1
    elif tier == "Bc":
        fc.fields_bc(rd, st, patches, mrf=case.kind == "rotor")
    else:
        fc.fields_b(rd, meta["U"], patches, mrf=case.kind == "rotor")
    if np_ > 1:
        foam_cmd(cfg, rd, "log.decomposePar", ["decomposePar", "-force"], T, setup)
    meta["py_setup_s"] = py_setup
    # solve (rc 124 = stopped by the timeout: scored on what it wrote)
    tmo = timeout_s or cap_s * 1.5
    meta["solver_timeout_s"] = tmo
    r = timed(cfg, mpirun(np_, app), os.path.join(rd, "log.solver"), tmo, rd,
              env=foam_env(cfg), power=True)
    pts, _ = wall_map(r["lines"], "foam")
    setup_s = sum(s["wall"] for s in setup) + py_setup
    res = dict(case=case.name, tier=tier, solver="foam", app=app, level=label, case_dir=rd,
               np=np_, rc=r["rc"], run_wall_s=r["wall"], cells=meta["cells"],
               resolution=resolution_foam(case, lvl, meta), mach=c["mach"] if sa else 0.0,
               setup_steps=setup, meta={k: v for k, v in meta.items() if k != "mesh"},
               provenance=provenance(cfg))
    if r["rc"] not in (0, 124) or not pts:
        res["error"] = f"{app} rc {r['rc']}"
        return finish(rd, res)
    obs = foam_observables(case, tier, rd, meta)
    if not obs:
        res["error"] = "no force output"
        return finish(rd, res)
    ts, fin = evaluate(case, obs, cfg["suite"]["settle_tol"])
    if tier == "A":
        sim = ts / (meta["U"] / 1.0)
    else:
        sim = ts                      # steady: iterations
    w = wall_at(pts, sim) if math.isfinite(ts) else float("nan")
    nsteps = len(pts)
    res.update(observables=fin, t_settle=ts, t_end=float(obs[next(iter(obs))][0][-1]),
               setup_s=setup_s, solve_s=w, total_s=setup_s + w,
               gpu_energy_settle_J="n/a (CPU run)", cpu_energy_J="n/a (RAPL root-only)",
               steps=nsteps,
               throughput=dict(cell_iterations_per_s=meta["cells"] * nsteps / max(r["wall"], 1e-9)))
    return finish(rd, res)


def resolution_foam(case, lvl, meta):
    if case.kind == "cylinder":
        return dict(nq=lvl[0], nr=lvl[1], first_cell_D=meta["mesh"]["first"])
    if case.kind == "section":
        return dict(nt=lvl[0], cells_around=8 * lvl[0], nr=lvl[1], first_cell_c=lvl[2])
    m = meta["mesh"]
    return dict(cells_per_D=lvl[0], refine=lvl[1:3], h_blade_D=m["h_blade"],
                D_hblade=1.0 / m["h_blade"][0])


# --- commands -------------------------------------------------------------------

def cmd_list(cfg, _):
    for name, c in cfg["cases"].items():
        print(f"{name:8s} {c['kind']:9s} Re {c['re']:<6g} M {c['mach']:<7g} "
              f"tiers {','.join(c['tiers'])}  observables {','.join(c['observables'])}")
        for tier in c["tiers"]:
            case = Case(cfg, name)
            print(f"    tier {tier}: dns {case.levels('dns', tier, False)}  "
                  f"foam {case.levels('foam', tier, False)}")


def cmd_run(cfg, a):
    case = Case(cfg, a.case)
    if a.tier not in case.c["tiers"]:
        die(f"{a.case}: no tier {a.tier}")
    if a.t_cap or a.tag or a.mach or a.dns_ny or a.zone_r:
        if not a.tag:
            die("--t-cap/--mach/--dns-ny/--zone-r are calibration overrides: give --tag "
                "(runs go to runs/parity/calib/)")
        case.tag = a.tag
        if a.t_cap:
            case.c["t_cap"] = a.t_cap
        if a.mach:
            case.c["mach"] = a.mach
        if a.dns_ny:
            case.c["dns_levels"] = [a.dns_ny]
        if a.zone_r:
            if case.kind != "rotor":
                die("--zone-r: rotor cases only")
            case.c["refine_r"] = case.refine_r()
            case.c["zone_r"] = a.zone_r
    solvers = [a.solver] if a.solver else ["dns", "foam"]
    for s in solvers:
        if s == "dns" and a.tier in ("B", "Bc") and case.c.get("b_dns") == "reuse_A":
            print(f"{a.case}: tier {a.tier} dns = tier A dns runs re-scored (cases.toml b_dns)")
            continue
        n = 1 if a.ref else len(case.levels(s, a.tier, False))
        ks = [a.level - 1] if a.level else range(n)
        for k in ks:
            if s == "dns":
                run_dns(case, a.tier, k, a.ref, a.force)
            else:
                run_foam(case, a.tier, k, a.ref, a.force, a.np, a.timeout)


def cmd_adopt(cfg, a):
    """Score an existing dns run (its run_stats.csv + log) as a suite level
    without re-running it. The grid comes from a saved --dry-run text of the
    same argv (--dry); timing is the log's `done: N steps ... X ms/step`, not
    harness-stamped, and setup is not measured (NaN)."""
    case = Case(cfg, a.case)
    lvl = case.levels("dns", a.tier, False)[a.level - 1]
    rd = case.run_dir(a.tier, "dns", f"L{a.level}", False)
    if os.path.isdir(rd):
        die(f"{rd} exists: mark it stale first (`stale`)")
    with open(a.dry) as f:
        dtxt = f.read()
    os.makedirs(rd)
    for fn in ("run_stats.csv", "log.txt", "run.json", "err.txt"):
        if os.path.exists(os.path.join(a.src, fn)):
            shutil.copy2(os.path.join(a.src, fn), os.path.join(rd, fn))
    with open(os.path.join(rd, "dry.txt"), "w") as f:
        f.write(dtxt)
    g, p = RE_GRID.search(dtxt), RE_PLACE.search(dtxt)
    dry = dict(nx=int(g[1]), ny=int(g[2]), nz=int(g[3]), h=float(g[4]),
               cx=float(p[1]), cy=float(p[2]), cz=float(p[3]),
               h_fine={m[1]: float(m[2]) for m in RE_FOCAL.finditer(dtxt)})
    argv = case.dns_argv(a.tier, lvl, rd)
    if a.ft:
        argv[argv.index("--ft") + 1] = f"{a.ft:.6f}"
    log = open(os.path.join(rd, "log.txt")).read()
    m = re.search(r"^done: (\d+) steps, \d+ frames, ([0-9.]+) ms/step", log, re.M)
    if not m:
        die(f"{a.src}/log.txt: no `done:` line")
    steps, ms = int(m[1]), float(m[2])
    obs = dns_observables(case, a.tier, rd, dry)
    ts, fin = evaluate(case, obs, cfg["suite"]["settle_tol"])
    t_end = float(obs[next(iter(obs))][0][-1])
    solve = steps * ms * 1e-3
    w = solve * min(ts / t_end, 1.0) if math.isfinite(ts) else float("nan")
    nan = float("nan")
    res = dict(case=case.name, tier=a.tier, solver="dns", level=f"L{a.level}", argv=argv,
               rc=0, run_wall_s=solve, cells=dry["nx"] * dry["ny"] * dry["nz"],
               grid=[dry["nx"], dry["ny"], dry["nz"]], h=dry["h"],
               resolution=resolution_dns(case, dry), mach=case.mach(a.tier),
               observables=fin, t_settle=ts, t_end=t_end, setup_s=nan, solve_s=w, total_s=w,
               gpu_energy_settle_J=nan, gpu_energy_total_J=nan, gpu_power_avg_W=nan,
               cpu_energy_J="n/a (RAPL root-only)", steps=steps,
               throughput=dict(cell_updates_per_s=dry["nx"] * dry["ny"] * dry["nz"] / (ms * 1e-3)),
               placement=dict(cx=dry["cx"], cy=dry["cy"], cz=dry["cz"]),
               adopted=dict(source=os.path.abspath(a.src), dry=os.path.abspath(a.dry),
                            timing="steps x ms/step from the dns log; setup not measured",
                            recorded=now()),
               provenance=dict(dns_binary="adopted run", git_rev=a.rev or "?",
                               stats_mtime=datetime.datetime.fromtimestamp(os.stat(
                                   os.path.join(rd, "run_stats.csv")).st_mtime).isoformat(
                                   timespec="seconds"), gpu="RX 7900 XTX", recorded=now()))
    print(f"[{now()}] {case.name} tier {a.tier} dns L{a.level} adopted from {a.src}")
    finish(rd, res)


def cmd_stale(cfg, a):
    """Mark a finished run stale (kept for the record, excluded from
    references, errors and cost curves) and move it aside to <run>_stale so a
    regenerated run can take its place."""
    case = Case(cfg, a.case)
    src = os.path.join(RUNS, *(["ref"] if a.ref else []), a.tier, case.name, a.run)
    rp = os.path.join(src, "result.json")
    if not os.path.exists(rp):
        die(f"{rp}: no such result")
    r = load_json(rp)
    r["stale"] = a.reason
    with open(rp, "w") as f:
        json.dump(r, f, indent=1, default=str)
    dst = src + "_stale"
    if os.path.exists(dst):
        die(f"{dst} exists")
    os.rename(src, dst)
    print(f"{src} -> {dst}: stale ({a.reason})")


def load_json(path):
    with open(path) as f:
        return json.load(f)


def cmd_calibrate(cfg, a):
    """Dry-run every dns level; print the grids. Timing is a short real run
    (`run ... --level k` with t_cap overridden) left to the caller."""
    case = Case(cfg, a.case)
    for tier in case.c["tiers"]:
        for ref in (False, True):
            try:
                lv = case.levels("dns", tier, ref)
            except SystemExit:
                continue
            for k, l in enumerate(lv):
                tmp = os.path.join(RUNS, "tmp_place", case.name)
                os.makedirs(tmp, exist_ok=True)
                d = dry_run(case.dns_argv(tier, l, tmp), tmp)
                print(f"{case.name} tier {tier} dns {'ref' if ref else f'L{k + 1}'} {l}: "
                      f"grid {d['nx']}x{d['ny']}x{d['nz']} = {d['nx'] * d['ny'] * d['nz'] / 1e6:.2f} M "
                      f"cells, res {resolution_dns(case, d)}")


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("list")
    c = sub.add_parser("calibrate")
    c.add_argument("case")
    r = sub.add_parser("run")
    r.add_argument("case")
    r.add_argument("--tier", required=True, choices=TIERS)
    r.add_argument("--solver", choices=("dns", "foam"))
    r.add_argument("--level", type=int)
    r.add_argument("--ref", action="store_true")
    r.add_argument("--np", type=int)
    r.add_argument("--force", action="store_true")
    r.add_argument("--t-cap", type=float, help="calibration: override the case cap (needs --tag)")
    r.add_argument("--tag", default="", help="calibration run-dir suffix (runs/parity/calib/)")
    r.add_argument("--mach", type=float, help="calibration: override the tier A Mach (needs --tag)")
    r.add_argument("--dns-ny", type=int, help="calibration: one dns level --ny (needs --tag)")
    r.add_argument("--zone-r", type=float,
                   help="calibration: rotor MRF zone radius / D, refinement kept (needs --tag)")
    r.add_argument("--timeout", type=float,
                   help="foam solver timeout s (default 1.5 x budget); a timed-out run "
                   "is scored on what it wrote")
    d = sub.add_parser("adopt")
    d.add_argument("case")
    d.add_argument("--tier", required=True, choices=("A",))
    d.add_argument("--level", type=int, required=True)
    d.add_argument("--from", dest="src", required=True, help="dir with run_stats.csv, log.txt")
    d.add_argument("--dry", required=True, help="saved dns --dry-run text of the same grid")
    d.add_argument("--ft", type=float, help="the run's actual --ft (argv record)")
    d.add_argument("--rev", help="git rev of the binary that made the run")
    s = sub.add_parser("stale")
    s.add_argument("case")
    s.add_argument("--tier", required=True, choices=TIERS)
    s.add_argument("--run", required=True, help="run dir name, e.g. foam_L1")
    s.add_argument("--ref", action="store_true", help="a reference run (runs/parity/ref/)")
    s.add_argument("--reason", required=True)
    sub.add_parser("report")
    a = p.parse_args()
    sys.stdout.reconfigure(line_buffering=True)
    cfg = load_cfg()
    if a.cmd == "report":
        import report
        report.main(cfg)
        return
    {"list": cmd_list, "run": cmd_run, "calibrate": cmd_calibrate, "adopt": cmd_adopt,
     "stale": cmd_stale}[a.cmd](cfg, a)


if __name__ == "__main__":
    main()
