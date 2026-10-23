#!/usr/bin/env python3
"""Section polars for the propeller's blade: XFOIL tables, Viterna-extended.

The thrust-curve tools (`dns prop`, via `--export`) need cl(alpha, Re) and
cd(alpha, Re) for the SAME section `propeller.py` lofts into the STL, over the
whole circle of alpha: a root station at J = 0 sits far past stall, and a
windmilling one at negative alpha. XFOIL supplies the attached range; past the
last converged point the Viterna-Corrigan flat-plate model takes over. A
thin-airfoil table (2 pi alpha, constant cd0, the same Viterna post-stall) is
the fallback, used ONLY when asked for by name (`--polar thin`) -- never as a
silent stand-in for a failed XFOIL run.

API (shared with foam_mrf.py; keep it stable):

    add_polar_args(parser)            --polar xfoil|thin, --ncrit, --re-points,
                                      --thin-cd0, --xfoil
    check_polar_args(a)               exits on flags no table can satisfy
    polar_from_args(a, re_list, mach) Polar spanning min..max of re_list
    load(a, re_list, mach, ncrit)     XFOIL table at exactly these Re
    Polar.cl_cd(alpha_rad, re)        vectorized; linear in alpha and log Re,
                                      Re clamped to the table's range
    Polar.from_thin_airfoil(a, cd0)   the --polar thin table

XFOIL runs are cached on disk under ~/.cache/pesto-polars/, keyed by a hash of
(section coordinates, Re, Mach, Ncrit, XFOIL version, sweep schedule); a
second call costs nothing. Each Re is two sweeps marching OUT from alpha = 0
(up and down, fresh boundary layers), because a viscous Newton solve converges
far better from its neighbour than from cold. Unconverged points never reach
the polar file; the table keeps the contiguous run from 0 and stops at the
first gap wider than MAX_GAP_DEG. XFOIL can hang on a post-stall point; a
no-output watchdog (STALL_S) kills it and the march resumes cold past that
alpha.

Re below RE_FLOOR = 1e4 is refused: there XFOIL's closures are uncalibrated
and convergence is no evidence (a 55%-thick lens "converges" at Re 1e3 to
+25 deg, yet fails at alpha 0 for every Re >= 3e4 -- its crest corners
separate the boundary layer). `polar_from_args` tables sit on a FIXED log
grid (`re_table`), so a thrust curve does not shift with its own J list.

CLI: `--re 1e5,2e5` (alias `--re-list`) computes the tables at exactly those
Re; `--re-span LO,HI` instead builds the table a BEMT curve needs for a blade
spanning [LO, HI] (`polar_from_args`: --re-points on the fixed grid, the floor
clamp). `--out` is the inspection CSV (Re,alpha_deg,cl,cd,source at
--step-deg); `--export` is the `dns prop` input: header `re,alpha_deg,cl,cd`,
one block per table Re ascending (the given Re for --polar thin, which is
Re-independent), alpha ascending over [-180, 180] deg in 1 deg steps and
0.25 deg steps inside +-25 deg -- a superset of XFOIL's 0.5 deg nodes, so
linear interpolation of the export reproduces the table there.

XFOIL 6.99 build (done once; gfortran 16, libX11 present):

    mkdir -p ~/.cache/src/xfoil-dl && cd ~/.cache/src/xfoil-dl
    curl -sSLfO https://web.mit.edu/drela/Public/web/xfoil/xfoil6.99.tgz
    tar xzf xfoil6.99.tgz && mkdir -p ~/.cache/src/xfoil
    cp -a Xfoil ~/.cache/src/xfoil/ && cd ~/.cache/src/xfoil/Xfoil/plotlib
    cp config.make.gfortranDP config.make
    make libPlt_gDP.a \\
      FFLAGS="-m64 -O2 -fdefault-real-8 -std=legacy -fallow-argument-mismatch" \\
      CFLAGS="-m64 -O2 -DUNDERSCORE -std=gnu17"
    cd ../bin
    F="-O2 -fdefault-real-8 -std=legacy -fallow-argument-mismatch"
    make -f Makefile_gfortran xfoil FFLAGS="$F" FFLOPT="$F" \\
      CFLAGS="-O2 -DUNDERSCORE -std=gnu17" BINDIR=. INSTALLCMD=true \\
      PLTLIB=-lX11
    install -m755 xfoil ~/.local/bin/xfoil

The Orr-Sommerfeld database (orrs/) is NOT needed: it only feeds the TS-wave
frequency plots, not the e^N envelope transition model. `-std=gnu17` because
gcc >= 15 defaults to C23, where plotlib's K&R prototypes are errors. Headless
use is `PLOP / G F` (graphics off) on stdin, which every run here sends.

Section file: x = u + 0.5 in [0, 1], ordered TE -> suction side -> LE ->
pressure side -> TE (XFOIL's counter-clockwise convention). The four-point lens
is sampled densely along its straight edges with its corners (LE and both
crests) DOUBLED: XFOIL splines a buffer airfoil, and a doubled point is how it
is told a slope break is real (`SEGSPL`, `PANGEN`'s corner pass) -- otherwise
the spline overshoots at the crests. naca and dat sections go in as
`section_unit` gives them, LE once (a dat file is cached by its RESAMPLED
coordinates, so editing the file or --section-points is a new key). Both are
then re-panelled by PANE (PPAR N = NPAN); the lens's sharp LE is a wedge,
which XFOIL tolerates but converges on worse than a round nose.
"""

import argparse
import concurrent.futures
import hashlib
import json
import math
import os
import re as _re
import shutil
import subprocess
import sys
import tempfile
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import propeller  # noqa: E402

CACHE = os.path.expanduser("~/.cache/pesto-polars")
NPAN = 200             # XFOIL panel nodes
ITER = 200             # viscous Newton iterations per point
STEP_DEG = 0.5         # sweep step
ALPHA_UP_DEG = 25.0    # sweep 0 -> +25
ALPHA_DOWN_DEG = -20.0  # sweep 0 -> -20
MAX_GAP_DEG = 3.0      # a wider hole in the converged points ends the range
XFOIL_TIMEOUT = 120.0  # seconds per XFOIL process, hard; keeps points so far
STALL_S = 8.0          # seconds without output before XFOIL is presumed hung
SWEEP_RESUMES = 2      # cold restarts past a halt or a hang
THIN_STALL_DEG = 10.0  # thin-airfoil linear range about the zero-lift angle
RE_FLOOR = 1e4         # below this XFOIL's closures are out of calibration
BACK_CL = 0.7          # reversed-flow cl scale (AirfoilPrep convention)
LENS_EDGE_POINTS = 40  # samples per straight lens edge
RE_GRID = (1.0, 1 / 2, 1 / 3, 1 / 4, 1 / 6, 1 / 8, 1 / 12)  # decades/step
USE_THIN = "Use --polar thin (thin-airfoil + Viterna, --thin-cd0) instead"


def _die(msg):
    sys.exit(f"polars: {msg}")


# --- section -> XFOIL coordinates -------------------------------------------

def section_xy(a):
    """The section as XFOIL buffer coordinates, TE -> upper -> LE -> lower -> TE.

    Pre: a passed `propeller.check_geometry`. Post: x in [0,1], unit chord.
    """
    lower, upper = propeller.ring_sides(propeller.section_unit(a))
    upper = upper[::-1]                           # TE, suction side, LE
    if a.section == "lens":
        pts = _densify(upper) + _densify(lower)   # LE doubled: a real corner
    else:
        pts = upper + lower[1:]                   # LE once
    return [(u + 0.5, w) for u, w in pts]


def _densify(poly):
    """Straight edges sampled cosine-clustered, interior corners doubled."""
    out = [poly[0]]
    n = LENS_EDGE_POINTS
    for (u0, w0), (u1, w1) in zip(poly[:-1], poly[1:]):
        for k in range(1, n + 1):
            t = 0.5 * (1.0 - math.cos(math.pi * k / n))
            out.append((u0 + t * (u1 - u0), w0 + t * (w1 - w0)))
        if (u1, w1) != poly[-1]:
            out.append((u1, w1))            # doubled corner point
    return out


# --- XFOIL ------------------------------------------------------------------

def find_xfoil(path=None):
    """The XFOIL executable: --xfoil, then PATH, then ~/.local/bin. Or None."""
    for c in (path, shutil.which("xfoil"),
              os.path.expanduser("~/.local/bin/xfoil")):
        if c and os.path.isfile(c) and os.access(c, os.X_OK):
            return c
    return None


def xfoil_version(exe):
    """Version string from XFOIL's banner; part of the cache key."""
    r = subprocess.run([exe], input="QUIT\n", capture_output=True, text=True,
                       timeout=30)
    m = _re.search(r"Version\s+(\S+)", r.stdout)
    return m.group(1) if m else "unknown"


def _script(re_, mach, ncrit, a_start, a_end):
    return "\n".join([
        "PLOP", "G F", "",
        "LOAD sec.dat",
        "PPAR", f"N {NPAN}", "", "",
        "OPER",
        "VPAR", f"N {ncrit:g}", "",
        f"ITER {ITER}",
        f"VISC {re_:.6e}",
        f"MACH {mach:g}",
        "PACC", "p.pol", "",
        f"ASEQ {a_start:g} {a_end:g} {math.copysign(STEP_DEG, a_end):g}",
        "PACC", "",
        "QUIT", ""])


def _parse_pol(path):
    """(alpha_deg, cl, cd) rows of an XFOIL polar file, file order. A killed
    XFOIL may have written none (or no file): that is zero rows."""
    if not os.path.exists(path):
        return []
    with open(path) as f:
        lines = f.read().splitlines()
    head = next((i for i, l in enumerate(lines) if l.lstrip().startswith("------")),
                len(lines))
    rows = []
    for line in lines[head + 1:]:
        try:
            al, cl, cd = (float(x) for x in line.split()[:3])
        except ValueError:          # short or non-numeric line
            continue
        if all(map(math.isfinite, (al, cl, cd))) and cd > 0.0:
            rows.append((al, cl, cd))
    return rows


def _run_xfoil(exe, d, script):
    """Run one XFOIL script in d under a no-progress watchdog.

    XFOIL can spin forever on one post-stall point (measured: NACA 4412 at
    Re 1.2e5 past 18 deg) without printing. It is killed after STALL_S with
    neither its log nor its polar growing, or after XFOIL_TIMEOUT in all.
    """
    env = dict(os.environ)
    env.pop("DISPLAY", None)
    log, pol = os.path.join(d, "log"), os.path.join(d, "p.pol")
    with open(log, "w") as lf:
        proc = subprocess.Popen([exe], stdin=subprocess.PIPE, stdout=lf,
                                stderr=subprocess.STDOUT, cwd=d, env=env,
                                text=True)
        proc.stdin.write(script)
        proc.stdin.close()
        t0 = seen_t = time.monotonic()
        seen = -1
        while proc.poll() is None:
            time.sleep(0.05)
            size = os.path.getsize(log) + (os.path.getsize(pol)
                                           if os.path.exists(pol) else 0)
            now = time.monotonic()
            if size != seen:
                seen, seen_t = size, now
            elif now - seen_t > STALL_S or now - t0 > XFOIL_TIMEOUT:
                proc.kill()
                proc.wait()
                break


def _last_attempted(d):
    """Last alpha XFOIL iterated on (its ' a = ' lines), or None."""
    last = None
    with open(os.path.join(d, "log"), errors="replace") as f:
        for line in f:
            m = _re.match(r"\s+a =\s*(-?[0-9.]+)\s+CL", line)
            last = float(m.group(1)) if m else last
    return last


def _sweep(exe, dat, re_, mach, ncrit, a_end):
    """Converged rows marching alpha from 0 to a_end.

    When XFOIL stops short -- its own 'Sequence halted' after four failures,
    or the watchdog's kill -- the march restarts cold one step past the last
    alpha it tried, at most SWEEP_RESUMES times. `_contiguous` decides later
    whether what lies past the hole is usable.
    """
    step = math.copysign(STEP_DEG, a_end)
    rows, start = [], 0.0
    for _ in range(SWEEP_RESUMES + 1):
        with tempfile.TemporaryDirectory(prefix="xf-", dir=CACHE) as d:
            with open(os.path.join(d, "sec.dat"), "w") as f:
                f.write(dat)
            _run_xfoil(exe, d, _script(re_, mach, ncrit, start, a_end))
            rows += _parse_pol(os.path.join(d, "p.pol"))
            last = _last_attempted(d)
        if last is None or (a_end - last) * step <= 1e-9:
            break
        start = last + step
        if (a_end - start) * step < 0:
            break
    return rows


def _contiguous(rows, sign):
    """The run of points marching out from alpha 0 without a wide gap."""
    rows = sorted((r for r in rows if sign * r[0] >= -1e-9),
                  key=lambda r: sign * r[0])
    out, last = [], 0.0
    for r in rows:
        if abs(r[0] - last) > MAX_GAP_DEG:
            break
        if out and abs(r[0] - out[-1][0]) < 1e-9:
            continue
        out.append(r)
        last = r[0]
    return out


def _key(xy, re_, mach, ncrit, version):
    blob = json.dumps({
        "schema": 2,          # bump when the sweep logic changes what is kept
        "xy": [[round(x, 9), round(y, 9)] for x, y in xy],
        "re": float(f"{re_:.6e}"), "mach": round(mach, 6),
        "ncrit": round(ncrit, 6), "xfoil": version, "npan": NPAN,
        "iter": ITER, "step": STEP_DEG, "up": ALPHA_UP_DEG,
        "down": ALPHA_DOWN_DEG}, sort_keys=True)
    return hashlib.sha256(blob.encode()).hexdigest()[:24]


def _xfoil_polar(exe, version, xy, re_, mach, ncrit):
    """Cached (up, down) converged rows for one Re. Third value: cache hit.
    A cache file that does not parse is recomputed and replaced."""
    path = os.path.join(CACHE, _key(xy, re_, mach, ncrit, version) + ".json")
    try:
        with open(path) as f:
            c = json.load(f)
        return c["up"], c["down"], True
    except (FileNotFoundError, ValueError, KeyError):
        pass
    dat = "pesto section\n" + "".join(f"{x:.9f} {y:.9f}\n" for x, y in xy)
    up = _sweep(exe, dat, re_, mach, ncrit, ALPHA_UP_DEG)
    down = _sweep(exe, dat, re_, mach, ncrit, ALPHA_DOWN_DEG)
    tmp = path + f".{os.getpid()}.tmp"
    with open(tmp, "w") as f:
        json.dump({"re": re_, "mach": mach, "ncrit": ncrit, "xfoil": version,
                   "up": up, "down": down}, f)
    os.replace(tmp, path)              # atomic: parallel runs share the cache
    return up, down, False


# --- post-stall extension ---------------------------------------------------

def aspect_ratio(a):
    """Blade aspect ratio (hub surface to tip) / mean chord, for Viterna."""
    rh, R = a.hub_radius * 0.5, 0.5
    rs = np.linspace(rh, R, 65)
    chords = [propeller.blade_law(a, r)[0] for r in rs]
    return (R - rh) / float(np.mean(chords))


def cd_max(ar):
    """Viterna's flat-plate drag at 90 deg: 1.11 + 0.018 AR, AR capped at 50."""
    return 1.11 + 0.018 * min(ar, 50.0)


def _viterna(al, als, cls, cds, cdm):
    """Viterna-Corrigan cl, cd at al in [als, pi/2]; matches (cls, cds) at als."""
    sa, ca = math.sin(als), math.cos(als)
    a2 = (cls - cdm * sa * ca) * sa / ca ** 2
    b2 = (cds - cdm * sa ** 2) / ca
    cl = 0.5 * cdm * np.sin(2 * al) + a2 * np.cos(al) ** 2 / np.sin(al)
    cd = cdm * np.sin(al) ** 2 + b2 * np.cos(al)
    return cl, cd


def extend(al_deg, cl, cd, cdm):
    """Data on [alo, ahi] -> table on [-180, 180] deg (radians, increasing).

    Each side continues from its own end point (Viterna to +-90, then the
    reversed-flow branch at BACK_CL of it, ramping to cl = 0 at +-180 where
    both sides meet at one cd). Pre: alo < 0 < ahi < 90 deg. Continuous.
    """
    al = np.radians(np.asarray(al_deg, float))
    cl, cd = np.asarray(cl, float), np.asarray(cd, float)
    ahi, clh, cdh = al[-1], cl[-1], cd[-1]
    alo, cll, cdl = -al[0], -cl[0], cd[0]     # low side, mirrored to +alpha
    cd180 = 0.5 * (cdh + cdl)
    half = math.pi / 2

    def back(ap, a_s, cls, cds):
        """Reversed flow at 180 - ap, ap in [0, pi/2): sign-flipped side."""
        ap = np.asarray(ap, float)
        vcl, vcd = _viterna(np.maximum(ap, a_s), a_s, cls, cds, cdm)
        t = np.clip(ap / a_s, 0.0, 1.0)
        bcl = np.where(ap >= a_s, -BACK_CL * vcl, -BACK_CL * cls * t)
        bcd = np.where(ap >= a_s, vcd, cd180 + (cds - cd180) * t)
        return bcl, bcd

    def grid(lo, hi):
        return np.linspace(lo, hi, max(int(math.degrees(hi - lo)) + 1, 2))

    pos = grid(ahi, half)[1:]
    pcl, pcd = _viterna(pos, ahi, clh, cdh, cdm)
    neg = -grid(alo, half)[1:][::-1]
    ncl, ncd = _viterna(-neg, alo, cll, cdl, cdm)
    bh = grid(half, math.pi)[1:]                  # (90, 180]
    bhcl, bhcd = back(math.pi - bh, ahi, clh, cdh)
    bl = -grid(half, math.pi)[1:][::-1]           # [-180, -90)
    blcl, blcd = back(math.pi + bl, alo, cll, cdl)
    A = np.concatenate([bl, neg, al, pos, bh])
    CL = np.concatenate([-blcl, -ncl, cl, pcl, bhcl])
    CD = np.concatenate([blcd, ncd, cd, pcd, bhcd])
    assert np.all(np.diff(A) > 0), "extend: alpha table not increasing"
    return A, CL, CD


# --- the polar --------------------------------------------------------------

class Polar:
    """cl, cd over alpha in [-pi, pi] at a list of Re; one source per table.

    `tables[i]` = dict(alpha, cl, cd [radians, full circle], lo, hi [the
    XFOIL/linear range, deg], n [data points]) at `res[i]`.
    """

    def __init__(self, res, tables, source, mach=0.0):
        self.res = np.asarray(res, float)
        self.tables = tables
        self.source = source
        self.mach = mach
        self._lr = np.log(self.res)

    def cl_cd(self, alpha, re):
        """cl, cd at alpha [rad, any range] and Re; numpy-broadcast.

        Linear in alpha, linear in log Re between tables, Re clamped to the
        computed range (never extrapolated).
        """
        al, re = np.broadcast_arrays(np.asarray(alpha, float),
                                     np.asarray(re, float))
        w = np.mod(al + math.pi, 2.0 * math.pi) - math.pi
        cls = np.stack([np.interp(w, t["alpha"], t["cl"]) for t in self.tables])
        cds = np.stack([np.interp(w, t["alpha"], t["cd"]) for t in self.tables])
        if len(self.tables) == 1:
            return cls[0], cds[0]
        x = np.log(np.clip(re, self.res[0], self.res[-1]))
        k = np.clip(np.searchsorted(self._lr, x) - 1, 0, len(self.res) - 2)
        f = (x - self._lr[k]) / (self._lr[k + 1] - self._lr[k])
        pick = lambda v, j: np.take_along_axis(v, j[None], 0)[0]  # noqa: E731
        cl = (1 - f) * pick(cls, k) + f * pick(cls, k + 1)
        cd = (1 - f) * pick(cds, k) + f * pick(cds, k + 1)
        return cl, cd

    def table_at(self, re):
        """The table nearest Re in log."""
        return self.tables[int(np.argmin(abs(self._lr - math.log(re))))]

    @classmethod
    def from_thin_airfoil(cls, a, cd0=0.02, stall_deg=THIN_STALL_DEG):
        """Thin-airfoil fallback: 2 pi (alpha - alpha0), constant cd0, Viterna
        past +-stall_deg about alpha0. Re-independent. Only for --polar thin.
        """
        a0 = math.degrees(zero_lift_alpha(a))
        al = np.linspace(a0 - stall_deg, a0 + stall_deg, 41)
        cl = 2.0 * math.pi * np.radians(al - a0)
        cd = np.full_like(al, cd0)
        A, CL, CD = extend(al, cl, cd, cd_max(aspect_ratio(a)))
        t = dict(alpha=A, cl=CL, cd=CD, lo=al[0], hi=al[-1], n=len(al))
        return cls([1.0], [t], "thin")


def zero_lift_alpha(a):
    """Thin-airfoil zero-lift angle [rad] of the camber line (0 for the lens).

    alpha_L0 = -(1/pi) int_0^pi dyc/dx (cos th - 1) dth, x = (1 - cos th)/2.
    A dat section's camber line is the mean of its two sides (linear in x),
    against the file's own x axis -- the line beta is measured from.
    """
    if a.section == "dat":
        lo, up = (np.array(s) + [0.5, 0.0]
                  for s in propeller.ring_sides(propeller.section_unit(a)))
        x = np.linspace(0.0, 1.0, 4001)
        yc = 0.5 * (np.interp(x, up[:, 0], up[:, 1])
                    + np.interp(x, lo[:, 0], lo[:, 1]))
        th = np.arccos(1.0 - 2.0 * x)
        dyc = np.gradient(yc, x)
        return float(-np.trapezoid(dyc * (np.cos(th) - 1.0), th) / math.pi)
    if a.section != "naca" or a.camber == 0.0:
        return 0.0
    n = 4000
    s = 0.0
    for k in range(n):
        th = math.pi * (k + 0.5) / n
        x = 0.5 * (1.0 - math.cos(th))
        dy = propeller.naca4(x, a.thickness, a.camber, a.camber_pos)[1]
        s += dy * (math.cos(th) - 1.0)
    return -s / n


def load(a, re_list, mach=0.0, ncrit=9.0):
    """XFOIL polar of a's section at each Re in re_list, Viterna-extended.

    Pre: a passed `propeller.check_geometry`; re_list > 0. Exits (not
    raises) when XFOIL is missing or a Re does not converge around alpha 0 --
    at that point the table would be garbage and --polar thin is the honest
    alternative.
    """
    res = sorted({float(f"{r:.6e}") for r in re_list})
    if not res or res[0] <= 0.0:
        _die("Reynolds numbers must be > 0")
    if res[0] < RE_FLOOR:
        _die(f"Re {res[0]:.3g} is below XFOIL's floor {RE_FLOOR:.0e}: its "
             "integral boundary-layer closures and e^N transition are not "
             "calibrated there, and a 'converged' polar is not evidence "
             "(a 55%-thick lens converges at Re 1e3 to +25 deg). " + USE_THIN)
    if not 0.0 <= mach < 0.8:
        _die(f"Mach {mach:g} outside XFOIL's Karman-Tsien range [0, 0.8)")
    exe = find_xfoil(getattr(a, "xfoil", None))
    if exe is None:
        _die("xfoil not found (--xfoil, PATH, ~/.local/bin/xfoil). Build it "
             "per tools/polars.py's docstring, or pass --polar thin for the "
             "thin-airfoil model")
    version = xfoil_version(exe)
    xy = section_xy(a)
    os.makedirs(CACHE, exist_ok=True)
    jobs = min(len(res), os.cpu_count() or 1)
    with concurrent.futures.ThreadPoolExecutor(max_workers=jobs) as ex:
        runs = list(ex.map(lambda r: _xfoil_polar(exe, version, xy, r, mach,
                                                  ncrit), res))
    cdm = cd_max(aspect_ratio(a))
    tables, bad = [], []
    for r, (up, down, hit) in zip(res, runs):
        up, down = _contiguous(up, +1), _contiguous(down, -1)
        rows = sorted({round(x[0], 6): x for x in down + up}.values())
        lo = rows[0][0] if rows else 0.0
        hi = rows[-1][0] if rows else 0.0
        print(f"polars: xfoil {version} Re {r:.3g} M {mach:g} Ncrit "
              f"{ncrit:g}: {len(rows)} points, alpha [{lo:+.1f},{hi:+.1f}] deg"
              f"{' (cached)' if hit else ''}", file=sys.stderr)
        if lo > -2.0 or hi < 2.0:
            bad.append(r)
            continue
        al, cl, cd = zip(*rows)
        A, CL, CD = extend(al, cl, cd, cdm)
        tables.append(dict(alpha=A, cl=CL, cd=CD, lo=lo, hi=hi, n=len(rows)))
    if bad:
        _die("XFOIL did not converge around alpha = 0 (need at least "
             "[-2, +2] deg) at Re " + ", ".join(f"{r:.3g}" for r in bad)
             + ". Its e^N transition / laminar-separation models give out at "
             "low Re (typically below ~1e4) and on sharp, very thick sections. "
             + USE_THIN)
    return Polar(res, tables, "xfoil", mach)


def re_table(lo, hi, n):
    """At most n Re covering [lo, hi], on a FIXED log grid.

    The grid is 10^(k s) for the finest step s in RE_GRID that fits in n
    points, bracketing [lo, hi] outward, values to 2 significant digits. Fixed
    so the table -- and through it every interpolated cl -- does not move
    with the caller's J list, and so neighbouring runs share cached polars.
    """
    l0, l1 = math.log10(lo), math.log10(hi)
    best = None
    for s in RE_GRID:
        k0, k1 = math.floor(l0 / s + 1e-9), math.ceil(l1 / s - 1e-9)
        ks = range(k0, max(k1, k0 + 1) + 1)
        if best is None or len(ks) <= max(n, 2):
            best = [float(f"{10 ** (k * s):.2g}") for k in ks]
    return sorted(set(best))


def add_polar_args(p):
    """The polar flags, shared by every tool that needs section data."""
    p.add_argument("--polar", choices=("xfoil", "thin"), default="xfoil",
                   help="section data: XFOIL viscous polars, or the thin-"
                        "airfoil model (2 pi alpha, constant cd0) [xfoil]")
    p.add_argument("--ncrit", type=float, default=9.0,
                   help="XFOIL e^N transition amplification [9]")
    p.add_argument("--re-points", type=int, default=6,
                   help="Reynolds numbers in the polar table, log spaced "
                        "over the blade's range [6]")
    p.add_argument("--thin-cd0", type=float, default=0.02,
                   help="--polar thin: constant profile drag [0.02]")
    p.add_argument("--xfoil", default=None,
                   help="XFOIL executable [PATH, then ~/.local/bin/xfoil]")


def check_polar_args(a):
    """Reject polar flags no table can satisfy. Exits with a message."""
    if not 0.0 < a.ncrit <= 20.0:
        _die("--ncrit must be in (0, 20]")
    if a.re_points < 1:
        _die("--re-points must be >= 1")
    if not 0.0 < a.thin_cd0 < 1.0:
        _die("--thin-cd0 must be in (0, 1)")


def polar_from_args(a, re_list, mach=0.0):
    """The Polar the flags ask for, spanning min..max of re_list.

    XFOIL: Re below RE_FLOOR is clamped to the floor's polar WITH a warning
    when only part of the blade is down there (typically the root at J = 0);
    a blade wholly below it exits, pointing at --polar thin.
    """
    check_polar_args(a)
    if a.polar == "thin":
        return Polar.from_thin_airfoil(a, a.thin_cd0)
    lo, hi = min(re_list), max(re_list)
    if hi < RE_FLOOR:
        _die(f"the blade's Re range [{lo:.3g}, {hi:.3g}] lies below XFOIL's "
             f"floor {RE_FLOOR:.0e}; use --polar thin (thin-airfoil + "
             "Viterna, --thin-cd0)")
    if lo < RE_FLOOR:
        print(f"polars: warning: Re down to {lo:.3g} is below XFOIL's floor "
              f"{RE_FLOOR:.0e}; those stations use the Re {RE_FLOOR:.0e} "
              "polar (clamped)", file=sys.stderr)
        lo = RE_FLOOR
    return load(a, re_table(lo, hi, a.re_points), mach, a.ncrit)


# --- CLI ----------------------------------------------------------------------

def _floats(flag, s, n=None):
    """Comma-separated positive numbers (exactly n of them when n is set)."""
    try:
        v = [float(x) for x in s.split(",") if x.strip()]
    except ValueError:
        _die(f"{flag}: not a number list: {s!r}")
    if not v or min(v) <= 0 or (n is not None and len(v) != n):
        _die(f"{flag} needs {f'{n} ' if n else ''}positive values")
    return v


def export_alphas():
    """[-180, 180] deg: 1 deg steps, 0.25 deg inside +-25 (exact quarters)."""
    return [q / 4.0 for q in range(-720, 721) if abs(q) <= 100 or q % 4 == 0]


def write_export(path, pol, res):
    """CSV re,alpha_deg,cl,cd: Re blocks ascending, alpha ascending."""
    al = np.array(export_alphas())
    with open(path, "w") as f:
        f.write("re,alpha_deg,cl,cd\n")
        for r in sorted(set(res)):
            cl, cd = pol.cl_cd(np.radians(al), r)
            f.writelines(f"{r:.10g},{x:g},{c:.10g},{d:.10g}\n"
                         for x, c, d in zip(al, cl, cd))
    print(f"wrote {path}: {len(set(res))} Re x {len(al)} alpha ({pol.source})")


def write_inspection(path, pol, res, a):
    """CSV Re,alpha_deg,cl,cd,source at --step-deg, and the stdout summary."""
    al = np.arange(-180.0, 180.0 + 1e-9, a.step_deg)
    with open(path, "w") as f:
        f.write("Re,alpha_deg,cl,cd,source\n")
        for r in res:
            cl, cd = pol.cl_cd(np.radians(al), r)
            t = pol.table_at(r)
            for x, c, d in zip(al, cl, cd):
                src = pol.source if t["lo"] <= x <= t["hi"] else "viterna"
                f.write(f"{r:.6g},{x:.4f},{c:.6f},{d:.6f},{src}\n")
    print(f"wrote {path}: {len(res)} Re x {len(al)} alpha "
          f"(AR {aspect_ratio(a):.2f}, CDmax {cd_max(aspect_ratio(a)):.3f})")
    print("       Re   range [deg]    clmax  @deg    cdmin   (L/D)max  @deg")
    for r in res:
        cl, cd = pol.cl_cd(np.radians(al), r)
        t = pol.table_at(r)
        m = (al >= t["lo"]) & (al <= t["hi"])
        ld = np.where(m, cl / cd, -np.inf)
        i, j = int(np.argmax(np.where(m, cl, -np.inf))), int(np.argmax(ld))
        print(f"  {r:9.3g} [{t['lo']:+5.1f},{t['hi']:+5.1f}] {cl[i]:8.3f} "
              f"{al[i]:5.1f} {cd[m].min():8.5f} {ld[j]:9.1f} {al[j]:5.1f}")


def main():
    p = argparse.ArgumentParser(
        description="Section polars of the propeller blade (XFOIL + Viterna "
                    "to +-180 deg) -> CSV, for inspection (--out) or for "
                    "`dns prop` (--export).")
    propeller.add_geometry_args(p)
    add_polar_args(p)
    p.add_argument("--re", "--re-list", dest="re",
                   help="comma-separated Reynolds numbers, e.g. 1e5,2e5: the "
                        "tables are computed at exactly these")
    p.add_argument("--re-span", metavar="LO,HI",
                   help="instead of --re: the table a BEMT curve needs for a "
                        "blade spanning Re LO..HI (--re-points on the fixed "
                        "log grid, Re < 1e4 clamped)")
    p.add_argument("--mach", type=float, default=0.0)
    p.add_argument("--out", help="inspection CSV Re,alpha_deg,cl,cd,source")
    p.add_argument("--export", metavar="CSV",
                   help="dns prop input re,alpha_deg,cl,cd: table Re "
                        "ascending, alpha -180..180 deg, 1 deg steps, "
                        "0.25 inside +-25")
    p.add_argument("--step-deg", type=float, default=1.0,
                   help="alpha spacing of the --out CSV [1]")
    a = p.parse_args()
    if (a.re is None) == (a.re_span is None):
        p.error("give exactly one of --re/--re-list and --re-span")
    if not a.out and not a.export:
        p.error("need --out and/or --export")
    propeller.check_geometry(a)
    check_polar_args(a)
    if not 0.0 < a.step_deg <= 30.0:
        _die("--step-deg must be in (0, 30]")
    if a.re is not None:
        res = _floats("--re", a.re)
        pol = (Polar.from_thin_airfoil(a, a.thin_cd0) if a.polar == "thin"
               else load(a, res, a.mach, a.ncrit))
    else:
        res = sorted(_floats("--re-span", a.re_span, 2))
        pol = polar_from_args(a, res, a.mach)
    # XFOIL: the Re the tables sit at (thin is Re-independent: the given Re).
    tab = [float(r) for r in pol.res] if pol.source == "xfoil" else res
    if a.out:
        write_inspection(a.out, pol, res if a.re is not None else tab, a)
    if a.export:
        write_export(a.export, pol, tab)


if __name__ == "__main__":
    main()
