#!/usr/bin/env python3
"""Rotor CT/CQ split: hub vs blade, pressure vs convective vs viscous, for a
dns ghost-wall surface table and for an OpenFOAM case, on the SAME geometric
sort and the SAME normalization (masterplan_verify W6).

  split_dns.py dns  CASE RUN_surface.csv [--tol T] [--json OUT]
  split_dns.py foam CASE FOAM_CASE_DIR --work DIR [--reuse] [--tol T] [--json OUT]

CASE is a `tools/parity/cases.toml` rotor case: its `blade` flags rebuild the
propeller (geom.rotor_mesh) and give the hub prism (propeller.py --hub-radius,
--hub-length) in swept-diameter units. A face is HUB when r <= r_hub + T and
|x_axial| <= hub half-length + T, everything else is BLADE; T (in D) defaults
to sqrt(median face area) of the dns table -- the staircase sits up to a
cell off the true surface, so the dns hub sort needs that slack, and the foam
sort must be given the SAME T (`--tol`) so both put the same blade-root strip
on the hub.

dns: `dns flow ... --emit surface` on the immersed ghost wall writes one row
per interface face (docs/viewer-format.md): oriented area vector s, gauge
face pressure, and the full / pressure / convective / viscous face flux in
solver units. The header carries the pivot (shaft point), ctDen = rho n^2 D^4
and cqDen = rho n^2 D^5, so D = cqDen / ctDen. CT = -sum fx / ctDen and
CQ = -sum (r x f)_x / cqDen (the suite's sign: CT, CQ = -ct, -cq of the stats
file).

foam: post-processes a COPY under --work (never the run directory): it
reconstructs the latest time, writes per-face force fields at pRef 0 and 1
(their difference is S_f) and the face centres, then splits pressure
(p - p_inf) S_f from the rest (viscous; foam has no convective wall flux).
Run it inside gpurun (it runs OpenFOAM). Units: D = 1, n = result.json meta n,
rho = 1 (tier B, kinematic) or meta rho (tier Bc).
"""
import argparse
import json
import math
import os
import re
import shutil
import subprocess
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import geom  # noqa: E402
import suite  # noqa: E402

FOAM_ENV = "/opt/OpenFOAM/OpenFOAM-v2606/etc/bashrc"
EDGES = [0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.6]


def hub_geometry(case):
    """(hub circumradius, hub half-length) in swept-diameter units."""
    c = suite.Case(suite.load_cfg(), case)
    if c.kind != "rotor":
        sys.exit(f"split_dns: case '{case}' is not a rotor")
    a, m, _ = geom.rotor_mesh(c.c["blade"])
    sw = geom.swept_diameter(m)
    R = 0.5                                   # propeller.py's model tip radius
    return a.hub_radius * R / sw, a.hub_length * R / sw


def split(x, F, P, C, ctden, cqden, hub_r, hub_a, tol):
    """x: (n,3) face centres relative to the shaft point, in D; F/P/C: (n,3)
    total / pressure / convective forces. Returns the table dict."""
    r = np.hypot(x[:, 1], x[:, 2])
    hub = (r <= hub_r + tol) & (np.abs(x[:, 0]) <= hub_a + tol)
    V = F - P - C

    def ct(G, s):     # + 0.0: no "-0" for an identically zero part
        return float(-G[s, 0].sum() / ctden) + 0.0

    def cq(G, s):
        return float(-(x[s, 1] * G[s, 2] - x[s, 2] * G[s, 1]).sum() / cqden) + 0.0

    def part(s):
        return {"faces": int(s.sum()),
                "CT": ct(F, s), "CT_p": ct(P, s), "CT_c": ct(C, s), "CT_v": ct(V, s),
                "CQ": cq(F, s), "CQ_p": cq(P, s), "CQ_c": cq(C, s), "CQ_v": cq(V, s)}

    out = {"hub_r": hub_r, "hub_half": hub_a, "tol": tol,
           "total": part(np.ones_like(hub)), "hub": part(hub), "blade": part(~hub),
           "radial": []}
    lo = hub_r + tol
    for hi in EDGES:
        if hi <= lo:
            continue
        s = ~hub & (r > lo) & (r <= hi)
        out["radial"].append({"r": [lo, hi], **part(s)})
        lo = hi
    return out


def show(title, t):
    print(f"# {title}: hub r <= {t['hub_r']:.4f} + {t['tol']:.4f} D, "
          f"|x| <= {t['hub_half']:.4f} + {t['tol']:.4f} D")
    print(f"{'part':10s} {'faces':>6s} {'CT':>9s} {'CT_p':>9s} {'CT_c':>9s} {'CT_v':>9s}"
          f" {'CQ':>9s} {'CQ_p':>9s} {'CQ_c':>9s} {'CQ_v':>9s}")
    rows = [("total", t["total"]), ("hub", t["hub"]), ("blade", t["blade"])]
    rows += [(f"r{q['r'][0]:.3f}-{q['r'][1]:.2f}", q) for q in t["radial"]]
    for name, q in rows:
        print(f"{name:10s} {q['faces']:6d} {q['CT']:+9.5f} {q['CT_p']:+9.5f} "
              f"{q['CT_c']:+9.5f} {q['CT_v']:+9.5f} {q['CQ']:+9.5f} "
              f"{q['CQ_p']:+9.5f} {q['CQ_c']:+9.5f} {q['CQ_v']:+9.5f}")


# --- dns ----------------------------------------------------------------------

def read_dns(path):
    with open(path) as f:
        head = f.readline()
    if "immersed ghost wall" not in head:
        sys.exit(f"split_dns: {path} is not a ghost-wall surface table")

    def num(key):
        m = re.search(re.escape(key) + r"\s+([-+0-9.eE]+(?:\s+[-+0-9.eE]+)*)", head)
        if not m:
            sys.exit(f"split_dns: header has no '{key}'")
        return [float(v) for v in m.group(1).split()]

    pivot, ctden, cqden = num("pivot"), num("ctDen")[0], num("cqDen")[0]
    shaft = int(num("shaft")[0])
    if shaft != 0 or ctden <= 0:
        sys.exit("split_dns: needs a rotor about x (shaft 0, ctDen > 0)")
    a = np.genfromtxt(path, delimiter=",", names=True, skip_header=1)
    return a, np.array(pivot), ctden, cqden, head.strip()


def run_dns(args):
    a, pivot, ctden, cqden, head = read_dns(args.surface)
    D = cqden / ctden
    x = (np.stack([a["x"], a["y"], a["z"]], 1) - pivot) / D
    F = np.stack([a["fx"], a["fy"], a["fz"]], 1)
    P = np.stack([a["fpx"], a["fpy"], a["fpz"]], 1)
    C = np.stack([a["fcx"], a["fcy"], a["fcz"]], 1)
    tol = args.tol if args.tol is not None else math.sqrt(float(np.median(a["area"]))) / D
    hub_r, hub_a = hub_geometry(args.case)
    # x is in D, so the world moment is D times split()'s: cqDen / D.
    t = split(x, F, P, C, ctden, cqden / D, hub_r, hub_a, tol)
    t["source"] = {"surface": os.path.abspath(args.surface), "header": head,
                   "D": D, "area": float(a["area"].sum()) / D ** 2,
                   "sum_s": [float(a[k].sum()) for k in ("sx", "sy", "sz")]}
    show(f"dns {args.surface}", t)
    return t


# --- foam ---------------------------------------------------------------------

def patch_values(path, patch):
    s = open(path).read()
    i = s.index("boundaryField")
    j = s.index(patch, i)
    k = s.index("value", j)
    m = re.compile(r"nonuniform\s+List<(\w+)>\s*(\d+)\s*\(").search(s, k)
    if not m or m.start() - k > 200:
        sys.exit(f"split_dns: {path}: {patch} value is not a nonuniform list")
    n, vec = int(m.group(2)), m.group(1) == "vector"
    body = s[m.end():]
    nums = body.replace("(", " ").replace(")", " ").split()[: n * (3 if vec else 1)]
    return np.array(nums, float).reshape(n, -1)


def foam_scalars(text, pos):
    """The first `N ( v0 v1 ... )` scalar list in `text` at or after pos."""
    m = re.compile(r"(\d+)\s*\(").search(text, pos)
    n = int(m.group(1))
    return np.array(text[m.end():].split(None, n)[:n], float)


def patch_cell_values(case_dir, tl, field, patch):
    """A field's values in the cells owning `patch`'s faces, in face order:
    the face value of a zeroGradient patch, which writes none of its own."""
    b = open(os.path.join(case_dir, "constant", "polyMesh", "boundary")).read()
    j = re.search(r"\b" + re.escape(patch) + r"\s*\{", b).end()
    blk = b[j:b.index("}", j)]
    n = int(re.search(r"nFaces\s+(\d+);", blk).group(1))
    start = int(re.search(r"startFace\s+(\d+);", blk).group(1))
    own = open(os.path.join(case_dir, "constant", "polyMesh", "owner")).read()
    owner = foam_scalars(own, own.index("}")).astype(np.int64)
    f = open(os.path.join(case_dir, tl, field)).read()
    vals = foam_scalars(f, f.index("internalField"))
    return vals[owner[start:start + n]]


def patch_p(case_dir, tl, patch):
    """Face pressure on the wall patch: its written value, or the owner
    cells' (zeroGradient)."""
    path = os.path.join(case_dir, tl, "p")
    s = open(path).read()
    j = s.index(patch, s.index("boundaryField"))
    blk = s[j:s.index("}", j)]
    if "zeroGradient" in blk:
        return patch_cell_values(case_dir, tl, "p", patch)
    return patch_values(path, patch)[:, 0]


def uniform_internal(path):
    m = re.search(r"internalField\s+uniform\s+([-+0-9.eE]+)\s*;", open(path).read())
    if not m:
        sys.exit(f"split_dns: {path}: internalField is not a uniform scalar")
    return float(m.group(1))


def foam(cmd, cwd, log):
    with open(os.path.join(cwd, log), "w") as f:
        rc = subprocess.run(["bash", "-c", f"source {FOAM_ENV} >/dev/null 2>&1; {cmd}"],
                            cwd=cwd, stdout=f, stderr=subprocess.STDOUT).returncode
    if rc != 0:
        sys.exit(f"split_dns: '{cmd}' failed (rc {rc}), see {cwd}/{log}")


def postprocess(src, w, procs, tl, app, comp):
    """Copy the latest time of `src` into `w`, reconstruct, and write the
    per-face force fields (pRef 0 and 1) and the face centres there."""
    shutil.rmtree(w, ignore_errors=True)
    os.makedirs(w)
    for d in ("system", "constant", "0"):
        shutil.copytree(os.path.join(src, d), os.path.join(w, d))
    for p in procs:
        os.makedirs(os.path.join(w, p))
        for d in ("constant", tl):
            shutil.copytree(os.path.join(src, p, d), os.path.join(w, p, d))
    foam("reconstructPar -latestTime", w, "log.reconstruct")
    for p in procs:
        shutil.rmtree(os.path.join(w, p))
    rho = "rho rho;" if comp else "rho rhoInf; rhoInf 1;"
    cd = os.path.join(w, "system", "controlDict")
    s = open(cd).read()
    i = s.index("functions")
    fo = f"""functions
{{
    fA {{ type forces; libs (forces); patches (propeller); {rho}
         CofR (0 0 0); pRef 0; writeFields yes; useNamePrefix true; log false; }}
    fB {{ type forces; libs (forces); patches (propeller); {rho}
         CofR (0 0 0); pRef 1; writeFields yes; useNamePrefix true; log false; }}
    cc {{ type writeCellCentres; libs (fieldFunctionObjects); }}
}}
"""
    open(cd, "w").write(s[:i] + fo)
    foam(f"{app} -postProcess -latestTime", w, "log.pp")


def run_foam(args):
    src = os.path.abspath(args.foam_case)
    res = json.load(open(os.path.join(src, "result.json")))
    meta, app = res["meta"], res["app"]
    procs = sorted(d for d in os.listdir(src) if d.startswith("processor"))
    times = sorted((d for d in os.listdir(os.path.join(src, procs[0]))
                    if re.fullmatch(r"[0-9.]+", d) and d != "0"), key=float)
    tl = times[-1]
    w = os.path.abspath(args.work)
    if w.startswith(os.path.abspath(os.path.join(HERE, "..", ".."))):
        sys.exit("split_dns: --work must be outside the repository")
    comp = app != "simpleFoam"
    if args.reuse:
        # --reuse never runs OpenFOAM: it may be called outside gpurun.
        if not os.path.exists(os.path.join(w, tl, "fA:force")):
            sys.exit(f"split_dns: --reuse but {w}/{tl} has no fA:force")
    else:
        postprocess(src, w, procs, tl, app, comp)
    td = os.path.join(w, tl)
    FA =patch_values(os.path.join(td, "fA:force"), "propeller")
    FB = patch_values(os.path.join(td, "fB:force"), "propeller")
    C = patch_values(os.path.join(td, "C"), "propeller")
    p = patch_p(w, tl, "propeller")
    pinf = uniform_internal(os.path.join(w, "0", "p"))
    Sf = FA - FB                       # unit pRef shift
    P = (p - pinf)[:, None] * Sf
    F = FA - pinf * Sf                 # gauge total
    rhoN = (meta["rho"] if comp else 1.0) * meta["n"] ** 2
    hub_r, hub_a = hub_geometry(args.case)
    tol = args.tol if args.tol is not None else 0.02
    t = split(C, F, P, np.zeros_like(F), rhoN, rhoN, hub_r, hub_a, tol)
    t["source"] = {"case_dir": src, "app": app, "time": tl, "p_inf": pinf,
                   "observables": res.get("observables"),
                   "area": float(np.linalg.norm(Sf, axis=1).sum()),
                   "sum_s": Sf.sum(0).tolist()}
    show(f"foam {src} ({app}, time {tl})", t)
    print(f"# result.json observables: {res.get('observables')}")
    return t


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("dns")
    d.add_argument("case")
    d.add_argument("surface")
    f = sub.add_parser("foam")
    f.add_argument("case")
    f.add_argument("foam_case")
    f.add_argument("--work", required=True)
    f.add_argument("--reuse", action="store_true",
                   help="skip OpenFOAM when --work already holds the fields")
    for p in (d, f):
        p.add_argument("--tol", type=float, default=None, help="hub slack in D")
        p.add_argument("--json", default=None)
    args = ap.parse_args()
    t = run_dns(args) if args.cmd == "dns" else run_foam(args)
    if args.json:
        with open(args.json, "w") as fh:
            json.dump(t, fh, indent=1)


if __name__ == "__main__":
    main()
