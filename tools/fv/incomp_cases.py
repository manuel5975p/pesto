#!/usr/bin/env python3
"""FV vs simpleFoam on one mesh, one BC set (docs/fvincomp.md, "Same mesh").

    incomp_cases.py cavity N OUT            the stored cavity's dictionaries at N^2
    incomp_cases.py tunnel N OUT            cyl_tunnel_mesh.py + central simpleFoam dictionaries
    incomp_cases.py ogrid NTH NR OUT [FAR BETA]   `--ogrid` export, far split at x = 0 into
                                            farA, farI, farB: fixed inflow / fixed p

Each OUT is a case: `dns flow --polymesh OUT --fv-precision fp64 --max-iters I --out S`
writes simpleFoam's fixed point; `test_gpu_fv_incomp fixedpoint OUT OUT/bc.toml S/I/U
NU T CFL` (or `cavity OUT S/I/U T CFL`) marches from it. tunnel also writes
bc_conv.toml, the MAC tunnel's convective outlet.
"""
import math
import os
import re
import shutil
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import cyl_tunnel_mesh as ctm

ROOT = os.path.dirname(os.path.dirname(HERE))
DNS = os.environ.get("DNS", os.path.join(ROOT, "build", "dns"))
IN = ("type fixedValue; value uniform (1 0 0);", "type zeroGradient;")
OUT = ("type inletOutlet; inletValue uniform (0 0 0); value uniform (1 0 0);", "type fixedValue; value uniform 0;")
WALL = ("type noSlip;", "type zeroGradient;")
SLIP = ("type slip;", "type slip;")
EMPTY = ("type empty;", "type empty;")


def cavity(n, out):
    src = os.path.join(ROOT, "tests", "data", "fv", "cavity")
    shutil.rmtree(out, ignore_errors=True)
    for sub in ("0", "system", "constant"):
        shutil.copytree(f"{src}/{sub}", f"{out}/{sub}")
    shutil.rmtree(f"{out}/constant/polyMesh")
    length, depth, h = 0.1, 0.01, 0.1 / n
    nid = lambda i, j: j * (n + 1) + i
    pts = [(i * h, j * h) for j in range(n + 1) for i in range(n + 1)]
    quads = [(nid(i, j), nid(i + 1, j), nid(i + 1, j + 1), nid(i, j + 1)) for j in range(n) for i in range(n)]
    edges = {}
    for c, q in enumerate(quads):
        for e in range(4):
            a, b = q[e], q[(e + 1) % 4]
            edges.setdefault((min(a, b), max(a, b)), []).append((c, a, b))
    internal, moving, fixed = [], [], []
    for v in edges.values():
        if len(v) == 2:
            (c0, a0, b0), (c1, _, _) = sorted(v)
            internal.append((c0, c1, a0, b0))
        else:
            c, a, b = v[0]
            lid = abs(pts[a][1] - length) < 1e-9 * h and abs(pts[b][1] - length) < 1e-9 * h
            (moving if lid else fixed).append((c, a, b))
    internal.sort()
    np2 = len(pts)
    faces, owner, nei = [], [], []
    for c0, c1, a, b in internal:
        faces.append(f"4({a} {b} {b + np2} {a + np2})")
        owner.append(c0)
        nei.append(c1)
    boundary, start = [], len(faces)
    for name, fl in (("movingWall", sorted(moving)), ("fixedWalls", sorted(fixed))):
        for c, a, b in fl:
            faces.append(f"4({a} {b} {b + np2} {a + np2})")
            owner.append(c)
        boundary.append((name, "wall", len(fl), start))
        start += len(fl)
    for c, q in enumerate(quads):
        faces.append(f"4({q[3]} {q[2]} {q[1]} {q[0]})")
        owner.append(c)
    for c, q in enumerate(quads):
        faces.append(f"4({q[0] + np2} {q[1] + np2} {q[2] + np2} {q[3] + np2})")
        owner.append(c)
    boundary.append(("frontAndBack", "empty", 2 * len(quads), start))
    d = f"{out}/constant/polyMesh"
    os.makedirs(d)
    note = f'\n    note        "nPoints:{2 * np2}  nCells:{len(quads)}  nFaces:{len(faces)}  nInternalFaces:{len(internal)}";'
    ctm.write_list(f"{d}/points", "points", "vectorField",
                   [f"({x:.17g} {y:.17g} 0)" for x, y in pts] + [f"({x:.17g} {y:.17g} {depth})" for x, y in pts])
    ctm.write_list(f"{d}/faces", "faces", "faceList", faces)
    ctm.write_list(f"{d}/owner", "owner", "labelList", [str(c) for c in owner], note)
    ctm.write_list(f"{d}/neighbour", "neighbour", "labelList", [str(c) for c in nei], note)
    with open(f"{d}/boundary", "w") as f:
        f.write(ctm.HEADER.format(note="", cls="polyBoundaryMesh", obj="boundary"))
        f.write(f"\n{len(boundary)}\n(\n")
        for name, kind, nf, st in boundary:
            f.write(f"    {name}\n    {{\n        type {kind};\n        nFaces {nf};\n        startFace {st};\n    }}\n")
        f.write(")\n")
    p = f"{out}/system/fvSolution"
    s = open(p).read()
    s = s.replace("tolerance 1e-13; relTol 0;", "tolerance 1e-11; relTol 0.01;")
    s = s.replace("tolerance 1e-14; relTol 0;", "tolerance 1e-11; relTol 0.01;")
    open(p, "w").write(s)


def central(o):
    p = f"{o}/system/fvSchemes"
    s = open(p).read().replace("bounded Gauss linearUpwind grad(U)", "bounded Gauss linear")
    open(p, "w").write(s)
    p = f"{o}/system/fvSolution"
    s = open(p).read().replace("nNonOrthogonalCorrectors 1", "nNonOrthogonalCorrectors 2")
    open(p, "w").write(s)


def fields(o, patches):
    head = lambda cls, ob: f"FoamFile\n{{\n    version 2.0;\n    format ascii;\n    class {cls};\n    object {ob};\n}}\n"
    u = head("volVectorField", "U") + "dimensions [0 1 -1 0 0 0 0];\ninternalField uniform (1 0 0);\nboundaryField\n{\n"
    p = head("volScalarField", "p") + "dimensions [0 2 -2 0 0 0 0];\ninternalField uniform 0;\nboundaryField\n{\n"
    for n, a, b in patches:
        u += f"    {n} {{ {a} }}\n"
        p += f"    {n} {{ {b} }}\n"
    open(f"{o}/0/U", "w").write(u + "}\n")
    open(f"{o}/0/p", "w").write(p + "}\n")


def export(o, n, nr, far=20, beta=2):
    subprocess.check_call([DNS, "flow", "--solver", "simple", "--ogrid", "--body", "cylinder", "--n", str(n), "--ny", str(nr),
                           "--re", "40", "--ogrid-far", str(far), "--ogrid-beta", str(beta), "--export", o],
                          stdout=subprocess.DEVNULL)


def ogrid(n, nr, o, far, beta):
    shutil.rmtree(o, ignore_errors=True)
    export(o, n, nr, far, beta)
    central(o)
    p = f"{o}/constant/polyMesh/boundary"
    s = open(p).read()
    m = re.search(r"far\s*\{\s*type\s+patch;\s*nFaces\s+(\d+);\s*startFace\s+(\d+);\s*\}", s)
    nf, st = int(m.group(1)), int(m.group(2))
    a, b = nf // 4, 3 * nf // 4
    blk = lambda name, c, f: f"far{name}\n    {{\n        type patch;\n        nFaces {c};\n        startFace {f};\n    }}"
    s = s[:m.start()] + blk("A", a, st) + "\n    " + blk("I", b - a, st + a) + "\n    " + blk("B", nf - b, st + b) + s[m.end():]
    open(p, "w").write(re.sub(r"\n4\n\(", "\n6\n(", s, count=1))
    fields(o, [("body",) + WALL, ("farA",) + OUT, ("farI",) + IN, ("farB",) + OUT, ("front",) + EMPTY, ("back",) + EMPTY])
    open(f"{o}/bc.toml", "w").write(f"""[reference]
u = 1.0
length = {math.pi / 4:.17g}
area = {math.pi / 4:.17g}
[initial]
u = [1, 0, 0]
[patch.body]
kind = "wall"
force = true
[patch.farA]
kind = "outflow"
[patch.farI]
kind = "inflow"
u = [1, 0, 0]
[patch.farB]
kind = "outflow"
""")


def tunnel(n, o):
    shutil.rmtree(o, ignore_errors=True)
    subprocess.check_call([sys.executable, f"{HERE}/cyl_tunnel_mesh.py", "--n", str(n), "--out", o], stdout=subprocess.DEVNULL)
    tmp = o + ".tmpl"
    shutil.rmtree(tmp, ignore_errors=True)
    export(tmp, 32, 16)
    shutil.copytree(f"{tmp}/system", f"{o}/system")
    os.makedirs(f"{o}/0")
    for f in ("transportProperties", "turbulenceProperties"):
        shutil.copy(f"{tmp}/constant/{f}", f"{o}/constant/{f}")
    shutil.rmtree(tmp)
    central(o)
    p = f"{o}/system/controlDict"
    s = open(p).read().replace("patches     (body)", "patches     (cylinder)")
    open(p, "w").write(s)
    fields(o, [("inlet",) + IN, ("outlet",) + OUT, ("lower",) + SLIP, ("upper",) + SLIP, ("cylinder",) + WALL,
               ("back",) + EMPTY, ("front",) + EMPTY])
    s = open(f"{o}/bc.toml").read()
    open(f"{o}/bc_conv.toml", "w").write(s)
    open(f"{o}/bc.toml", "w").write(s.replace('kind = "convective"\nuc = 1.0', 'kind = "outflow"'))


def main():
    a = sys.argv[1:]
    if len(a) == 3 and a[0] == "cavity":
        cavity(int(a[1]), a[2])
    elif len(a) == 3 and a[0] == "tunnel":
        tunnel(int(a[1]), a[2])
    elif len(a) in (4, 6) and a[0] == "ogrid":
        ogrid(int(a[1]), int(a[2]), a[3], *(a[4:] or (20, 2)))
    else:
        sys.exit(__doc__)


if __name__ == "__main__":
    main()
