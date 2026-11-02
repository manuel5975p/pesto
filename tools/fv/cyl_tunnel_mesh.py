#!/usr/bin/env python3
"""The MAC tunnel as a body-fitted polyMesh (docs/fvincomp.md).

`dns flow --body cylinder --far tunnel --n N --dr 8` runs on a uniform h = L/N
box with an immersed circle. This writes the same box, the same h and the same
circle (the same off-node placement, L/3 + 0.31 h, L/2 + 0.17 h) as a 2D
polyMesh: the Cartesian lattice with a square hole around the body, filled by
an O-ring whose outer nodes ARE the lattice nodes (conformal, all quads, one
cell thick, `empty` front and back), plus the --bc table of the tunnel's own
boundary conditions (inflow, free-slip laterals, convective outlet).

    tools/fv/cyl_tunnel_mesh.py --n 256 --out /tmp/dns/incomp/tun256

Then:  dns flow --polymesh OUT --bc OUT/bc.toml --re 40 --ft 96 ...
(--ft counts D / U here: the MAC run's --ft F is 8 F.)
"""
import argparse
import math
import os

HEADER = """FoamFile
{{
    version     2.0;
    format      ascii;
    arch        "LSB;label=32;scalar=64";{note}
    class       {cls};
    location    "constant/polyMesh";
    object      {obj};
}}
"""


def write_list(path, obj, cls, items, note=""):
    with open(path, "w") as f:
        f.write(HEADER.format(note=note, cls=cls, obj=obj))
        f.write(f"\n{len(items)}\n(\n")
        f.write("\n".join(items))
        f.write("\n)\n")


def build(n, half_d, beta, nr_factor, dr):
    L = 2 * math.pi
    h = L / n
    D = L / dr
    cx, cy = L / 3 + 0.31 * h, L / 2 + 0.17 * h
    a = half_d * D
    i0, i1 = math.floor((cx - a) / h), math.ceil((cx + a) / h)
    j0, j1 = math.floor((cy - a) / h), math.ceil((cy + a) / h)
    nid = {}
    pts = []

    def lattice(i, j):
        k = (i, j)
        if k not in nid:
            nid[k] = len(pts)
            pts.append((i * h, j * h))
        return nid[k]

    quads = []
    for j in range(n):
        for i in range(n):
            if i0 <= i < i1 and j0 <= j < j1:
                continue
            quads.append((lattice(i, j), lattice(i + 1, j), lattice(i + 1, j + 1), lattice(i, j + 1)))
    # The hole's perimeter, counter-clockwise.
    per = [(i, j0) for i in range(i0, i1)] + [(i1, j) for j in range(j0, j1)]
    per += [(i, j1) for i in range(i1, i0, -1)] + [(i0, j) for j in range(j1, j0, -1)]
    R = 0.5 * D
    nr = max(3, round(nr_factor * (a - R) / h))

    def dist(t):
        return 1 + math.tanh(beta * (t - 1)) / math.tanh(beta) if beta > 0 else t

    rings = []  # rings[k][p]: k = 0 the wall, k = nr the perimeter
    for k in range(nr + 1):
        row = []
        for p, (i, j) in enumerate(per):
            if k == nr:
                row.append(lattice(i, j))
                continue
            x, y = i * h, j * h
            th = math.atan2(y - cy, x - cx)
            wx, wy = cx + R * math.cos(th), cy + R * math.sin(th)
            s = dist(k / nr)
            row.append(len(pts))
            pts.append((wx + s * (x - wx), wy + s * (y - wy)))
        rings.append(row)
    m = len(per)
    for k in range(nr):
        for p in range(m):
            q = (p + 1) % m
            quads.append((rings[k][p], rings[k + 1][p], rings[k + 1][q], rings[k][q]))
    return L, D, h, pts, quads


def write_mesh(out, L, h, pts, quads, depth):
    eps = 1e-9 * h
    # edge (a, b) of a cell, counter-clockwise: the cell lies on its left
    edges = {}
    for c, q in enumerate(quads):
        for e in range(4):
            a, b = q[e], q[(e + 1) % 4]
            k = (a, b) if a < b else (b, a)
            edges.setdefault(k, []).append((c, a, b))
    internal, patches = [], {"inlet": [], "outlet": [], "lower": [], "upper": [], "cylinder": []}
    for k, v in edges.items():
        if len(v) == 2:
            (c0, a0, b0), (c1, _, _) = v
            if c0 > c1:
                (c0, a0, b0), (c1, _, _) = v[1], v[0]
            internal.append((c0, c1, a0, b0))
            continue
        c, a, b = v[0]
        xa, ya = pts[a]
        xb, yb = pts[b]
        if abs(xa) < eps and abs(xb) < eps:
            patches["inlet"].append((c, a, b))
        elif abs(xa - L) < eps and abs(xb - L) < eps:
            patches["outlet"].append((c, a, b))
        elif abs(ya) < eps and abs(yb) < eps:
            patches["lower"].append((c, a, b))
        elif abs(ya - L) < eps and abs(yb - L) < eps:
            patches["upper"].append((c, a, b))
        else:
            patches["cylinder"].append((c, a, b))
    internal.sort()
    np2 = len(pts)
    faces, owner, nei = [], [], []
    for c0, c1, a, b in internal:
        faces.append(f"4({a} {b} {b + np2} {a + np2})")
        owner.append(c0)
        nei.append(c1)
    boundary = []
    start = len(faces)
    for name in ("inlet", "outlet", "lower", "upper", "cylinder"):
        fl = sorted(patches[name])
        for c, a, b in fl:
            faces.append(f"4({a} {b} {b + np2} {a + np2})")
            owner.append(c)
        kind = "wall" if name == "cylinder" else "patch"
        boundary.append((name, kind, len(fl), start))
        start += len(fl)
    fr = []
    for c, q in enumerate(quads):
        faces.append(f"4({q[3]} {q[2]} {q[1]} {q[0]})")
        owner.append(c)
        fr.append(c)
    boundary.append(("back", "empty", len(quads), start))
    start += len(quads)
    for c, q in enumerate(quads):
        faces.append(f"4({q[0] + np2} {q[1] + np2} {q[2] + np2} {q[3] + np2})")
        owner.append(c)
    boundary.append(("front", "empty", len(quads), start))
    d = os.path.join(out, "constant", "polyMesh")
    os.makedirs(d, exist_ok=True)
    note = (f'\n    note        "nPoints:{2 * np2}  nCells:{len(quads)}  nFaces:{len(faces)}  '
            f'nInternalFaces:{len(internal)}";')
    write_list(os.path.join(d, "points"), "points", "vectorField",
               [f"({x:.17g} {y:.17g} 0)" for x, y in pts] + [f"({x:.17g} {y:.17g} {depth:.17g})" for x, y in pts])
    write_list(os.path.join(d, "faces"), "faces", "faceList", faces)
    write_list(os.path.join(d, "owner"), "owner", "labelList", [str(c) for c in owner], note)
    write_list(os.path.join(d, "neighbour"), "neighbour", "labelList", [str(c) for c in nei], note)
    with open(os.path.join(d, "boundary"), "w") as f:
        f.write(HEADER.format(note="", cls="polyBoundaryMesh", obj="boundary"))
        f.write(f"\n{len(boundary)}\n(\n")
        for name, kind, nf, st in boundary:
            f.write(f"    {name}\n    {{\n        type            {kind};\n        nFaces          {nf};\n"
                    f"        startFace       {st};\n    }}\n")
        f.write(")\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, required=True, help="cells per L, the MAC --n")
    ap.add_argument("--dr", type=float, default=8, help="L / D, the MAC --dr")
    ap.add_argument("--half", type=float, default=0.75, help="half side of the O-ring square / D")
    ap.add_argument("--beta", type=float, default=0.0, help="tanh wall clustering of the ring (0: uniform)")
    ap.add_argument("--nr", type=float, default=1.0, help="ring cells per h of the gap body - square side")
    ap.add_argument("--depth", type=float, default=1.0, help="the one cell's thickness")
    ap.add_argument("--out", required=True)
    o = ap.parse_args()
    L, D, h, pts, quads = build(o.n, o.half, o.beta, o.nr, o.dr)
    write_mesh(o.out, L, h, pts, quads, o.depth)
    with open(os.path.join(o.out, "bc.toml"), "w") as f:
        f.write(f"""# The MAC tunnel (docs/fvincomp.md): --ft counts D / U.
[reference]
u = 1.0
length = {D:.17g}
area = {D * o.depth:.17g}

[initial]
u = [1, 0, 0]

[patch.inlet]
kind = "inflow"
u = [1, 0, 0]

[patch.outlet]
kind = "convective"
uc = 1.0

[patch.lower]
kind = "slip"

[patch.upper]
kind = "slip"

[patch.cylinder]
kind = "wall"
force = true
""")
    print(f"{o.out}: {len(quads)} cells, h = {h:.5g}, D = {D:.5g}")


if __name__ == "__main__":
    main()
