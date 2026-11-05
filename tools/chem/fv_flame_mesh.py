#!/usr/bin/env python3
"""A 2D polyMesh for the planar flame on the FV path (docs/fvcomp.md "Species").

A box [0, Lx] x [0, Ly], one cell thick (`empty`), x-lo and x-hi open, y cyclic,
mapped so the flame plane stops being a mesh plane:

    --shear DEG   y' = y + x tan(DEG): j lines tilted, every face non-orthogonal
    --wavy A      x' = x + A h sin(2 pi y/Ly) sin^2(pi x/Lx): i lines cross the flame,
                  ends flat
    --rotate DEG  the whole mesh about z: S, d and the flame normal off the axes

Writes OUT/constant/polyMesh and OUT/bc.toml (x-lo fresh Far, x-hi burnt Outflow,
direction = the rotated x axis; with --jet the slot jet of `dns flame --mode diffusion`:
x-lo jet Inflow, x-hi Outflow at p0, origin the slot centre at y = Ly/2). Then:

    tools/chem/fv_flame_mesh.py --nx 362 --ny 32 --lx 38.6532 --ly 3.41688 --shear 30 --out DIR
    dns flame --polymesh DIR --bc DIR/bc.toml --upstream U --inflow-speed 0.98 --tt 20

--lowmach writes the low-Mach file (inflow y = --inflow-y, p = 0 outflow); --lowmach --jet the
low-Mach slot burner (`dns flame --physics lowmach --mode bunsen --polymesh`): xlo jet = true.
--open-y (with --lowmach): ylo, yhi plain patches of type = "open" at the ambient --ambient-t,
--ambient-y (the burner in a room: `--laterals open` on a polyMesh).

--nz N (N >= 2): a 3D mesh, N cells over --depth, z cyclic (patches zlo, zhi) instead of the
empty pair: the polyMesh of `dns flame --les` (LES needs a 3D mesh). --premix: the compressible
file without y (`dns flame --equivalence` sets the premix on every open patch).
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


def main():
    a = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    a.add_argument("--nx", type=int, required=True)
    a.add_argument("--ny", type=int, required=True)
    a.add_argument("--lx", type=float, required=True)
    a.add_argument("--ly", type=float, required=True)
    a.add_argument("--shear", type=float, default=0.0)
    a.add_argument("--wavy", type=float, default=0.0)
    a.add_argument("--rotate", type=float, default=0.0)
    a.add_argument("--depth", type=float, default=1.0)
    a.add_argument("--lowmach", action="store_true", help="bc.toml for --physics lowmach: inflow and p = 0 outflow")
    a.add_argument("--jet", action="store_true")
    a.add_argument("--inflow-y", default="1.0",
                   help="--lowmach: the inflow's y list (--equivalence: Y_F1,Y_O1 of the premix)")
    a.add_argument("--open-y", action="store_true", help="--lowmach: ylo, yhi open faces, not cyclic")
    a.add_argument("--ambient-t", type=float, default=1.0, help="--open-y: the ambient T / Tu")
    a.add_argument("--ambient-y", default="0.0", help="--open-y: the ambient's y list")
    a.add_argument("--nz", type=int, default=1, help="cells along z over --depth; >= 2: z cyclic (3D)")
    a.add_argument("--premix", action="store_true", help="the compressible far/outflow file without y (--equivalence)")
    a.add_argument("--out", required=True)
    o = a.parse_args()
    if o.nz > 1:
        if o.lowmach or o.jet or o.open_y:
            raise SystemExit("--nz > 1 is the compressible planar flame's mesh (no --lowmach, --jet, --open-y)")
        return main3d(o)
    nx, ny, lx, ly = o.nx, o.ny, o.lx, o.ly
    hx, hy = lx / nx, ly / ny
    ts = math.tan(math.radians(o.shear))
    th = math.radians(o.rotate)
    ct, st = math.cos(th), math.sin(th)

    def xy(i, j):
        x, y = i * hx, j * hy
        xm = x + o.wavy * hx * math.sin(2 * math.pi * y / ly) * math.sin(math.pi * x / lx) ** 2
        ym = y + x * ts
        return ct * xm - st * ym, st * xm + ct * ym

    pid = lambda i, j, k: (i + (nx + 1) * j) * 2 + k
    pts = []
    for j in range(ny + 1):
        for i in range(nx + 1):
            px, py = xy(i, j)
            for k in (0, 1):
                pts.append(f"({px:.17g} {py:.17g} {k * o.depth:.17g})")

    cid = lambda i, j: i + nx * j
    faces, own, nei = [], [], []

    def q(*p):
        return "4(" + " ".join(str(v) for v in p) + ")"

    # Internal faces, upper-triangular: owners ascending, neighbours ascending.
    for j in range(ny):
        for i in range(nx):
            if i + 1 < nx:
                faces.append(q(pid(i + 1, j, 0), pid(i + 1, j + 1, 0), pid(i + 1, j + 1, 1), pid(i + 1, j, 1)))
                own.append(cid(i, j))
                nei.append(cid(i + 1, j))
            if j + 1 < ny:
                faces.append(q(pid(i, j + 1, 0), pid(i, j + 1, 1), pid(i + 1, j + 1, 1), pid(i + 1, j + 1, 0)))
                own.append(cid(i, j))
                nei.append(cid(i, j + 1))
    n_int = len(faces)

    patches = []

    def patch(name, body, flist):
        patches.append((name, body, len(flist), len(faces)))
        for fv, c in flist:
            faces.append(fv)
            own.append(c)

    patch("xlo", "type patch;",
          [(q(pid(0, j, 0), pid(0, j, 1), pid(0, j + 1, 1), pid(0, j + 1, 0)), cid(0, j)) for j in range(ny)])
    patch("xhi", "type patch;",
          [(q(pid(nx, j, 0), pid(nx, j + 1, 0), pid(nx, j + 1, 1), pid(nx, j, 1)), cid(nx - 1, j)) for j in range(ny)])
    sep = (-st * ly, ct * ly, 0.0)
    if o.open_y and not o.lowmach:
        raise SystemExit("--open-y needs --lowmach (the open face kind is the low-Mach route's)")
    yb = ("type patch;", "type patch;") if o.open_y else (
        f"type cyclic; neighbourPatch yhi; transform translational; separationVector ({sep[0]:.17g} {sep[1]:.17g} 0);",
        f"type cyclic; neighbourPatch ylo; transform translational; separationVector ({-sep[0]:.17g} {-sep[1]:.17g} 0);")
    patch("ylo", yb[0],
          [(q(pid(i, 0, 0), pid(i + 1, 0, 0), pid(i + 1, 0, 1), pid(i, 0, 1)), cid(i, 0)) for i in range(nx)])
    patch("yhi", yb[1],
          [(q(pid(i, ny, 0), pid(i, ny, 1), pid(i + 1, ny, 1), pid(i + 1, ny, 0)), cid(i, ny - 1)) for i in range(nx)])
    fb = []
    for j in range(ny):
        for i in range(nx):
            fb.append((q(pid(i, j, 0), pid(i, j + 1, 0), pid(i + 1, j + 1, 0), pid(i + 1, j, 0)), cid(i, j)))
            fb.append((q(pid(i, j, 1), pid(i + 1, j, 1), pid(i + 1, j + 1, 1), pid(i, j + 1, 1)), cid(i, j)))
    patch("frontAndBack", "type empty;", fb)

    d = os.path.join(o.out, "constant", "polyMesh")
    os.makedirs(d, exist_ok=True)
    note = f'\n    note        "nPoints:{len(pts)} nCells:{nx * ny} nFaces:{len(faces)} nInternalFaces:{n_int}";'
    write_list(os.path.join(d, "points"), "points", "vectorField", pts)
    write_list(os.path.join(d, "faces"), "faces", "faceList", faces)
    write_list(os.path.join(d, "owner"), "owner", "labelList", [str(v) for v in own], note)
    write_list(os.path.join(d, "neighbour"), "neighbour", "labelList", [str(v) for v in nei], note)
    bnd = [f"    {n}\n    {{\n        {b}\n        nFaces {c};\n        startFace {s};\n    }}" for n, b, c, s in patches]
    write_list(os.path.join(d, "boundary"), "boundary", "polyBoundaryMesh", bnd)
    with open(os.path.join(o.out, "bc.toml"), "w") as f:
        f.write("[reference]\n")
        f.write(f"direction = [{ct:.17g}, {st:.17g}, 0.0]\nlift = [{-st:.17g}, {ct:.17g}, 0.0]\n")
        if o.lowmach and o.jet:
            # The low-Mach slot burner: rows on xlo (the flags' profile), slot centre y = Ly/2.
            f.write(f"origin = [{-st * 0.5 * ly:.17g}, {ct * 0.5 * ly:.17g}, 0.0]\n")
            f.write('[patch.xlo]\ntype = "inflow"\njet = true\n[patch.xhi]\ntype = "outflow"\n')
        elif o.lowmach:
            ys = ", ".join(repr(float(v)) for v in o.inflow_y.split(","))
            f.write(f'[patch.xlo]\ntype = "inflow"\ny = [{ys}]\n[patch.xhi]\ntype = "outflow"\n')
        elif o.jet:
            f.write(f"origin = [{-st * 0.5 * ly:.17g}, {ct * 0.5 * ly:.17g}, 0.0]\n")
            f.write('[patch.xlo]\ntype = "inflow"\njet = true\n[patch.xhi]\ntype = "outflow"\n')
        else:
            f.write(far_file(o.premix))
        if o.open_y:
            ay = ", ".join(repr(float(v)) for v in o.ambient_y.split(","))
            for n in ("ylo", "yhi"):
                f.write(f'[patch.{n}]\ntype = "open"\nt = {o.ambient_t!r}\ny = [{ay}]\n')
    print(f"{o.out}: {nx * ny} cells, shear {o.shear}, wavy {o.wavy} h, rotate {o.rotate}")


def far_file(premix):
    """The compressible planar flame's patches: fresh Far in, burnt Outflow out."""
    y = "" if premix else "y = [1.0]\n"
    return f'[patch.xlo]\ntype = "far"\n{y}[patch.xhi]\ntype = "outflow"\nburnt = true\n'


def main3d(o):
    """The 3D box: nx x ny x nz hexes, the x-y map of main() on every z plane, y and z
    cyclic. Faces upper-triangular (owner ascending, neighbours +x < +y < +z)."""
    nx, ny, nz, lx, ly, lz = o.nx, o.ny, o.nz, o.lx, o.ly, o.depth
    hx, hy, hz = lx / nx, ly / ny, lz / nz
    ts = math.tan(math.radians(o.shear))
    th = math.radians(o.rotate)
    ct, st = math.cos(th), math.sin(th)

    def xy(i, j):
        x, y = i * hx, j * hy
        xm = x + o.wavy * hx * math.sin(2 * math.pi * y / ly) * math.sin(math.pi * x / lx) ** 2
        ym = y + x * ts
        return ct * xm - st * ym, st * xm + ct * ym

    pid = lambda i, j, k: i + (nx + 1) * (j + (ny + 1) * k)
    pts = []
    for k in range(nz + 1):
        for j in range(ny + 1):
            for i in range(nx + 1):
                px, py = xy(i, j)
                pts.append(f"({px:.17g} {py:.17g} {k * hz:.17g})")
    cid = lambda i, j, k: i + nx * (j + ny * k)

    def q(*p):
        return "4(" + " ".join(str(v) for v in p) + ")"

    # +x, +y, +z oriented faces at the low corner (i, j, k) of their plane.
    fx = lambda i, j, k: q(pid(i, j, k), pid(i, j + 1, k), pid(i, j + 1, k + 1), pid(i, j, k + 1))
    fy = lambda i, j, k: q(pid(i, j, k), pid(i, j, k + 1), pid(i + 1, j, k + 1), pid(i + 1, j, k))
    fz = lambda i, j, k: q(pid(i, j, k), pid(i + 1, j, k), pid(i + 1, j + 1, k), pid(i, j + 1, k))
    rev = lambda f: f[:2] + " ".join(reversed(f[2:-1].split())) + ")"
    faces, own, nei = [], [], []
    for k in range(nz):
        for j in range(ny):
            for i in range(nx):
                c = cid(i, j, k)
                if i + 1 < nx:
                    faces.append(fx(i + 1, j, k))
                    own.append(c)
                    nei.append(cid(i + 1, j, k))
                if j + 1 < ny:
                    faces.append(fy(i, j + 1, k))
                    own.append(c)
                    nei.append(cid(i, j + 1, k))
                if k + 1 < nz:
                    faces.append(fz(i, j, k + 1))
                    own.append(c)
                    nei.append(cid(i, j, k + 1))
    n_int = len(faces)
    patches = []

    def patch(name, body, flist):
        patches.append((name, body, len(flist), len(faces)))
        for fv, c in flist:
            faces.append(fv)
            own.append(c)

    def cyc(name, other, sep):
        return (f"type cyclic; neighbourPatch {other}; transform translational; "
                f"separationVector ({sep[0]:.17g} {sep[1]:.17g} {sep[2]:.17g});")

    patch("xlo", "type patch;", [(rev(fx(0, j, k)), cid(0, j, k)) for k in range(nz) for j in range(ny)])
    patch("xhi", "type patch;", [(fx(nx, j, k), cid(nx - 1, j, k)) for k in range(nz) for j in range(ny)])
    sy = (-st * ly, ct * ly, 0.0)
    patch("ylo", cyc("ylo", "yhi", sy), [(rev(fy(i, 0, k)), cid(i, 0, k)) for k in range(nz) for i in range(nx)])
    patch("yhi", cyc("yhi", "ylo", tuple(-v for v in sy)),
          [(fy(i, ny, k), cid(i, ny - 1, k)) for k in range(nz) for i in range(nx)])
    patch("zlo", cyc("zlo", "zhi", (0.0, 0.0, lz)), [(rev(fz(i, j, 0)), cid(i, j, 0)) for j in range(ny) for i in range(nx)])
    patch("zhi", cyc("zhi", "zlo", (0.0, 0.0, -lz)),
          [(fz(i, j, nz), cid(i, j, nz - 1)) for j in range(ny) for i in range(nx)])
    d = os.path.join(o.out, "constant", "polyMesh")
    os.makedirs(d, exist_ok=True)
    nc = nx * ny * nz
    note = f'\n    note        "nPoints:{len(pts)} nCells:{nc} nFaces:{len(faces)} nInternalFaces:{n_int}";'
    write_list(os.path.join(d, "points"), "points", "vectorField", pts)
    write_list(os.path.join(d, "faces"), "faces", "faceList", faces)
    write_list(os.path.join(d, "owner"), "owner", "labelList", [str(v) for v in own], note)
    write_list(os.path.join(d, "neighbour"), "neighbour", "labelList", [str(v) for v in nei], note)
    bnd = [f"    {n}\n    {{\n        {b}\n        nFaces {c};\n        startFace {s};\n    }}" for n, b, c, s in patches]
    write_list(os.path.join(d, "boundary"), "boundary", "polyBoundaryMesh", bnd)
    with open(os.path.join(o.out, "bc.toml"), "w") as f:
        f.write("[reference]\n")
        f.write(f"direction = [{ct:.17g}, {st:.17g}, 0.0]\nlift = [{-st:.17g}, {ct:.17g}, 0.0]\n")
        f.write(far_file(o.premix))
    print(f"{o.out}: {nc} cells ({nx} x {ny} x {nz}), shear {o.shear}, wavy {o.wavy} h, rotate {o.rotate}")


if __name__ == "__main__":
    main()
