#!/usr/bin/env python3
"""dns `--emit mesh` -> an OpenFOAM polyMesh of the SAME cells.

The parity cases (sph1e4, sph100; masterplan_flowrans section 8) compare dns
and OpenFOAM on identical meshes, never regenerated: this reads
<out>_mesh.bin (dns/meshio.hpp, docs/viewer-format.md) and writes
constant/polyMesh/{points,faces,owner,neighbour,boundary} directly.

Revolved WEDGE (the only layout in this plan): the real cells i = 1..nTheta,
j = 1..ny-2 of plane k = 1. The degenerate axis faces are dropped and the
coincident axis points merged, so the axis-adjacent cells become prisms
(their j-faces triangles). Patches: sphere (wall), farfield (patch), wedge0 /
wedge1 (wedge, node planes 1 and 2: the wedge is symmetric about z = 0 with
the axis along x, OpenFOAM's convention). Lengths are divided by the dns
diameter D = pi/4 (box units) so the OpenFOAM case has D = 1.

Also returns the per-wall-face first-cell distance d1 (in D) for the SST
omega wall list, and the dns (i, j) of every OpenFOAM cell.
"""
import math
import os
import struct
import sys

import numpy as np

D_BOX = math.pi / 4.0


def read_mesh(path):
    """The fields of a DNSMESH1 file as a dict."""
    with open(path, "rb") as f:
        b = f.read()
    if b[:8] != b"DNSMESH1":
        raise ValueError(f"{path}: not a DNSMESH1 file")
    off = 8
    def u32(n):
        nonlocal off
        v = struct.unpack_from(f"<{n}I", b, off)
        off += 4 * n
        return v
    def f64(n):
        nonlocal off
        v = struct.unpack_from(f"<{n}d", b, off)
        off += 8 * n
        return v
    (version,) = u32(1)
    nx, ny, nz = u32(3)
    wrap = u32(3)
    (flags,) = u32(1)
    (ntheta,) = u32(1)
    (dphi,) = f64(1)
    centre = f64(3)
    (mirror_z0,) = f64(1)
    n = nx * ny * nz
    x = np.frombuffer(b, "<f8", n, off); off += 8 * n
    y = np.frombuffer(b, "<f8", n, off); off += 8 * n
    z = np.frombuffer(b, "<f8", n, off); off += 8 * n
    d1 = np.frombuffer(b, "<f8", nx * nz, off); off += 8 * nx * nz
    shape = (nz, ny, nx)
    return dict(version=version, nx=nx, ny=ny, nz=nz, wrap=wrap, flags=flags,
                ntheta=ntheta, dphi=dphi, centre=centre, mirror_z0=mirror_z0,
                x=x.reshape(shape), y=y.reshape(shape), z=z.reshape(shape),
                d1=d1.reshape((nz, nx)))


def node(m, i, j, k):
    """CurvMesh::node for the stored range (wedge real cells never need
    more than i <= nx-1, j <= ny-1, k <= 2)."""
    return np.array([m["x"][k, j, i], m["y"][k, j, i], m["z"][k, j, i]])


def newell(pts):
    n = np.zeros(3)
    for a in range(len(pts)):
        p, q = pts[a], pts[(a + 1) % len(pts)]
        n += np.cross(p, q)
    return 0.5 * n


def build(m):
    """points, faces, owner, neighbour, patches, cellij for the wedge."""
    if not (m["flags"] & 4):
        raise ValueError("only the revolved wedge layout is supported")
    nT, ny = m["ntheta"], m["ny"]
    nf = ny - 2
    centre = np.array(m["centre"])
    # Points: (i, j, kk) for i in [1, nT+1], j in [1, ny-1], kk 0/1 (node
    # planes 1/2); the two axis columns merged across kk.
    pid = {}
    pts = []
    for j in range(1, ny):
        for i in range(1, nT + 2):
            axis = i in (1, nT + 1)
            for kk in (0, 1):
                if axis and kk == 1:
                    pid[(i, j, 1)] = pid[(i, j, 0)]
                    continue
                pid[(i, j, kk)] = len(pts)
                pts.append((node(m, i, j, kk + 1) - centre) / D_BOX)
    pts = np.array(pts)
    cell = lambda i, j: (i - 1) + nT * (j - 1)
    ncell = nT * nf
    cc = np.zeros((ncell, 3))
    for j in range(1, ny - 1):
        for i in range(1, nT + 1):
            corners = {pid[(i + di, j + dj, kk)] for di in (0, 1)
                       for dj in (0, 1) for kk in (0, 1)}
            cc[cell(i, j)] = pts[list(corners)].mean(axis=0)

    def clean(f):
        out = []
        for p in f:
            if not out or out[-1] != p:
                out.append(p)
        if len(out) > 1 and out[0] == out[-1]:
            out.pop()
        return out

    def oriented(f, towards):
        """f with its normal pointing along `towards`."""
        f = clean(f)
        if np.dot(newell(pts[f]), towards) < 0:
            f = f[::-1]
        return f

    faces, owner, neigh = [], [], []
    # Internal faces, upper-triangular: owner ascending, then neighbour.
    for j in range(1, ny - 1):
        for i in range(1, nT + 1):
            c = cell(i, j)
            if i < nT:  # i-face at node column i+1
                f = [pid[(i + 1, j, 0)], pid[(i + 1, j + 1, 0)],
                     pid[(i + 1, j + 1, 1)], pid[(i + 1, j, 1)]]
                faces.append(oriented(f, cc[cell(i + 1, j)] - cc[c]))
                owner.append(c)
                neigh.append(cell(i + 1, j))
            if j < ny - 2:  # j-face at node row j+1
                f = [pid[(i, j + 1, 0)], pid[(i + 1, j + 1, 0)],
                     pid[(i + 1, j + 1, 1)], pid[(i, j + 1, 1)]]
                faces.append(oriented(f, cc[cell(i, j + 1)] - cc[c]))
                owner.append(c)
                neigh.append(cell(i, j + 1))
    patches = []

    def patch(name, typ, items):
        start = len(faces)
        for c, f in items:
            fc = pts[clean(f)].mean(axis=0)
            faces.append(oriented(f, fc - cc[c]))
            owner.append(c)
        patches.append((name, typ, start, len(faces) - start))

    patch("sphere", "wall",
          [(cell(i, 1), [pid[(i, 1, 0)], pid[(i + 1, 1, 0)],
                         pid[(i + 1, 1, 1)], pid[(i, 1, 1)]])
           for i in range(1, nT + 1)])
    patch("farfield", "patch",
          [(cell(i, ny - 2), [pid[(i, ny - 1, 0)], pid[(i + 1, ny - 1, 0)],
                              pid[(i + 1, ny - 1, 1)], pid[(i, ny - 1, 1)]])
           for i in range(1, nT + 1)])
    for kk, name in ((0, "wedge0"), (1, "wedge1")):
        patch(name, "wedge",
              [(cell(i, j), [pid[(i, j, kk)], pid[(i + 1, j, kk)],
                             pid[(i + 1, j + 1, kk)], pid[(i, j + 1, kk)]])
               for j in range(1, ny - 1) for i in range(1, nT + 1)])
    cellij = [(i, j) for j in range(1, ny - 1) for i in range(1, nT + 1)]
    d1 = m["d1"][1, 1:nT + 1] / D_BOX
    return pts, faces, owner, neigh, patches, cellij, d1


def foam_header(cls, obj, note=""):
    return ("FoamFile\n{\n    version 2.0;\n    format ascii;\n"
            f"    class {cls};\n    location \"constant/polyMesh\";\n"
            f"    object {obj};\n" + (f"    note \"{note}\";\n" if note else "")
            + "}\n\n")


def write_polymesh(case, m):
    pts, faces, owner, neigh, patches, cellij, d1 = build(m)
    d = os.path.join(case, "constant", "polyMesh")
    os.makedirs(d, exist_ok=True)
    nc = len(cellij)
    note = (f"nPoints:{len(pts)} nCells:{nc} nFaces:{len(faces)} "
            f"nInternalFaces:{len(neigh)}")
    with open(os.path.join(d, "points"), "w") as f:
        f.write(foam_header("vectorField", "points"))
        f.write(f"{len(pts)}\n(\n")
        for p in pts:
            f.write(f"({p[0]:.17g} {p[1]:.17g} {p[2]:.17g})\n")
        f.write(")\n")
    with open(os.path.join(d, "faces"), "w") as f:
        f.write(foam_header("faceList", "faces"))
        f.write(f"{len(faces)}\n(\n")
        for fc in faces:
            f.write(f"{len(fc)}({' '.join(map(str, fc))})\n")
        f.write(")\n")
    for name, arr in (("owner", owner), ("neighbour", neigh)):
        with open(os.path.join(d, name), "w") as f:
            f.write(foam_header("labelList", name, note))
            f.write(f"{len(arr)}\n(\n")
            f.write("\n".join(map(str, arr)))
            f.write("\n)\n")
    with open(os.path.join(d, "boundary"), "w") as f:
        f.write(foam_header("polyBoundaryMesh", "boundary"))
        f.write(f"{len(patches)}\n(\n")
        for name, typ, start, n in patches:
            extra = "        inGroups List<word> 1(wall);\n" if typ == "wall" else ""
            f.write(f"    {name}\n    {{\n        type {typ};\n{extra}"
                    f"        nFaces {n};\n        startFace {start};\n    }}\n")
        f.write(")\n")
    return cellij, d1


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit("usage: dnsmesh.py <out>_mesh.bin CASEDIR")
    m = read_mesh(sys.argv[1])
    cellij, d1 = write_polymesh(sys.argv[2], m)
    print(f"cells {len(cellij)}  wall faces {len(d1)}  d1 {d1.min():.3e}.."
          f"{d1.max():.3e} D")
