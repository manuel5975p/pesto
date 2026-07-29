#!/usr/bin/env python3
"""Parametric propeller -> binary STL, in the tens of triangles per blade.

A stand-in for a real CAD prop while the rotor path is being exercised: coarse
on purpose, so the mesh is legible, the BVH is trivial and the winding-number
sign test costs nothing. It is a CONTENT tool, not solver code -- it produces
an input file exactly as a CAD export would, and the solver reads it through
the same `loadStl` path.

Conventions that matter to `dns`:

  * The shaft is +x, matching `--rotor-axis x` (an axial-flow rotor with the
    freestream along the shaft). Blades sweep the y-z plane.
  * The SHAFT IS THE ORIGIN. That is the contract for a rotor mesh: `dns`
    anchors a spinning body on the mesh's own origin, not on its bounding-box
    centre, because a propeller's box is not centred on its axis (three blades
    put it 10% of the tip radius off) and anchoring there would spin the model
    about a point that is not its axis.
  * The model is normalized so the SWEPT diameter is exactly 1, and `dns`
    auto-fits a rotor by that same swept circle rather than by the bounding
    box. So no `--scale` is needed and the nominal tip radius D/2 that
    `--rotor-mach` is quoted on is the real one.
  * Hub and blades are separate CLOSED shells that OVERLAP at the root. Every
    edge still joins exactly two triangles, so `MeshSdf::watertight()` reports
    true and no warning fires -- but the closest-triangle sign test is wrong in
    the overlap (a point inside a blade whose nearest triangle is on the hub
    reads as fluid, notching the root). Pass `--robust` to `dns`: the
    generalized winding number sums over both shells and is >= 1 inside either.
    At this triangle count it is free.

Cross-section is a symmetric lens: four points (leading edge, upper crest,
trailing edge, lower crest), lofted between spanwise stations and capped at
both ends. Per blade that is 8*(stations-1) + 4 triangles.
"""

import argparse
import math
import struct
import sys


def rotate_x(p, phi):
    """Rotate about the shaft. Right-handed, so +phi is the sense of +omega."""
    c, s = math.cos(phi), math.sin(phi)
    return (p[0], p[1] * c - p[2] * s, p[1] * s + p[2] * c)


class Mesh:
    """Indexed triangle soup with a shared vertex table.

    Vertices are interned so that a shared corner is ONE entry: `loadStl` merges
    bit-identical vertices, and a vertex computed twice by different expressions
    might not round to the same float32. Interning here makes the topology
    watertight by construction rather than by luck.
    """

    def __init__(self):
        self.verts = []
        self.index = {}
        self.tris = []

    def vert(self, p):
        key = (round(p[0], 12), round(p[1], 12), round(p[2], 12))
        i = self.index.get(key)
        if i is None:
            i = len(self.verts)
            self.index[key] = i
            self.verts.append(p)
        return i

    def tri(self, a, b, c):
        self.tris.append((self.vert(a), self.vert(b), self.vert(c)))

    def quad(self, a, b, c, d):
        """CCW quad -> two triangles with the same outward normal."""
        self.tri(a, b, c)
        self.tri(a, c, d)

    def loft(self, inner, outer):
        """Side wall between two equally sized rings.

        `inner`/`outer` run counter-clockwise seen from the OUTER end, so
        (A_k, A_k+1, B_k+1, B_k) faces away from the axis.
        """
        n = len(inner)
        for k in range(n):
            j = (k + 1) % n
            self.quad(inner[k], inner[j], outer[j], outer[k])

    def cap(self, ring, outward):
        """Fan-close one end. `outward` true at the end the ring faces."""
        for k in range(1, len(ring) - 1):
            if outward:
                self.tri(ring[0], ring[k], ring[k + 1])
            else:
                self.tri(ring[0], ring[k + 1], ring[k])

    def write_stl(self, path, name="propeller"):
        with open(path, "wb") as f:
            f.write(name.encode()[:79].ljust(80, b"\0"))
            f.write(struct.pack("<I", len(self.tris)))
            for a, b, c in self.tris:
                pa, pb, pc = self.verts[a], self.verts[b], self.verts[c]
                u = [pb[i] - pa[i] for i in range(3)]
                v = [pc[i] - pa[i] for i in range(3)]
                nx = u[1] * v[2] - u[2] * v[1]
                ny = u[2] * v[0] - u[0] * v[2]
                nz = u[0] * v[1] - u[1] * v[0]
                m = math.sqrt(nx * nx + ny * ny + nz * nz) or 1.0
                f.write(struct.pack("<3f", nx / m, ny / m, nz / m))
                for p in (pa, pb, pc):
                    f.write(struct.pack("<3f", *p))
                f.write(struct.pack("<H", 0))


def section(r, chord, thick, beta, crest):
    """Four section points at radius r, ordered CCW seen from outboard.

    Local frame: s = radial (+y at azimuth 0), c = chord, n = thickness, with
    s x c = n so (c, n, s) is right-handed and a CCW ring in the c-n plane
    caps outward along +s. beta is the pitch angle about the span, so beta = 0
    puts the chord purely tangential (no thrust) and beta -> 90 deg edge-on.
    """
    sb, cb = math.sin(beta), math.cos(beta)
    cdir = (sb, 0.0, cb)          # chord: tangential, tilted into the axis
    ndir = (cb, 0.0, -sb)         # thickness: axial, tilted out of it
    half = 0.5 * chord
    umax = -half + crest * chord  # chordwise station of maximum thickness

    def at(u, w):
        return (u * cdir[0] + w * ndir[0],
                r + u * cdir[1] + w * ndir[1],
                u * cdir[2] + w * ndir[2])

    return [at(-half, 0.0),            # leading edge
            at(umax, 0.5 * thick),     # upper crest
            at(half, 0.0),             # trailing edge
            at(umax, -0.5 * thick)]    # lower crest


def build(a):
    m = Mesh()
    R = 0.5                       # tip radius; the model's diameter is 1
    rh = a.hub_radius * R
    pitch = a.pitch * 2.0 * R     # constant pitch P, quoted as P/D

    # --- hub: a regular prism along the shaft ------------------------------
    nh = a.hub_sides
    half = 0.5 * a.hub_length * 2.0 * R
    back, front = [], []
    for k in range(nh):
        th = 2.0 * math.pi * k / nh
        y, z = rh * math.cos(th), rh * math.sin(th)
        back.append((-half, y, z))     # ring seen CCW from +x
        front.append((half, y, z))
    m.loft(back, front)
    m.cap(front, True)
    m.cap(back, False)

    # --- blades -------------------------------------------------------------
    # The root is buried INSIDE the hub (overlapping shells, see the module
    # docstring): it starts a little inboard of the hub surface so the two
    # solids genuinely intersect rather than touching tangentially, which is
    # what makes the winding-number union well defined.
    r0 = rh * a.root_overlap
    rings, stats = [], []
    for i in range(a.stations):
        f = i / (a.stations - 1)
        r = r0 + (R - r0) * f
        span = max((r - r0) / (R - r0), 0.0)
        chord = (a.root_chord + (a.tip_chord - a.root_chord) * span) * 2.0 * R
        thick = a.thickness * chord
        # Constant-pitch law: beta = atan(P / 2 pi r), steep at the root and
        # shallow at the tip, which is what makes a real prop's twist.
        beta = math.atan2(pitch, 2.0 * math.pi * max(r, 1e-6))
        beta = min(beta, math.radians(a.max_pitch_deg))
        rings.append(section(r, chord, thick, beta, a.crest))
        stats.append((r, chord, thick, math.degrees(beta)))

    for b in range(a.blades):
        phi = 2.0 * math.pi * b / a.blades
        turned = [[rotate_x(p, phi) for p in ring] for ring in rings]
        for i in range(a.stations - 1):
            m.loft(turned[i], turned[i + 1])
        m.cap(turned[-1], True)    # tip
        m.cap(turned[0], False)    # root, inside the hub
    return m, stats


def main():
    p = argparse.ArgumentParser(
        description="Parametric propeller -> binary STL (shaft = +x, "
                    "diameter = 1).")
    p.add_argument("out", help="output .stl path")
    p.add_argument("--blades", type=int, default=3)
    p.add_argument("--pitch", type=float, default=0.8,
                   help="constant pitch P/D; sets the twist [0.8]")
    p.add_argument("--root-chord", type=float, default=0.16,
                   help="chord at the root, in diameters [0.16]")
    p.add_argument("--tip-chord", type=float, default=0.09,
                   help="chord at the tip, in diameters [0.09]")
    p.add_argument("--thickness", type=float, default=0.12,
                   help="thickness / chord. 0.12 is an honest airfoil and is "
                        "NOT resolvable on an affordable grid; raise it to "
                        "trade realism for cells [0.12]")
    p.add_argument("--crest", type=float, default=0.35,
                   help="chord fraction of maximum thickness [0.35]")
    p.add_argument("--stations", type=int, default=5,
                   help="spanwise sections; triangles/blade = 8*(n-1)+4 [5]")
    p.add_argument("--hub-radius", type=float, default=0.16,
                   help="hub radius as a fraction of the tip radius [0.16]")
    p.add_argument("--hub-length", type=float, default=0.10,
                   help="hub length along the shaft, in diameters [0.10]")
    p.add_argument("--hub-sides", type=int, default=8)
    p.add_argument("--root-overlap", type=float, default=0.6,
                   help="blade root radius as a fraction of the hub radius; "
                        "< 1 so the shells genuinely intersect [0.6]")
    p.add_argument("--max-pitch-deg", type=float, default=60.0,
                   help="clamp on the root twist [60]")
    a = p.parse_args()

    if a.blades < 1 or a.stations < 2 or a.hub_sides < 3:
        sys.exit("propeller: need blades >= 1, stations >= 2, hub-sides >= 3")
    if not 0.0 < a.root_overlap < 1.0:
        sys.exit("propeller: --root-overlap must be in (0,1) -- at 1 the shells "
                 "touch tangentially and the union is not a solid")

    m, stats = build(a)
    m.write_stl(a.out)

    lo = [min(v[i] for v in m.verts) for i in range(3)]
    hi = [max(v[i] for v in m.verts) for i in range(3)]
    ext = max(hi[i] - lo[i] for i in range(3))
    # The swept diameter: twice the largest distance from the shaft to any
    # vertex. This -- not the bounding box -- is what --rotor-mach is quoted on.
    swept = 2.0 * max(math.hypot(v[1], v[2]) for v in m.verts)
    tmin = min(s[2] for s in stats)
    off = math.hypot(0.5 * (lo[1] + hi[1]), 0.5 * (lo[2] + hi[2]))

    print(f"wrote {a.out}: {len(m.tris)} triangles, {len(m.verts)} vertices "
          f"({(len(m.tris) - _hub_tris(a)) // a.blades} per blade)")
    print(f"  bbox    x [{lo[0]:+.4f},{hi[0]:+.4f}]  "
          f"y [{lo[1]:+.4f},{hi[1]:+.4f}]  z [{lo[2]:+.4f},{hi[2]:+.4f}]")
    print(f"  swept diameter {swept:.4f}, largest static extent {ext:.4f} "
          f"(ratio {ext / swept:.3f})")
    print(f"  bbox centre sits {off / (0.5 * swept):.3f} tip radii off the "
          f"shaft -- which is")
    print("  why dns anchors a rotor on the ORIGIN and fits the SWEPT circle.")
    print("  station    r/R    chord/D   thick/D   twist")
    for r, c, t, b in stats:
        print(f"    {r / 0.5:8.3f} {c:9.4f} {t:9.4f} {b:7.1f} deg")
    print()
    print(f"  Thinnest section {tmin:.4f} D. The ghost wall needs a mirror band")
    print("  >= 2 cells per side, so the blade must span >= 4 cells: that is")
    print(f"  {4.0 / tmin:.0f} cells across D, or --blade-thicken to make up "
          "the difference.")
    print()
    print("  Pass --robust: hub and blades are overlapping closed shells, so the")
    print("  closest-triangle sign test notches the root and the winding number")
    print("  does not.")


def _hub_tris(a):
    return a.hub_sides * 2 + 2 * (a.hub_sides - 2)


if __name__ == "__main__":
    main()
