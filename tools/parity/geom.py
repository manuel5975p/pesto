"""Parity-suite geometry: the section polygon/STL, the suite rotor STL, and
the replica of dns' measured frontal area (cd/cl reference length).

Content only -- writes input files the way a CAD export would; no solver code.
"""
import argparse
import math
import os
import struct
import sys

import numpy as np

TOOLS = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, TOOLS)
import propeller as prop  # noqa: E402


# --- section (prop1c r = 0.75 R, lens t/c 0.55) ------------------------------

def section_unit(thickness=0.55, crest=0.35):
    """Lens ring, unit chord, about mid-chord: LE, lower crest, TE, upper crest."""
    um = -0.5 + crest
    return [(-0.5, 0.0), (um, -0.5 * thickness), (0.5, 0.0), (um, 0.5 * thickness)]


def section_polygon(aoa_deg, unit):
    """Ring rotated nose-up by aoa (freestream +x, lift +y), chord 1."""
    a = math.radians(aoa_deg)
    ca, sa = math.cos(a), math.sin(a)
    return [(u * ca + w * sa, -u * sa + w * ca) for u, w in unit]


def bbox_centre(poly):
    xs = [p[0] for p in poly]
    ys = [p[1] for p in poly]
    return 0.5 * (max(xs) + min(xs)), 0.5 * (max(ys) + min(ys))


def _write_tris(path, tris, name):
    with open(path, "wb") as f:
        f.write(name.encode()[:79].ljust(80, b"\0"))
        f.write(struct.pack("<I", len(tris)))
        for a, b, c in tris:
            u = [b[i] - a[i] for i in range(3)]
            v = [c[i] - a[i] for i in range(3)]
            n = (u[1] * v[2] - u[2] * v[1], u[2] * v[0] - u[0] * v[2],
                 u[0] * v[1] - u[1] * v[0])
            m = math.sqrt(sum(x * x for x in n)) or 1.0
            f.write(struct.pack("<3f", *(x / m for x in n)))
            for p in (a, b, c):
                f.write(struct.pack("<3f", *p))
            f.write(struct.pack("<H", 0))


def write_section_stl(path, ring, half_span=1.0):
    """Closed prism, ring (CCW in x-y) extruded over z in [-hs, hs]. With hs = 1 c
    the z extent 2 c is dns' D (max bbox extent), so Re_D = 2 Re_c."""
    n = len(ring)
    lo = [(x, y, -half_span) for x, y in ring]
    hi = [(x, y, half_span) for x, y in ring]
    tris = []
    for k in range(n):
        j = (k + 1) % n
        tris += [(lo[k], lo[j], hi[j]), (lo[k], hi[j], hi[k])]
    for k in range(1, n - 1):
        tris += [(hi[0], hi[k], hi[k + 1]), (lo[0], lo[k + 1], lo[k])]
    _write_tris(path, tris, "parity section")


# --- rotor --------------------------------------------------------------------

def rotor_args(spec):
    """propeller.py namespace from a list of flags (the case's `blade`)."""
    p = argparse.ArgumentParser()
    prop.add_geometry_args(p)
    a = p.parse_args(spec)
    prop.check_geometry(a)
    return a


def rotor_mesh(spec):
    a = rotor_args(spec)
    m, stats = prop.build(a)
    return a, m, stats


def swept_diameter(m):
    """Diameter of the circle the model sweeps about +x (dns fits THIS to D)."""
    return 2.0 * max(math.hypot(v[1], v[2]) for v in m.verts)


def write_rotor_stl(path, spec, unit_swept=True):
    """Suite rotor STL. unit_swept: scaled so the swept diameter is exactly 1,
    the same normalisation dns applies (it fits the swept circle to D)."""
    a, m, stats = rotor_mesh(spec)
    sw = swept_diameter(m)
    m.write_stl(path, scale=(1.0 / sw) if unit_swept else 1.0)
    return dict(swept=sw, stats=stats, bounds=m.bounds(), a=a)


def rotor_resolution(spec, h_over_D):
    """(min c/h, min t/h, c/h and t/h at 0.75 R) for a cell size h/D, on the
    swept-diameter-1 model."""
    a, m, _ = rotor_mesh(spec)
    sw = swept_diameter(m)
    r0 = a.hub_radius * 0.5 * a.root_overlap
    rs = np.linspace(a.hub_radius * 0.5, 0.5, 41)
    cs = [prop.blade_law(a, r)[0] / sw for r in rs]
    ts = [prop.blade_law(a, r)[1] / sw for r in rs]
    c75, t75 = (x / sw for x in prop.blade_law(a, 0.375)[:2])
    del r0
    return (min(cs) / h_over_D, min(ts) / h_over_D, c75 / h_over_D, t75 / h_over_D)


# --- dns frontal-area replica (analyze.py of runs/verify/L3_section) ---------

def poly_sdf(poly, x, y):
    """Signed distance to a convex CCW polygon, > 0 outside."""
    n = len(poly)
    d2 = np.full(x.shape, np.inf)
    inside = np.ones(x.shape, bool)
    for k in range(n):
        ax, ay = poly[k]
        bx, by = poly[(k + 1) % n]
        ex, ey = bx - ax, by - ay
        t = np.clip(((x - ax) * ex + (y - ay) * ey) / (ex * ex + ey * ey), 0, 1)
        dx, dy = x - (ax + t * ex), y - (ay + t * ey)
        d2 = np.minimum(d2, dx * dx + dy * dy)
        inside &= (ex * (y - ay) - ey * (x - ax)) >= 0
    return np.where(inside, -np.sqrt(d2), np.sqrt(d2))


def section_frontal_area(poly_chord1, scale, cx, cy, h):
    """dns referenceQComp per unit depth, replicated on the exact polygon:
    sum_j (1 - min_i ax_x) h, x-face apertures from 8 samples per face.
    poly_chord1 is centred on its bbox centre at (cx, cy) and scaled by `scale`."""
    mx, my = bbox_centre(poly_chord1)
    poly = [((px - mx) * scale + cx, (py - my) * scale + cy) for px, py in poly_chord1]
    c = scale
    i0, i1 = int((cx - 0.7 * c) / h) - 2, int((cx + 0.7 * c) / h) + 3
    j0, j1 = int((cy - 0.7 * c) / h) - 2, int((cy + 0.7 * c) / h) + 3
    xf = np.arange(i0, i1 + 1) * h
    a = 0.0
    for j in range(j0, j1 + 1):
        ys = j * h + (np.arange(8) + 0.5) * h / 8
        XX, YY = np.meshgrid(xf, ys, indexing="ij")
        solid = poly_sdf(poly, XX, YY) <= 0.0
        a += solid.mean(axis=1).max() * h
    return a
