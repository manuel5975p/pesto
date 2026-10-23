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

  * HANDEDNESS: the blade is wound for `dns`'s rotation, +omega about +x (the
    solver accepts only a positive tip Mach). The blade at +y moves toward +z,
    so the leading edge sits at +z and upstream (-x), and the suction side
    faces upstream: thrust on the body is along -x, against the freestream.
    Generators before 2026-10-23 wound it the other way, so every blade ran
    trailing edge first and the "thrust" column carried the sign of drag.

Cross-section (`--section`): `lens` is the symmetric four-point lens (leading
edge, lower crest, trailing edge, upper crest); `naca` is a NACA 4-digit
profile with `--camber`/`--camber-pos` and `--thickness`, sampled at
`--section-points` per side; `dat` reads `--section-file`, Selig (one loop,
TE -> upper -> LE -> lower -> TE) or Lednicer (count line, then upper and
lower LE -> TE) airfoil coordinates. A dat section is splined in arc length
(natural cubic, as XFOIL does), its leading edge is the spline's minimum x,
and it is resampled to `--section-points` per side, cosine spaced in arc
length. Unit chord along the FILE'S OWN x axis (shifted so the LE is at u =
-0.5, both coordinates divided by the LE -> TE x extent, no derotation): the
file's x axis is the chord line beta is measured against. Upper surface =
suction side. An open (blunt) trailing edge is closed at the AVERAGE of its two
end points, each surface sheared onto it linearly in chord fraction (LE fixed),
which removes the TE thickness evenly rather than as a kink in the last panel.
Its thickness comes from the file, so `--thickness`, `--camber`,
`--camber-pos`, `--crest` are rejected with it. Sections are lofted between
spanwise stations and capped at both ends: per blade 2*N*(stations-1) +
2*(N-2) triangles for an N-point ring (N = 4 for the lens).

Planform (`--planform FILE`): a table of r/R, c/R, beta_deg (comma or
whitespace separated; lines that are not three numbers, e.g. a header, are
skipped). c is in TIP RADII -- the UIUC propeller database's convention -- not
diameters. Chord and beta are linear in r/R between rows and held constant
past either end (the root end is buried in the hub). It replaces the
parametric law, so `--pitch`, `--root-chord`, `--tip-chord`,
`--max-pitch-deg` are rejected with it. The tip radius stays 0.5 model units;
set `--hub-radius` to the table's first r/R (where the measured blade starts).
`--export-planform CSV` writes `blade_law` back out in that format (header
`r_R,c_R,beta_deg`, hub surface to tip, `--stations` rows if given, else 40)
for any planform, parametric or tabulated; the STL is then optional.

Every flag combination without `--planform` / `--section dat` builds the
bit-identical STL and `blade_law` it did before those options existed.

`polars.py` and `foam_mrf.py` import this module, and `dns prop` reads its
`--export-planform`: `add_geometry_args` + `blade_law` + `section_unit` ARE the
propeller, so a thrust curve and the STL a solver reads are the same blade by
construction.
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

    def bounds(self):
        """(lo, hi) corners of the vertex bounding box."""
        return ([min(v[i] for v in self.verts) for i in range(3)],
                [max(v[i] for v in self.verts) for i in range(3)])

    def write_stl(self, path, name="propeller", scale=1.0):
        """Binary STL, every coordinate multiplied by `scale` first."""
        verts = [tuple(c * scale for c in v) for v in self.verts]
        with open(path, "wb") as f:
            f.write(name.encode()[:79].ljust(80, b"\0"))
            f.write(struct.pack("<I", len(self.tris)))
            for a, b, c in self.tris:
                pa, pb, pc = verts[a], verts[b], verts[c]
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


def naca4(x, t, m, p):
    """NACA 4-digit camber line and half thickness at chord fraction x in [0,1].

    Closed trailing edge (the -0.1036 coefficient), so the loft has no base.
    Returns (yc, dyc/dx, yt), all in chords.
    """
    yt = 5.0 * t * (0.2969 * math.sqrt(x) - 0.1260 * x - 0.3516 * x * x
                    + 0.2843 * x ** 3 - 0.1036 * x ** 4)
    if m == 0.0:
        return 0.0, 0.0, yt
    if x < p:
        return m / p ** 2 * (2 * p * x - x * x), 2 * m / p ** 2 * (p - x), yt
    return (m / (1 - p) ** 2 * (1 - 2 * p + 2 * p * x - x * x),
            2 * m / (1 - p) ** 2 * (p - x), yt)


def section_unit(a):
    """The section ring for UNIT chord, as (u, w) pairs.

    u runs along the chord, -0.5 at the leading edge to +0.5 at the trailing
    edge; w is normal to it, + on the SUCTION side. Order: leading edge, the
    pressure side (w < 0) to the trailing edge, then the suction side back --
    CCW in (u, w), which is what `section` needs to cap outward.
    """
    if a.section == "dat":
        return list(_dat_section(a.section_file, a.section_points)[0])
    if a.section == "lens":
        um = -0.5 + a.crest
        return [(-0.5, 0.0), (um, -0.5 * a.thickness),
                (0.5, 0.0), (um, 0.5 * a.thickness)]
    n = a.section_points
    xs = [0.5 * (1 - math.cos(math.pi * k / n)) for k in range(n + 1)]
    lower, upper = [], []
    for x in xs:
        yc, dy, yt = naca4(x, a.thickness, a.camber, a.camber_pos)
        th = math.atan(dy)
        upper.append((x - yt * math.sin(th) - 0.5, yc + yt * math.cos(th)))
        lower.append((x + yt * math.sin(th) - 0.5, yc - yt * math.cos(th)))
    # LE and TE are shared by both sides: one point each.
    return [(-0.5, 0.0)] + lower[1:n] + [(0.5, 0.0)] + upper[n - 1:0:-1]


def section(r, chord, beta, unit):
    """The unit section placed at radius r, chord `chord`, pitch `beta`.

    Local frame: s = radial (+y at azimuth 0), c = chord (leading -> trailing
    edge), n = suction-side normal, with s x c = n so (c, n, s) is
    right-handed and a CCW ring in the c-n plane caps outward along +s. For
    +omega about +x the blade at +y moves toward +z, so the leading edge is at
    +z, tilted upstream by beta; beta = 0 puts the chord purely tangential (no
    thrust) and beta -> 90 deg edge-on. The suction side faces upstream (-x).
    """
    sb, cb = math.sin(beta), math.cos(beta)
    cdir = (sb, 0.0, -cb)         # chord: against the rotation, tilted downstream
    ndir = (-cb, 0.0, -sb)        # suction normal = s x c: upstream
    return [(chord * (u * cdir[0] + w * ndir[0]),
             r + chord * (u * cdir[1] + w * ndir[1]),
             chord * (u * cdir[2] + w * ndir[2])) for u, w in unit]


def blade_law(a, r):
    """Planform at radius r (diameter 1, tip radius 0.5): (chord, thick, beta).

    The ONE definition of the blade -- the STL loft and every thrust-curve tool
    evaluate it. Pre: a from `add_geometry_args`, r >= root radius.
    """
    R = 0.5
    table = getattr(a, "planform", None)
    if table is not None:
        chord, beta = _planform_at(_load_planform(table), r / R)
    else:
        r0 = a.hub_radius * R * a.root_overlap
        span = max((r - r0) / (R - r0), 0.0)
        chord = (a.root_chord + (a.tip_chord - a.root_chord) * span) * 2.0 * R
        # Constant-pitch law: beta = atan(P / 2 pi r), steep at the root and
        # shallow at the tip, which is what makes a real prop's twist.
        beta = math.atan2(a.pitch * 2.0 * R, 2.0 * math.pi * max(r, 1e-6))
        beta = min(beta, math.radians(a.max_pitch_deg))
    return chord, section_thickness(a) * chord, beta


def section_thickness(a):
    """Thickness / chord of the section: the flag, or the dat file's own."""
    if a.section == "dat":
        return _dat_section(a.section_file, a.section_points)[1]
    return a.thickness


def ring_sides(ring):
    """A `section_unit` ring split at the trailing edge (its max u) into
    (pressure side LE -> TE, suction side LE -> TE); both hold LE and TE."""
    ite = max(range(len(ring)), key=lambda k: ring[k][0])
    return ring[:ite + 1], [ring[0]] + ring[ite:][::-1]


def planform_rows(a, n):
    """[(r/R, c/R, beta_deg)] of `blade_law` at n stations evenly spaced from
    the hub surface to the tip: the --planform table format, so a file
    written from these rows reads back as the same blade at those stations.
    Pre: a passed check_geometry, n >= 2."""
    out = []
    for i in range(n):
        t = i / (n - 1)
        x = a.hub_radius * (1.0 - t) + t        # exactly hub_radius and 1
        chord, _, beta = blade_law(a, 0.5 * x)  # tip radius 0.5
        out.append((x, chord / 0.5, math.degrees(beta)))
    return out


def write_planform(path, rows):
    """CSV `r_R,c_R,beta_deg`, header first, no comments; shortest repr that
    reads back as the same double."""
    with open(path, "w") as f:
        f.write("r_R,c_R,beta_deg\n")
        f.writelines(f"{x!r},{c!r},{b!r}\n" for x, c, b in rows)


# --- tabulated planform and airfoil files -----------------------------------
# Both are parsed once per (path, parameters) and memoized: blade_law runs per
# station and per element, and the file must not be re-read (or change) under
# a sweep.

_PLANFORMS = {}
_SECTIONS = {}


class GeometryFileError(Exception):
    """A --planform / --section-file that cannot describe a blade."""


def _number_rows(path, ncol):
    """Rows of >= ncol floats (comma or whitespace), other lines skipped."""
    try:
        with open(path, errors="replace") as f:
            lines = f.read().splitlines()
    except OSError as e:
        raise GeometryFileError(f"cannot read {path}: {e}") from None
    rows = []
    for line in lines:
        parts = line.replace(",", " ").split()
        if len(parts) < ncol:
            continue
        try:
            vals = [float(x) for x in parts[:ncol]]
        except ValueError:
            continue
        if all(map(math.isfinite, vals)):
            rows.append(vals)
    return rows


def _load_planform(path):
    """[(r/R, c/R, beta rad)], r/R strictly increasing. Raises on a bad table."""
    got = _PLANFORMS.get(path)
    if got is not None:
        return got
    rows = _number_rows(path, 3)
    if len(rows) < 2:
        raise GeometryFileError(f"{path}: need >= 2 rows of 'r/R c/R beta_deg'")
    for k, (x, c, b) in enumerate(rows):
        if not 0.0 < x <= 1.0 + 1e-9 or c <= 0.0 or not -90.0 < b < 90.0:
            raise GeometryFileError(
                f"{path} row {k + 1}: need 0 < r/R <= 1, c/R > 0, |beta| < 90 "
                f"deg, got {x:g} {c:g} {b:g}")
        if k and x <= rows[k - 1][0]:
            raise GeometryFileError(f"{path}: r/R not strictly increasing at "
                                    f"row {k + 1}")
    table = [(x, c, math.radians(b)) for x, c, b in rows]
    _PLANFORMS[path] = table
    return table


def _planform_at(table, x):
    """(chord in diameters, beta rad) at r/R = x; constant past either end."""
    if x <= table[0][0]:
        _, c, b = table[0]
    elif x >= table[-1][0]:
        _, c, b = table[-1]
    else:
        k = next(i for i in range(1, len(table)) if table[i][0] >= x)
        (x0, c0, b0), (x1, c1, b1) = table[k - 1], table[k]
        f = (x - x0) / (x1 - x0)
        c, b = c0 + f * (c1 - c0), b0 + f * (b1 - b0)
    return 0.5 * c, b               # c/R -> c/D: the tip radius is 0.5 D


def _airfoil_loop(path):
    """The file's points as ONE loop TE -> upper -> LE -> lower -> TE."""
    rows = _number_rows(path, 2)
    pts = [(x, y) for x, y in rows]
    if pts and pts[0][0] > 1.5 and pts[0][1] > 1.5:     # Lednicer count line
        nu, nl = int(round(pts[0][0])), int(round(pts[0][1]))
        body = pts[1:]
        if len(body) != nu + nl:
            raise GeometryFileError(f"{path}: Lednicer header says {nu}+{nl} "
                                    f"points, file has {len(body)}")
        upper, lower = body[:nu], body[nu:]
        if upper[0] == lower[0]:
            lower = lower[1:]       # LE listed on both sides: keep it once
        pts = upper[::-1] + lower
    if len(pts) < 8:
        raise GeometryFileError(f"{path}: need >= 8 coordinate points")
    # Signed area > 0 <=> counter-clockwise <=> upper surface first.
    area = sum(x0 * y1 - x1 * y0 for (x0, y0), (x1, y1)
               in zip(pts, pts[1:] + pts[:1]))
    if area < 0.0:
        pts = pts[::-1]
    # Drop consecutive duplicates (a doubled LE would give a zero-length knot).
    out = [pts[0]]
    for p in pts[1:]:
        if math.hypot(p[0] - out[-1][0], p[1] - out[-1][1]) > 1e-12:
            out.append(p)
    return out


def _spline(s, v):
    """Natural cubic spline through (s, v): returns v(t). s increasing."""
    n = len(s) - 1
    h = [s[i + 1] - s[i] for i in range(n)]
    # Tridiagonal system for the second derivatives m (m_0 = m_n = 0).
    sub, dia, sup, rhs = [0.0] * (n + 1), [1.0] * (n + 1), [0.0] * (n + 1), \
        [0.0] * (n + 1)
    for i in range(1, n):
        sub[i], dia[i], sup[i] = h[i - 1], 2.0 * (h[i - 1] + h[i]), h[i]
        rhs[i] = 6.0 * ((v[i + 1] - v[i]) / h[i] - (v[i] - v[i - 1]) / h[i - 1])
    for i in range(1, n + 1):
        w = sub[i] / dia[i - 1]
        dia[i] -= w * sup[i - 1]
        rhs[i] -= w * rhs[i - 1]
    m = [0.0] * (n + 1)
    m[n] = rhs[n] / dia[n]
    for i in range(n - 1, -1, -1):
        m[i] = (rhs[i] - sup[i] * m[i + 1]) / dia[i]

    def at(t):
        k = min(max(next((i for i in range(n) if s[i + 1] >= t), n - 1), 0),
                n - 1)
        a_, b_ = s[k + 1] - t, t - s[k]
        return ((m[k] * a_ ** 3 + m[k + 1] * b_ ** 3) / (6.0 * h[k])
                + (v[k] / h[k] - m[k] * h[k] / 6.0) * a_
                + (v[k + 1] / h[k] - m[k + 1] * h[k] / 6.0) * b_)
    return at


def _dat_section(path, n):
    """(ring, t/c) of an airfoil file at n points per side; see the module
    docstring for normalization and TE closure. Ring order as section_unit.
    Raises GeometryFileError on a file that is not one airfoil."""
    key = (path, n)
    got = _SECTIONS.get(key)
    if got is not None:
        return got
    pts = _airfoil_loop(path)
    s = [0.0]
    for (x0, y0), (x1, y1) in zip(pts, pts[1:]):
        s.append(s[-1] + math.hypot(x1 - x0, y1 - y0))
    X, Y = _spline(s, [p[0] for p in pts]), _spline(s, [p[1] for p in pts])
    # Leading edge: minimum x of the spline, golden section about the
    # data point of minimum x.
    k = min(range(len(pts)), key=lambda i: pts[i][0])
    if k == 0 or k == len(pts) - 1:
        raise GeometryFileError(f"{path}: minimum x at an end point -- not a "
                                "TE -> LE -> TE loop")
    lo, hi = s[k - 1], s[k + 1]
    g = 0.5 * (math.sqrt(5.0) - 1.0)
    for _ in range(80):
        c1, c2 = hi - g * (hi - lo), lo + g * (hi - lo)
        if X(c1) < X(c2):
            hi = c2
        else:
            lo = c1
    sle = 0.5 * (lo + hi)

    def side(s0, s1):
        """n+1 points LE -> TE, cosine spaced in arc length."""
        return [(X(t), Y(t)) for t in
                (s0 + (s1 - s0) * 0.5 * (1.0 - math.cos(math.pi * j / n))
                 for j in range(n + 1))]
    upper, lower = side(sle, s[0]), side(sle, s[-1])
    xle, yle = upper[0]
    tm = (0.5 * (upper[-1][0] + lower[-1][0]), 0.5 * (upper[-1][1] + lower[-1][1]))

    def close(surf):
        """Shear onto the mean TE point, linear in chord fraction."""
        xt, yt = surf[-1]
        dx, dy, span = tm[0] - xt, tm[1] - yt, xt - xle
        return [(x + (x - xle) / span * dx, y + (x - xle) / span * dy)
                for x, y in surf]
    upper, lower = close(upper), close(lower)
    chord = tm[0] - xle
    if chord <= 0.0:
        raise GeometryFileError(f"{path}: trailing edge not downstream of the "
                                "leading edge")
    norm = lambda p: ((p[0] - xle) / chord - 0.5, p[1] / chord)  # noqa: E731
    up, lw = [norm(p) for p in upper], [norm(p) for p in lower]
    up[0] = lw[0] = (-0.5, yle / chord)
    up[-1] = lw[-1] = (0.5, tm[1] / chord)
    # Thickness, and the check that the two sides never cross: lower w at
    # each upper u, by linear interpolation along the lower side.
    tc = 0.0
    for u, w in up[1:-1]:
        j = next((i for i in range(1, n + 1) if lw[i][0] >= u), None)
        if j is None or lw[j - 1][0] > u:
            raise GeometryFileError(f"{path}: lower surface not single-valued "
                                    "in x")
        (u0, w0), (u1, w1) = lw[j - 1], lw[j]
        wl = w0 + (u - u0) / (u1 - u0) * (w1 - w0) if u1 > u0 else w0
        if w <= wl:
            raise GeometryFileError(f"{path}: upper and lower surfaces cross "
                                    f"at x/c {u + 0.5:.4f} (after TE closure)")
        tc = max(tc, w - wl)
    ring = [lw[0]] + lw[1:n] + [lw[n]] + up[n - 1:0:-1]
    _SECTIONS[key] = (tuple(ring), tc)
    return _SECTIONS[key]


def build(a):
    m = Mesh()
    R = 0.5                       # tip radius; the model's diameter is 1
    rh = a.hub_radius * R

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
    unit = section_unit(a)
    rings, stats = [], []
    for i in range(a.stations):
        r = r0 + (R - r0) * i / (a.stations - 1)
        chord, thick, beta = blade_law(a, r)
        rings.append(section(r, chord, beta, unit))
        stats.append((r, chord, thick, math.degrees(beta)))

    for b in range(a.blades):
        phi = 2.0 * math.pi * b / a.blades
        turned = [[rotate_x(p, phi) for p in ring] for ring in rings]
        for i in range(a.stations - 1):
            m.loft(turned[i], turned[i + 1])
        m.cap(turned[-1], True)    # tip
        m.cap(turned[0], False)    # root, inside the hub
    return m, stats


class _Default(float):
    """A flag's default value, distinguishable from the same value given
    explicitly (argparse does not pass a non-string default through `type`).
    Arithmetic on it yields plain floats, bit-identical to the literal."""


class _DefaultInt(int):
    """`_Default` for an int flag."""


def _given(v):
    """True when the flag was given on the command line."""
    return not isinstance(v, (_Default, _DefaultInt))


def add_geometry_args(p):
    """The propeller's flags, shared by every tool that reads this blade."""
    p.add_argument("--blades", type=int, default=3)
    p.add_argument("--pitch", type=float, default=_Default(0.8),
                   help="constant pitch P/D; sets the twist [0.8]")
    p.add_argument("--root-chord", type=float, default=_Default(0.16),
                   help="chord at the root, in diameters [0.16]")
    p.add_argument("--tip-chord", type=float, default=_Default(0.09),
                   help="chord at the tip, in diameters [0.09]")
    p.add_argument("--planform", default=None, metavar="FILE",
                   help="table r/R, c/R (chord in TIP RADII), beta_deg; "
                        "linear in r/R, replaces --pitch/--root-chord/"
                        "--tip-chord/--max-pitch-deg")
    p.add_argument("--thickness", type=float, default=_Default(0.12),
                   help="thickness / chord. 0.12 is an honest airfoil and is "
                        "NOT resolvable on an affordable grid; raise it to "
                        "trade realism for cells [0.12]")
    p.add_argument("--section", choices=("lens", "naca", "dat"),
                   default="lens",
                   help="cross-section: four-point lens, NACA 4-digit, or "
                        "airfoil coordinates from --section-file [lens]")
    p.add_argument("--section-file", default=None, metavar="FILE",
                   help="dat: Selig or Lednicer airfoil coordinates; upper "
                        "surface = suction side")
    p.add_argument("--crest", type=float, default=_Default(0.35),
                   help="lens: chord fraction of maximum thickness [0.35]")
    p.add_argument("--camber", type=float, default=_Default(0.0),
                   help="naca: max camber / chord, toward the suction side; "
                        "0.04 with --thickness 0.12 is NACA 4412 [0]")
    p.add_argument("--camber-pos", type=float, default=_Default(0.4),
                   help="naca: chord fraction of max camber [0.4]")
    p.add_argument("--section-points", type=int, default=24,
                   help="naca, dat: points per side, cosine spaced [24]")
    p.add_argument("--stations", type=int, default=_DefaultInt(5),
                   help="spanwise sections; triangles/blade = 8*(n-1)+4 [5]")
    p.add_argument("--hub-radius", type=float, default=0.16,
                   help="hub radius as a fraction of the tip radius [0.16]")
    p.add_argument("--hub-length", type=float, default=0.10,
                   help="hub length along the shaft, in diameters [0.10]")
    p.add_argument("--hub-sides", type=int, default=8)
    p.add_argument("--root-overlap", type=float, default=0.6,
                   help="blade root radius as a fraction of the hub radius; "
                        "< 1 so the shells genuinely intersect [0.6]")
    p.add_argument("--max-pitch-deg", type=float, default=_Default(60.0),
                   help="clamp on the root twist [60]")


def check_geometry(a):
    """Reject a geometry no tool can build. Exits with a message."""
    if a.blades < 1 or a.stations < 2 or a.hub_sides < 3:
        sys.exit("propeller: need blades >= 1, stations >= 2, hub-sides >= 3")
    if not 0.0 < a.root_overlap < 1.0:
        sys.exit("propeller: --root-overlap must be in (0,1) -- at 1 the shells "
                 "touch tangentially and the union is not a solid")
    if not 0.0 < a.thickness < 1.0:
        sys.exit("propeller: --thickness must be in (0,1)")
    if a.section == "naca" and (a.section_points < 4 or not 0.0 <= a.camber < 0.2
                                or not 0.05 < a.camber_pos < 0.95):
        sys.exit("propeller: naca needs --section-points >= 4, --camber in "
                 "[0,0.2), --camber-pos in (0.05,0.95)")
    if (a.section == "dat") != (a.section_file is not None):
        sys.exit("propeller: --section dat and --section-file go together")
    if a.section == "dat":
        given = [f for f, v in (("--thickness", a.thickness),
                                ("--camber", a.camber),
                                ("--camber-pos", a.camber_pos),
                                ("--crest", a.crest)) if _given(v)]
        if given:
            sys.exit(f"propeller: {', '.join(given)} cannot be given with "
                     "--section dat: the section's shape and thickness come "
                     "from --section-file")
        if a.section_points < 4:
            sys.exit("propeller: dat needs --section-points >= 4")
    if a.planform is not None:
        given = [f for f, v in (("--pitch", a.pitch),
                                ("--root-chord", a.root_chord),
                                ("--tip-chord", a.tip_chord),
                                ("--max-pitch-deg", a.max_pitch_deg))
                 if _given(v)]
        if given:
            sys.exit(f"propeller: {', '.join(given)} cannot be given with "
                     "--planform: the table replaces the parametric law")
    try:
        if a.planform is not None:
            _load_planform(a.planform)
        if a.section == "dat":
            _dat_section(a.section_file, a.section_points)
    except GeometryFileError as e:
        sys.exit(f"propeller: {e}")


def main():
    p = argparse.ArgumentParser(
        description="Parametric propeller -> binary STL (shaft = +x, "
                    "diameter = 1, wound for +omega about +x).")
    p.add_argument("out", nargs="?", help="output .stl path")
    p.add_argument("--out", dest="out_flag", metavar="STL",
                   help="output .stl path (same as the positional)")
    p.add_argument("--export-planform", metavar="CSV",
                   help="write blade_law as a --planform table r_R,c_R,"
                        "beta_deg, hub surface to tip, at --stations "
                        "stations if given, else 40; no STL unless one is "
                        "named")
    add_geometry_args(p)
    a = p.parse_args()
    if a.out and a.out_flag:
        p.error("give the STL path once (positional or --out)")
    a.out = a.out or a.out_flag
    if not a.out and not a.export_planform:
        p.error("need an output .stl path and/or --export-planform CSV")
    check_geometry(a)

    if a.export_planform:
        n = a.stations if _given(a.stations) else 40
        write_planform(a.export_planform, planform_rows(a, n))
        print(f"wrote {a.export_planform}: {n} stations, r/R "
              f"[{a.hub_radius:g}, 1]")
        if not a.out:
            return
    m, stats = build(a)
    m.write_stl(a.out)

    lo, hi = m.bounds()
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
