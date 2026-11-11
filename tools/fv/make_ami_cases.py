#!/usr/bin/env python3
"""Writes the OpenFOAM v2606 cyclicAMI MESH cases of the FV path (stage 3c:
dns/fvami against AMIInterpolation, tests/test_fvami.cpp) into DEST/<name>/:
system/blockMeshDict plus the minimal controlDict / fvSchemes / fvSolution an
fvMesh needs. No 0/ fields, no flow dictionaries: each case is one mesh
function (`mesh_*`), the flow set-up is added on top of it later.

The interface is two UNMERGED block faces (their own vertices at the same
coordinates) declared cyclicAMI in the M17 spelling:
    type cyclicAMI; neighbourPatch ...; transform noOrdering;
    matchTolerance 0.0001; AMIMethod faceAreaWeightAMI;
    restartUncoveredSourceFace 1;
AMI1 (the owner: lower patch index) is the left / inner / rotor side.

  ami_planar     2D channel [0,2] x [0,1], one cell thick (empty front/back),
                 split at x = 1: left block 10 x 8, right block 10 x 12
                 (faces coincide at y = 0, 1/4, 1/2, 3/4, 1 only). inlet x = 0,
                 outlet x = 2, walls y = 0, 1.
  ami_match      the same with 8 cells on both sides: faces match point for point.
  ami_partial    ami_planar with the right block shifted to y in [0.3, 1.3]: a
                 WEIGHTS-ONLY case (not a flow case): source faces below 0.3
                 have no target (sum(weights) 0), partly covered ones restart;
                 it drives the walk's stall search, the octree and
                 restartUncoveredSourceFace.
  ami_match_int  the same geometry as ONE conformal 20 x 8 block (no AMI).
  ami_cyl        3D, cylinder r 0.5, axis z, height 0.4: inner O-grid (central
                 5 x 5 square + 4 blocks, 20 azimuthal cells, 3 radial, 3 in z,
                 cellZone rotor) inside 4 blocks reaching the box [-1,1]^2
                 (28 azimuthal cells starting 0.13 rad off the box diagonals,
                 4 radial, 4 in z): flat quads of a 20-gon against a 28-gon,
                 non-matching in z too. inlet x = -1, outlet x = 1, walls
                 (y = +-1 and the stator's z ends), rotorEnds (the rotor's z ends).
  ami_mixer2d    2D (empty front/back): rotor disc r 0.5 around a rectangular
                 paddle hole 0.4 x 0.06 (wall patch blade; 4 blocks, 14 + 6 +
                 14 + 6 = 40 azimuthal cells, 6 radial, cellZone rotor) inside
                 4 blocks reaching the box [-1,1]^2 (48 azimuthal cells 0.1 rad
                 off the diagonals, 6 radial), wall patch walls.
  ami_mixer3d    ami_mixer2d extruded to z in [0, 0.3]: rotor 4 cells, stator 5
                 cells in z; rotorEnds / walls at the z ends.

Reference dumps: tools/fv/run_ami_foam.sh (blockMesh, checkMesh, foamAmiDump).

usage: make_ami_cases.py DEST [name ...]
"""
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import make_cases as mc  # noqa: E402

put = mc.put

AMI = ("type cyclicAMI; neighbourPatch {nbr}; transform noOrdering; matchTolerance 0.0001; "
       "AMIMethod faceAreaWeightAMI; restartUncoveredSourceFace 1;")


class BlockMesh:
    """A z-extruded blockMeshDict: 2D vertices, each at z0 and z1."""

    def __init__(self, z0, z1):
        self.z0, self.z1 = z0, z1
        self.xy = []
        self.blocks = []
        self.arcs = []
        self.patches = []  # (name, entries, faces)

    def vert(self, x, y):
        self.xy.append((x, y))
        return len(self.xy) - 1

    def polar(self, r, deg):
        return self.vert(r * math.cos(math.radians(deg)), r * math.sin(math.radians(deg)))

    def block(self, a, b, c, d, n1, n2, nz, zone=""):
        """Bottom face a b c d counter-clockwise seen from +z."""
        self.blocks.append((a, b, c, d, n1, n2, nz, zone))

    def arc(self, a, b, r, deg_a, deg_b):
        m = math.radians(0.5 * (deg_a + deg_b))
        self.arcs.append((a, b, r * math.cos(m), r * math.sin(m)))

    def patch(self, name, entries):
        self.patches.append((name, entries, []))
        return self.patches[-1][2]

    # Faces are kept as 2D vertex tuples: the top layer's offset is known in text().
    def side(self, a, b):
        return (a, b)

    def ends(self, a, b, c, d):
        return [(a, d, c, b, 0), (a, b, c, d, 1)]

    def face(self, f, n):
        if len(f) == 2:
            return f"({f[0]} {f[1]} {f[1] + n} {f[0] + n})"
        o = n * f[4]
        return f"({f[0] + o} {f[1] + o} {f[2] + o} {f[3] + o})"

    def text(self):
        n = len(self.xy)
        o = ["scale 1;", "vertices", "("]
        for z in (self.z0, self.z1):
            o += [f"    ({x!r} {y!r} {z!r})" for x, y in self.xy]
        o += [");", "blocks", "("]
        for a, b, c, d, n1, n2, nz, zone in self.blocks:
            o.append(f"    hex ({a} {b} {c} {d} {a + n} {b + n} {c + n} {d + n}) {zone} "
                     f"({n1} {n2} {nz}) simpleGrading (1 1 1)")
        o += [");", "edges", "("]
        for a, b, mx, my in self.arcs:
            o.append(f"    arc {a} {b} ({mx!r} {my!r} {self.z0!r})")
            o.append(f"    arc {a + n} {b + n} ({mx!r} {my!r} {self.z1!r})")
        o += [");", "boundary", "("]
        for name, entries, faces in self.patches:
            o.append(f"    {name} {{ {entries} faces ({' '.join(self.face(f, n) for f in faces)}); }}")
        o += [");", ""]
        return "\n".join(o)


def support(root):
    """What createTime / createMesh need; the flow dictionaries replace these."""
    put(root, "system/controlDict", "dictionary", "controlDict", """application pimpleFoam;
startFrom startTime;
startTime 0;
stopAt endTime;
endTime 1;
deltaT 1;
writeControl timeStep;
writeInterval 1;
writeFormat ascii;
writePrecision 16;
timePrecision 8;
runTimeModifiable false;
""")
    put(root, "system/fvSchemes", "dictionary", "fvSchemes", """ddtSchemes { default Euler; }
gradSchemes { default Gauss linear; }
divSchemes { default none; }
laplacianSchemes { default Gauss linear corrected; }
interpolationSchemes { default linear; }
snGradSchemes { default corrected; }
""")
    put(root, "system/fvSolution", "dictionary", "fvSolution", "solvers\n{\n}\n")


def mesh_planar(root, ny2, name1="AMI1", name2="AMI2", y_off=0.0):
    b = BlockMesh(0.0, 0.1)
    l = [b.vert(0, 0), b.vert(1, 0), b.vert(1, 1), b.vert(0, 1)]
    r = [b.vert(1, y_off), b.vert(2, y_off), b.vert(2, 1 + y_off), b.vert(1, 1 + y_off)]
    b.block(*l, 10, 8, 1)
    b.block(*r, 10, ny2, 1)
    b.patch("inlet", "type patch;").append(b.side(l[3], l[0]))
    b.patch("outlet", "type patch;").append(b.side(r[1], r[2]))
    b.patch("walls", "type wall;").extend(
        [b.side(l[0], l[1]), b.side(l[2], l[3]), b.side(r[0], r[1]), b.side(r[2], r[3])])
    b.patch(name1, AMI.format(nbr=name2)).append(b.side(l[1], l[2]))
    b.patch(name2, AMI.format(nbr=name1)).append(b.side(r[3], r[0]))
    b.patch("frontAndBack", "type empty;").extend(b.ends(*l) + b.ends(*r))
    put(root, "system/blockMeshDict", "dictionary", "blockMeshDict", b.text())
    support(root)


def mesh_ami_planar(root):
    mesh_planar(root, 12)


def mesh_ami_match(root):
    mesh_planar(root, 8)


def mesh_ami_partial(root):
    mesh_planar(root, 12, y_off=0.3)


def mesh_ami_match_int(root):
    b = BlockMesh(0.0, 0.1)
    v = [b.vert(0, 0), b.vert(2, 0), b.vert(2, 1), b.vert(0, 1)]
    b.block(*v, 20, 8, 1)
    b.patch("inlet", "type patch;").append(b.side(v[3], v[0]))
    b.patch("outlet", "type patch;").append(b.side(v[1], v[2]))
    b.patch("walls", "type wall;").extend([b.side(v[0], v[1]), b.side(v[2], v[3])])
    b.patch("frontAndBack", "type empty;").extend(b.ends(*v))
    put(root, "system/blockMeshDict", "dictionary", "blockMeshDict", b.text())
    support(root)


R_AMI = 0.5


def stator(b, n_az, n_r, nz, offset_deg):
    """4 blocks from the circle (its own vertices, offset_deg off the box
    diagonals) to the box [-1,1]^2. Returns (circle faces, box side faces by
    block: y = -1, x = 1, y = 1, x = -1, z-end faces)."""
    ang = [225 + 90 * i + offset_deg for i in range(4)]
    s = [b.polar(R_AMI, a) for a in ang]
    box = [b.vert(-1, -1), b.vert(1, -1), b.vert(1, 1), b.vert(-1, 1)]
    circle, sides, ends = [], [], []
    for i in range(4):
        j = (i + 1) % 4
        b.block(s[j], s[i], box[i], box[j], n_az, n_r, nz)
        b.arc(s[i], s[j], R_AMI, ang[i], ang[i] + 90)
        circle.append(b.side(s[i], s[j]))
        sides.append(b.side(box[i], box[j]))
        ends += b.ends(s[j], s[i], box[i], box[j])
    return circle, sides, ends


def mesh_ami_cyl(root):
    b = BlockMesh(0.0, 0.4)
    a = 0.2
    ang = [225, 315, 45, 135]
    c = [b.vert(-a, -a), b.vert(a, -a), b.vert(a, a), b.vert(-a, a)]
    r = [b.polar(R_AMI, t) for t in ang]
    b.block(*c, 5, 5, 3, "rotor")
    rotor_circle, rotor_ends = [], b.ends(*c)
    for i in range(4):
        j = (i + 1) % 4
        b.block(c[j], c[i], r[i], r[j], 5, 3, 3, "rotor")
        b.arc(r[i], r[j], R_AMI, ang[i], ang[i] + 90)
        rotor_circle.append(b.side(r[i], r[j]))
        rotor_ends += b.ends(c[j], c[i], r[i], r[j])
    circle, sides, ends = stator(b, 7, 4, 4, math.degrees(0.13))
    b.patch("inlet", "type patch;").append(sides[3])
    b.patch("outlet", "type patch;").append(sides[1])
    b.patch("walls", "type wall;").extend([sides[0], sides[2]] + ends)
    b.patch("rotorEnds", "type wall;").extend(rotor_ends)
    b.patch("AMI1", AMI.format(nbr="AMI2")).extend(rotor_circle)
    b.patch("AMI2", AMI.format(nbr="AMI1")).extend(circle)
    put(root, "system/blockMeshDict", "dictionary", "blockMeshDict", b.text())
    support(root)


def mesh_mixer(root, z1, nz_rotor, nz_stator, three_d):
    b = BlockMesh(0.0, z1)
    hx, hy = 0.2, 0.03
    ang = [215, 325, 395, 505]  # 110, 70, 110, 70 degrees
    n_az = [14, 6, 14, 6]
    h = [b.vert(-hx, -hy), b.vert(hx, -hy), b.vert(hx, hy), b.vert(-hx, hy)]
    r = [b.polar(R_AMI, t) for t in ang]
    blade, rotor_circle, rotor_ends = [], [], []
    for i in range(4):
        j = (i + 1) % 4
        b.block(h[j], h[i], r[i], r[j], n_az[i], 6, nz_rotor, "rotor")
        b.arc(r[i], r[j], R_AMI, ang[i], ang[i] + (110 if i % 2 == 0 else 70))
        blade.append(b.side(h[i], h[j]))
        rotor_circle.append(b.side(r[i], r[j]))
        rotor_ends += b.ends(h[j], h[i], r[i], r[j])
    circle, sides, ends = stator(b, 12, 6, nz_stator, math.degrees(0.1))
    b.patch("walls", "type wall;").extend(sides + (ends if three_d else []))
    b.patch("blade", "type wall;").extend(blade)
    if three_d:
        b.patch("rotorEnds", "type wall;").extend(rotor_ends)
    b.patch("AMI1", AMI.format(nbr="AMI2")).extend(rotor_circle)
    b.patch("AMI2", AMI.format(nbr="AMI1")).extend(circle)
    if not three_d:
        b.patch("frontAndBack", "type empty;").extend(rotor_ends + ends)
    put(root, "system/blockMeshDict", "dictionary", "blockMeshDict", b.text())
    support(root)


def mesh_ami_mixer2d(root):
    mesh_mixer(root, 0.1, 1, 1, False)


def mesh_ami_mixer3d(root):
    mesh_mixer(root, 0.3, 4, 5, True)


CASES = {"ami_planar": mesh_ami_planar, "ami_match": mesh_ami_match, "ami_partial": mesh_ami_partial,
         "ami_match_int": mesh_ami_match_int,
         "ami_cyl": mesh_ami_cyl, "ami_mixer2d": mesh_ami_mixer2d, "ami_mixer3d": mesh_ami_mixer3d}

if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    dest = sys.argv[1]
    names = sys.argv[2:] or list(CASES)
    for n in names:
        if n not in CASES:
            sys.exit(f"unknown case {n} (one of {', '.join(CASES)})")
        CASES[n](os.path.join(dest, n))
        print("wrote", os.path.join(dest, n))
