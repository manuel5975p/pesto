#!/usr/bin/env python3
"""OpenFOAM steady-RANS MRF propeller: case generator + J-sweep runner.

The third leg of the thrust-curve tripod (`dns prop --method bemt|liftline`): the same
blade, from `propeller.py`, put in a box and solved with simpleFoam in a
multiple-reference-frame zone. It is a CONTENT tool -- it writes an ordinary
OpenFOAM case and drives the stock v2606 utilities; nothing here is solver code.
Every dictionary is a plain file under `foam_templates/` (laid out as the case
is, `$name` = string.Template substitution); this driver only fills them in.

Case, in metres (the STL is the diameter-1 model scaled by --diameter):

  * Box: x in [-3D, +7D] (inlet at x-min, outlet at x-max), y and z in
    [-3D, +3D] ("sides"). blockMesh at --cells-per-D, then snappyHexMesh.
  * Rotor zone: a searchableCylinder of radius 0.6 D about the shaft, axial
    extent the blade's bounding box +/- 0.1 D, given to snappy as an INTERNAL
    faceZone + cellZone "rotor" -- so the zone boundary is snapped to a true
    cylinder, not stair-cased. constant/MRFProperties spins it at +omega about
    +x (omega = 2 pi rpm / 60) with nonRotatingPatches () -- hub and blades
    are rotating walls, U = omega x r imposed by MRF on the "propeller" patch.
  * Refinement: blade surface at --refine (min max), its feature edges at max;
    the rotor zone at min-2, a 2 D slipstream cylinder behind it at min-3.
  * Hub and blades are OVERLAPPING closed shells in ONE STL surface. snappy
    keeps only cells reachable from locationInMesh without crossing a
    triangle, and the overlap is enclosed by the hub, so it is the union by
    construction. That is CHECKED, not assumed: after meshing a topoSet counts
    cells inside the hub core and the run stops if there is any.
  * --sector (DRY RUN ONLY): the 1/B wedge |azimuth| <= pi/B about blade 0 of
    a radius-3D cylinder, as a 6-block O-grid whose inner parallelograms touch
    the axis with a regular hex corner (snappy's hexRef8 cannot refine
    collapsed wedge cells); for B = 2 they are exact squares. snappy keeps
    what lies in the wedge; createPatch turns the azimuth planes into a
    rotational cyclicAMI pair (+-360/B about +x). The MESH works (checkMesh
    OK, AMI weights 0.93-1.00, hub core empty). The SOLUTION does not, so the
    sweep refuses it. Measured 2026-10-23, J 0.4: simpleFoam MRF + rotational
    cyclicAMI puts a pressure jump of order (omega r)^2 across the periodic
    planes INSIDE the rotor zone (p -3445 vs +1134 m^2/s^2 at azimuth +-59
    deg, r = 0.1 m; full 360: 8 vs 11). CT vs full 360: +6 % (J 0.4) / +11 %
    (J 0.6) for B = 2 on the EXACT half of the full mesh, -26 % for B = 3.
    TODO: SRFSimpleFoam (relative velocity, SRFFreestreamVelocity far field)
    is the stock periodic-rotor route; this blockMesh topology is reusable.
  * Far field. J > 0: inlet fixed U = J n D, sides slip, outlet p = 0.
    J = 0: inlet and sides totalPressure p0 = 0 + pressureInletOutletVelocity
    (the prop entrains from everywhere), outlet p = 0. Turbulence quantities
    are inletOutlet everywhere at 0.5 % of the 0.75 R blade speed, nut = nu.

Signs (fluid force F, moment M on the "propeller" patch, CofR = origin; the
forces FO gets rhoInf = --rho, so F and M are DIMENSIONAL, not kinematic):

  * T = -F_x: the suction side faces upstream and thrust pushes the body -x.
  * Q = -M_x: the prop turns at +omega about +x, the fluid resists, M_x < 0.
  * P = Q omega, CT = T/(rho n^2 D^4), CQ = Q/(rho n^2 D^5), CP = 2 pi CQ,
    eta = J CT / CP.
  * CT is the WHOLE patch, hub included. The hub is a bluff flat-ended
    cylinder whose pressure drag is a real share of the thrust (measured, J
    0.9 on 168k cells: hub CT -0.0073 against a total 0.0144), and BEMT /
    liftline have no hub. So the faceZone hubSkin (propeller faces inside the
    hub cylinder +2 %) is integrated every iteration and the sweep prints the
    hub's pressure CT and "blades alone" = CT - hub. Not in the CSV.
  * Re75 = W c / nu at 0.75 R with the GEOMETRIC W = sqrt(V^2 + (0.75 omega
    R)^2): it omits the few-percent induction BEMT/liftline include.

Mesh (measured 2026-10-23, runs/foam_mesh; kOmegaSST, Spalding nut +
omegaWallFunction, which blend across y+; 5 prism layers cover ~93 % of the
blade faces at a mean 3.5 of 5):

    preset   cells   y+ avg   CT / CP at J 0.6     J 0.8             J 0.9
    draft    0.17M   50       0.0797 / 0.0774      0.0386 / 0.0547   0.0144 / 0.0393
    (5.66,5) 0.43M   16       0.0818 / 0.0779      0.0404 / 0.0562   0.0164 / 0.0414
    medium   0.83M   13       0.0767 / 0.0732      0.0346 / 0.0498   0.0112 / 0.0349
    fine     1.81M   10       0.0768 / 0.0731      0.0349 / 0.0498   0.0117 / 0.0347

  Convergence is NOT monotone below medium; medium is the first mesh within
  ~1 % of fine (J 0.9 CT, a small number, 4 %). Wall time on 8 ranks: draft
  ~3 min/J, medium ~8 min/J, fine ~20-30 min/J. The laminar model on the
  0.43M mesh moved CP by -0.5 % / -3.5 % (J 0.6 / 0.8): skin friction is only
  5-10 % of the torque, so the wall/transition model is NOT what separates
  this tool from BEMT. kOmegaSSTLM is deliberately not offered: its Tu and
  lambda_theta correlations use |U|, and under MRF U is the ABSOLUTE velocity.

Sweep: one mesh, J run in the order given, each J restarted from the previous
J's latest fields with only the far-field U/p swapped (changeDictionary). A
J runs in chunks of --check-every iterations; after each chunk CT and CQ are
averaged over the last --window iterations, and the J is converged when that
mean has DRIFTED by less than --tol (relative) from the window before it, for
both; --iters caps it. The reported coefficients are the window means. The
band, (max - min) / |mean| over the window, goes to stdout: a snappy mesh
never settles to a fixed point, so the band plateaus (0.5-4 % on the 168k-cell
test mesh, widest at high J) while the mean stops moving -- the band is the
honest iteration noise, the drift is the convergence test.
"""

import argparse
import glob
import math
import os
import re
import shutil
import string
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import propeller as prop  # noqa: E402

TEMPLATES = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         "foam_templates")
# Domain, in diameters: upstream, downstream, half-width of the square box.
UPSTREAM, DOWNSTREAM, HALF = 3.0, 7.0, 3.0
ZONE_R = 0.6            # rotor-zone radius, D; tip is at 0.5
ZONE_MARGIN = 0.1       # rotor-zone axial margin past the blade bbox, D
WAKE_LEN = 2.0          # slipstream refinement length behind the zone, D
TURB_I = 0.005          # freestream turbulence intensity on the 0.75 R speed
HUB_ZONE = "hubSkin"    # faceZone of the hub's propeller faces (hub thrust)
# --quality: (cells-per-D, prism layers, snappy maxGlobalCells). Measured on
# the 2-blade 4412 P/D 0.8 test prop at --refine 5 6 (docstring, "Mesh").
# draft is the historical default, kept so an old command line reproduces.
QUALITY = {"draft": (4.0, 0, 2000000), "medium": (8.0, 5, 2000000),
           "fine": (11.31, 5, 4000000)}


def _die(msg):
    sys.exit(f"foam_mrf: {msg}")


def parse_j(s):
    """`--j` as "0.2,0.4" or "start:stop:step" (stop inclusive), in the order
    given: it is the restart order, each J starting from the one before."""
    try:
        if ":" in s:
            a, b, c = (float(x) for x in s.split(":"))
            if c <= 0 or b < a:
                raise ValueError
            n = int(math.floor((b - a) / c + 1e-9)) + 1
            return [round(a + k * c, 10) for k in range(n)]
        return [float(x) for x in s.replace(",", " ").split()]
    except ValueError:
        _die(f"bad --j '{s}' (want 'a,b,c' or 'start:stop:step')")


def parse_refine(s):
    """`--refine` "L" or "Lmin Lmax" (also "Lmin,Lmax")."""
    try:
        v = [int(x) for x in s.replace(",", " ").split()]
    except ValueError:
        v = []
    v = v * 2 if len(v) == 1 else v
    if len(v) != 2 or not 0 <= v[0] <= v[1] <= 10:
        _die(f"bad --refine '{s}' (want 'L' or 'Lmin Lmax', 0..10)")
    return v


def g9(x):
    return f"{x:.9g}"


def vec(v):
    return "(" + " ".join(map(g9, v)) + ")"


def render(tmpl, /, **v):
    """foam_templates/<tmpl> with $keys substituted (all must be given)."""
    with open(os.path.join(TEMPLATES, tmpl)) as f:
        return string.Template(f.read()).substitute(v)


def put(case, rel, tmpl=None, /, **v):
    """Render template `tmpl` (default: the case path itself) to case/rel."""
    path = os.path.join(case, rel)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write(render(tmpl or rel, **v))


class Setup:
    """Everything derived from the flags, in SI. One place, so the dry run
    prints exactly what the run would use."""

    def __init__(self, a):
        self.a, self.D = a, a.diameter
        self.n = a.rpm / 60.0
        self.omega = 2.0 * math.pi * self.n
        self.lmin, self.lmax = parse_refine(a.refine)
        self.lzone, self.lwake = max(self.lmin - 2, 0), max(self.lmin - 3, 0)
        self.js = parse_j(a.j)
        self.mesh, _ = prop.build(a)
        lo, hi = self.mesh.bounds()
        self.zx0 = lo[0] * self.D - ZONE_MARGIN * self.D
        self.zx1 = hi[0] * self.D + ZONE_MARGIN * self.D
        R = 0.5 * self.D
        self.w75 = self.omega * 0.75 * R
        self.c75 = prop.blade_law(a, 0.375)[0] * self.D
        self.k = 1.5 * (TURB_I * self.w75) ** 2
        self.om = self.k / a.nu                 # nut/nu = 1
        self.rh = a.hub_radius * R
        self.hub_half = 0.5 * a.hub_length * self.D

    def V(self, J):
        return J * self.n * self.D

    def re75(self, J):
        return math.hypot(self.V(J), self.w75) * self.c75 / self.a.nu


# --- case writers -----------------------------------------------------------

def polar(r, th):
    """(y, z) at radius r, azimuth th from +y toward +z (the sense of +omega)."""
    return (r * math.cos(th), r * math.sin(th))


def write_sector_block(case, s, nx):
    """blockMeshDict for the 1/B sector |azimuth| <= pi/B about blade 0: per
    half an inner parallelogram O-C-Q-P (side 1 D, regular hex corner on the
    axis; for B = 2 exact squares, so the blade sits in the same Cartesian
    cells as the box) and two outer blocks out to the radius-HALF D arc.
    perLo0/perHi0 become the cyclicAMI pair (createPatchDict)."""
    a, D = s.a, s.D
    beta = math.pi / a.blades
    h0 = D / a.cells_per_D
    ra, ro = 1.0 * D, HALF * D
    nt, nr = max(2, round(ra / h0)), max(2, round((ro - ra) / h0))
    pts = {"O": (0.0, 0.0), "C": polar(ra, 0.0), "Cr": polar(ro, 0.0)}
    for sg, tag in ((-1, "-"), (1, "+")):
        P = polar(ra, sg * beta)
        pts["P" + tag], pts["Q" + tag] = P, (P[0] + ra, P[1])
        pts["M" + tag] = polar(ro, sg * beta / 2)
        pts["Pr" + tag] = polar(ro, sg * beta)
    names = sorted(pts)
    ix = {k: i for i, k in enumerate(names)}
    N = len(names)
    x0, x1 = -UPSTREAM * D, DOWNSTREAM * D
    # Bases CCW about +x (right-handed hex with x3 = +x); (n1, n2) along
    # v0->v1 and v0->v3.
    blocks = [(["O", "C", "Q+", "P+"], nt, nt), (["C", "Cr", "M+", "Q+"], nr, nt),
              (["Q+", "M+", "Pr+", "P+"], nr, nt), (["O", "P-", "Q-", "C"], nt, nt),
              (["P-", "Pr-", "M-", "Q-"], nr, nt), (["Q-", "M-", "Cr", "C"], nr, nt)]
    arcs = [("Cr", "M+", beta / 4), ("M+", "Pr+", 3 * beta / 4),
            ("Pr-", "M-", -3 * beta / 4), ("M-", "Cr", -beta / 4)]

    def ids(b, o=0):
        return " ".join(str(ix[k] + o) for k in b)

    def sides(*pairs):
        return " ".join(f"({ix[u]} {ix[v]} {ix[v] + N} {ix[u] + N})"
                        for u, v in pairs)
    put(case, "system/blockMeshDict", "system/blockMeshDict.sector",
        vertices="\n".join("    " + vec((x, *pts[k]))
                           for x in (x0, x1) for k in names),
        hexes="\n".join(f"    hex ({ids(b)} {ids(b, N)}) ({n1} {n2} {nx}) "
                        "simpleGrading (1 1 1)" for b, n1, n2 in blocks),
        edges="\n".join(f"    arc {ix[u] + o} {ix[v] + o} {vec((x, *polar(ro, th)))}"
                        for u, v, th in arcs for o, x in ((0, x0), (N, x1))),
        inlet=" ".join(f"({ix[b[0]]} {ix[b[3]]} {ix[b[2]]} {ix[b[1]]})"
                       for b, _, _ in blocks),
        outlet=" ".join(f"({ids(b, N)})" for b, _, _ in blocks),
        sides=sides(("Cr", "M+"), ("M+", "Pr+"), ("Pr-", "M-"), ("M-", "Cr")),
        perlo=sides(("O", "P-"), ("P-", "Pr-")),
        perhi=sides(("P+", "O"), ("Pr+", "P+")))
    ang = 360.0 / a.blades
    put(case, "system/createPatchDict", angle=f"{ang:.12g}", mangle=f"{-ang:.12g}")


def write_mesh_dicts(case, s):
    a, D = s.a, s.D
    x0, x1, h = -UPSTREAM * D, DOWNSTREAM * D, HALF * D
    nx = max(round((UPSTREAM + DOWNSTREAM) * a.cells_per_D), 4)
    ny = max(round(2 * HALF * a.cells_per_D), 4)
    if a.sector:
        write_sector_block(case, s, nx)
        # Off every block face and off the shaft, so the seed is never on a face.
        loc = vec((-0.6123 * UPSTREAM * D,
                   *polar(0.6437 * h, 0.37 * math.pi / a.blades)))
    else:
        put(case, "system/blockMeshDict", x0=x0, x1=x1, h=h, mh=-h, nx=nx, ny=ny)
        loc = vec((-0.6123 * UPSTREAM * D, 0.4137 * h, 0.5371 * h))
    put(case, "system/surfaceFeatureExtractDict")
    zr = g9(ZONE_R * D)
    put(case, "system/snappyHexMeshDict",
        addLayers="true" if a.layers > 0 else "false",
        zx0=g9(s.zx0), zx1=g9(s.zx1), wx1=g9(s.zx1 + WAKE_LEN * D), zr=zr,
        maxCells=a.max_cells, lmin=s.lmin, lmax=s.lmax, lzone=s.lzone,
        lwake=s.lwake, loc=loc,
        layers=(f"propeller {{ nSurfaceLayers {a.layers}; }}"
                if a.layers > 0 else ""))
    # Hub core: inside the polygon's inscribed circle and the hub's length, so
    # a snapped cell centre outside the hub can never be counted. Hub skin:
    # propeller faces whose centre lies within the hub's own cylinder (+2 %),
    # as faceZone HUB_ZONE -- the hub's share of the thrust (it also catches
    # the few blade-root faces inside that cylinder).
    hx, ho = 0.9 * s.hub_half, 1.02 * s.hub_half
    put(case, "system/topoSetDict.hubcheck", mhx=g9(-hx), hx=g9(hx),
        rin=g9(0.9 * s.rh * math.cos(math.pi / a.hub_sides)),
        mho=g9(-ho), ho=g9(ho), hr=g9(1.02 * s.rh), zone=HUB_ZONE)
    if a.np > 1:
        put(case, "system/decomposeParDict", np=a.np)


def write_physics(case, s):
    a = s.a
    put(case, "constant/transportProperties", nu=g9(a.nu))
    kind = "laminar" if a.turbulence == "laminar" else "RAS"
    put(case, "constant/turbulenceProperties",
        f"constant/turbulenceProperties.{kind}", model=a.turbulence)
    put(case, "constant/MRFProperties", omega=f"{s.omega:.12g}")
    put(case, "system/fvSchemes")


def solution(case, J):
    """fvSolution for one J. SIMPLEC (U 0.7) with a freestream; at J = 0
    classic SIMPLE (p 0.2, U 0.5): the pressure far field with no inflow to
    damp it diverged under SIMPLEC at iteration ~150 on one snappy mesh of two
    (measured 2026-10-23, Ux residual 0.13 and climbing)."""
    if J == 0:
        put(case, "system/fvSolution", consistent="no", p=0.2, U=0.5, other=0.5)
    else:
        put(case, "system/fvSolution", consistent="yes", p=1, U=0.7, other=0.6)


def control(case, s, end, chunk):
    """controlDict: forces every iteration, y+ (RAS), and the hub pressure
    thrust -- integral of p n_x over HUB_ZONE, no skin friction -- when the
    mesh carries that zone."""
    fos = "" if s.a.turbulence == "laminar" else render("functions/yPlus")
    if has_hub_zone(case, s.a):
        fos += render("functions/hubP", zone=HUB_ZONE)
    put(case, "system/controlDict", end=end, chunk=chunk, rho=g9(s.a.rho),
        functions=fos)


def mesh_root(case, a):
    return os.path.join(case, "processor0") if a.np > 1 else case


def has_hub_zone(case, a):
    """True when the mesh carries HUB_ZONE (meshed by this version)."""
    try:
        with open(os.path.join(mesh_root(case, a), "constant/polyMesh/faceZones")) as f:
            return re.search(rf"^\s*{HUB_ZONE}\s*$", f.read(), re.M) is not None
    except OSError:
        return False


def far_u(s, J):
    """U far field for one J: fixed inflow, or pressure-driven at J = 0."""
    V = g9(s.V(J))
    if J > 0:
        inlet, sides = f"type fixedValue; value uniform ({V} 0 0);", "type slip;"
    else:
        inlet = sides = "type pressureInletOutletVelocity; value uniform (0 0 0);"
    out = f"type inletOutlet; inletValue uniform (0 0 0); value uniform ({V} 0 0);"
    return {"inlet": inlet, "sides": sides, "outlet": out}


def far_p(J):
    inlet = ("type zeroGradient;" if J > 0
             else "type totalPressure; p0 uniform 0; value uniform 0;")
    return {"inlet": inlet, "sides": inlet, "outlet": "type fixedValue; value uniform 0;"}


def bcs(patches, indent):
    return "".join(f"{' ' * indent}{k} {{ {v} }}\n" for k, v in patches.items())


def write_zero(d, s, J):
    """The initial fields for the first J, into time directory `d`."""
    def field(name, cls, dim, internal, patches):
        with open(os.path.join(d, name), "w") as f:
            f.write(render("0/field", cls=cls, name=name, dim=dim,
                           internal=f"uniform {internal}", patches=bcs(patches, 4)))

    def free(val):
        return f"type inletOutlet; inletValue uniform {g9(val)}; value uniform {g9(val)};"
    os.makedirs(d, exist_ok=True)
    far = ("inlet", "outlet", "sides")
    vec_u = f"({g9(s.V(J))} 0 0)"
    # MRF overwrites the wall value with omega x r.
    field("U", "volVectorField", "[0 1 -1 0 0 0 0]", vec_u,
          dict(far_u(s, J), propeller="type fixedValue; value uniform (0 0 0);"))
    field("p", "volScalarField", "[0 2 -2 0 0 0 0]", "0",
          dict(far_p(J), propeller="type zeroGradient;"))
    t = s.a.turbulence
    if t == "laminar":
        return
    field("nut", "volScalarField", "[0 2 -1 0 0 0 0]", "0",
          dict({f: "type calculated; value uniform 0;" for f in far},
               propeller="type nutUSpaldingWallFunction; value uniform 0;"))
    if t == "kOmegaSST":
        field("k", "volScalarField", "[0 2 -2 0 0 0 0]", g9(s.k),
              dict({f: free(s.k) for f in far},
                   propeller=f"type kqRWallFunction; value uniform {g9(s.k)};"))
        field("omega", "volScalarField", "[0 0 -1 0 0 0 0]", g9(s.om),
              dict({f: free(s.om) for f in far},
                   propeller=f"type omegaWallFunction; value uniform {g9(s.om)};"))
    else:
        nt = 3.0 * s.a.nu
        field("nuTilda", "volScalarField", "[0 2 -1 0 0 0 0]", g9(nt),
              dict({f: free(nt) for f in far},
                   propeller="type fixedValue; value uniform 0;"))


def write_change(case, s, J):
    """changeDictionaryDict swapping only the far-field U and p for a new J."""
    put(case, "system/changeDictionaryDict", U=bcs(far_u(s, J), 12),
        p=bcs(far_p(J), 12))


def write_stl(case, s):
    """The propeller.py model scaled to --diameter metres, as snappy reads it."""
    tri = os.path.join(case, "constant/triSurface")
    os.makedirs(tri, exist_ok=True)
    s.mesh.write_stl(os.path.join(tri, "propeller.stl"), scale=s.D)


# --- running ----------------------------------------------------------------

def foam(case, cmd, log, a, par=False):
    """Run one OpenFOAM command (sourced env), output to `log`. Exits on failure."""
    if par and a.np > 1:
        cmd = f"mpirun --oversubscribe -np {a.np} {cmd} -parallel"
    full = f"source {a.foam_bashrc} >/dev/null 2>&1 && cd '{case}' && {cmd}"
    lp = os.path.join(case, log)
    with open(lp, "w") as f:
        try:
            r = subprocess.run(["bash", "-c", full], stdout=f,
                               stderr=subprocess.STDOUT, timeout=a.timeout)
        except subprocess.TimeoutExpired:
            _die(f"'{cmd}' timed out; see {lp}")
    if r.returncode != 0:
        _die(f"'{cmd}' failed (rc {r.returncode}); see {lp}")
    with open(lp) as f:
        return f.read()


def time_dirs(case, a):
    root = mesh_root(case, a)
    out = []
    for d in os.listdir(root) if os.path.isdir(root) else []:
        try:
            out.append((float(d), d))
        except ValueError:
            pass
    return sorted(out)


def roots(case, a):
    return sorted(glob.glob(os.path.join(case, "processor*"))) if a.np > 1 else [case]


def drop_times(case, a, keep_latest):
    """Remove time directories (all, or all but the latest) in every root."""
    ts = time_dirs(case, a)          # read once: processor0 is a root too
    for _, d in ts[:-1] if keep_latest else ts:
        for r in roots(case, a):
            shutil.rmtree(os.path.join(r, d), ignore_errors=True)


def grab(pat, text):
    m = re.search(pat, text, re.M)
    return m.group(1) if m else "?"


def layer_coverage(log):
    """Prism-layer summary from the snappy log: the share of propeller faces
    extruded at all (last addLayers iteration), the mean layer count over ALL
    faces against the target, and the achieved share of the target thickness."""
    ext = re.findall(r"Extruding (\d+) out of (\d+) faces", log)
    row = re.findall(r"^propeller\s+\d+\s+(\d+)\s+([0-9.eE+-]+)\s+[0-9.eE+-]+\s+"
                     r"([0-9.eE+-]+)\s*$", log, re.M)
    if not ext or not row:
        return "? (no addLayers summary in log.snappyHexMesh)"
    n, tot = map(int, ext[-1])
    target, got, thick = row[-1]
    return (f"{n / tot:.1%} of faces extruded, mean {float(got):.2f} of "
            f"{target} layers ({float(got) / int(target):.0%}), "
            f"{float(thick):.0f}% of the target thickness")


def mesh(case, s):
    a = s.a
    for d in glob.glob(os.path.join(case, "processor*")) + [
            os.path.join(case, d) for d in ("constant/polyMesh", "0", "postProcessing")]:
        shutil.rmtree(d, ignore_errors=True)
    foam(case, "blockMesh", "log.blockMesh", a)
    foam(case, "surfaceFeatureExtract", "log.surfaceFeatureExtract", a)
    if a.np > 1:
        foam(case, "decomposePar -force -no-fields", "log.decomposePar", a)
    sn = foam(case, "snappyHexMesh -overwrite", "log.snappyHexMesh", a, par=True)
    if a.layers > 0:
        print(f"  layers: {layer_coverage(sn)}", flush=True)
    if a.sector:
        foam(case, "createPatch -overwrite", "log.createPatch", a, par=True)
    cm = foam(case, "checkMesh -constant -noZero", "log.checkMesh", a, par=True)
    hc = foam(case, "topoSet -constant -noZero -dict system/topoSetDict.hubcheck",
              "log.topoSet.hubcheck", a, par=True)
    mt = re.search(r"cellSet hubCore now size (\d+)", hc)
    inside = int(mt.group(1)) if mt else -1
    fails = re.search(r"Failed \d+ mesh checks", cm)
    print(f"  mesh: {grab(r'^\s*cells:\s+(\d+)', cm)} cells "
          f"({grab(r'^\s*rotor\s+(\d+)\s+\d+\s+[0-9.eE+-]+ \(', cm)} in the rotor zone, "
          f"{grab(r'^\s*propeller\s+(\d+)', cm)} propeller faces); checkMesh: "
          f"{'Mesh OK' if 'Mesh OK' in cm else (fails.group(0) if fails else '?')}; "
          f"max non-orth {grab(r'non-orthogonality Max:\s*([0-9.eE+-]+)', cm)}, "
          f"max skewness {grab(r'Max skewness = ([0-9.eE+-]+)', cm)}, "
          f"max aspect {grab(r'Max aspect ratio = ([0-9.eE+-]+)', cm)}; "
          f"cells inside the hub core: {inside}", flush=True)
    if inside != 0:
        _die(f"{inside} cells survive inside the hub -- the overlapping shells "
             "were NOT meshed as their union; see log.topoSet.hubcheck")


def series(case, name):
    """{time: {column: value}} from every postProcessing/forces/*/<name>."""
    rows = {}
    for f in sorted(glob.glob(os.path.join(case, "postProcessing/forces/*", name))):
        cols = None
        with open(f) as fh:
            for line in fh:
                if line.startswith("#"):
                    h = line[1:].split()
                    cols = h if h and h[0] == "Time" else cols
                    continue
                v = line.replace("(", " ").replace(")", " ").split()
                if v and cols is not None and len(v) == len(cols):
                    rows[float(v[0])] = dict(zip(cols, map(float, v)))
    return rows


def hub_ct(case, s, its):
    """Mean hub pressure CT over iterations `its` (None without HUB_ZONE).
    p is kinematic, so the force is rho * integral; CT = -F_x/(rho n^2 D^4)."""
    rows = {}
    for f in sorted(glob.glob(os.path.join(case, "postProcessing/hubP/*/surfaceFieldValue.dat"))):
        with open(f) as fh:
            for line in fh:
                v = line.split()
                if len(v) == 2 and not line.startswith("#"):
                    rows[float(v[0])] = float(v[1])
    vals = [rows[t] for t in its if t in rows]
    return -sum(vals) / len(vals) / (s.n ** 2 * s.D ** 4) if vals else None


def coeffs(s, frow, mrow):
    """(T, Q, CT, CQ) of one force/moment row; a sector holds one blade of B."""
    rho, n, D = s.a.rho, s.n, s.D
    B = s.a.blades if s.a.sector else 1
    T, Q = -B * frow["total_x"], -B * mrow["total_x"]
    return T, Q, T / (rho * n * n * D ** 4), Q / (rho * n * n * D ** 5)


def band(xs):
    m = sum(xs) / len(xs)
    return m, (max(xs) - min(xs)) / abs(m) if m else float("inf")


def yplus(case, t):
    """', y+ min/max/avg' on the propeller at time t, from the yPlus FO."""
    for f in glob.glob(os.path.join(case, "postProcessing/yPlus/*/yPlus.dat")):
        with open(f) as fh:
            for v in map(str.split, fh):
                if len(v) == 5 and v[1] == "propeller" and float(v[0]) == t:
                    return (f", y+ min {float(v[2]):.3g} max {float(v[3]):.3g} "
                            f"avg {float(v[4]):.3g}")
    return ""


def sweep(case, s):
    a = s.a
    shutil.rmtree(os.path.join(case, "postProcessing"), ignore_errors=True)
    drop_times(case, a, keep_latest=False)
    for r in roots(case, a):
        write_zero(os.path.join(r, "0"), s, s.js[0])
    out = []
    for idx, J in enumerate(s.js):
        t0 = time.monotonic()
        start = int(time_dirs(case, a)[-1][0])
        solution(case, J)
        if idx > 0:
            write_change(case, s, J)
            foam(case, "changeDictionary -latestTime", f"log.changeDictionary.J{J:.3f}",
                 a, par=True)
        it, conv, res = start, False, None
        while it - start < a.iters:
            step = min(a.check_every, a.iters - (it - start))
            control(case, s, it + step, a.check_every)
            log = foam(case, "simpleFoam", f"log.simpleFoam.J{J:.3f}.{it}", a, par=True)
            it += step
            drop_times(case, a, keep_latest=True)
            pr = re.findall(r"Solving for p, Initial residual = ([0-9.eE+-]+)", log)
            res = float(pr[-1]) if pr else res
            fr, mr = series(case, "force.dat"), series(case, "moment.dat")
            ks = [t for t in sorted(fr) if start < t <= it and t in mr]
            if not ks:
                _die(f"no force output for J {J}; see postProcessing/")
            win = ks[-a.window:]
            vals = [coeffs(s, fr[t], mr[t]) for t in win]
            ct, bct = band([v[2] for v in vals])
            cq, bcq = band([v[3] for v in vals])
            T, Q = (sum(v[0] for v in vals) / len(vals),
                    sum(v[1] for v in vals) / len(vals))
            # Drift: the window mean against the window before it.
            prev = [coeffs(s, fr[t], mr[t]) for t in ks[-2 * a.window:-a.window]]
            dct = dcq = float("inf")
            if len(prev) == a.window:
                dct = abs(ct - sum(v[2] for v in prev) / a.window) / max(abs(ct), 1e-12)
                dcq = abs(cq - sum(v[3] for v in prev) / a.window) / max(abs(cq), 1e-12)
            conv = max(dct, dcq) < a.tol
            print(f"  J {J:.3f} it {it - start:5d}: CT {ct:.5f} (drift {dct:.2%}, "
                  f"band {bct:.2%})  CQ {cq:.6f} (drift {dcq:.2%}, band {bcq:.2%})  "
                  f"p-res {res if res else float('nan'):.2e}", flush=True)
            if conv:
                break
        cp = 2.0 * math.pi * cq
        eta = J * ct / cp if cp > 0 else float("nan")
        out.append((J, ct, cp, cq, eta, T, Q, Q * s.omega, s.re75(J), conv))
        print(f"  J {J:.3f} done: {it - start} iterations, {time.monotonic() - t0:.0f} s, "
              f"{'converged' if conv else 'NOT converged'} (drift CT {dct:.2%}, "
              f"CQ {dcq:.2%}; band CT {bct:.2%}, CQ {bcq:.2%} over {len(win)} its), "
              f"eta {eta:.3f}{yplus(case, it)}", flush=True)
        hct = hub_ct(case, s, win)
        if hct is not None:
            print(f"  J {J:.3f} hub (pressure, faceZone {HUB_ZONE}): CT {hct:+.5f}; "
                  f"blades alone CT {ct - hct:.5f} -- BEMT/liftline have no hub, "
                  "compare them with the blades", flush=True)
    return out


def main():
    p = argparse.ArgumentParser(
        description="OpenFOAM steady-RANS MRF propeller: writes the case, "
                    "meshes it, sweeps J with simpleFoam -> CT/CP/eta curve.")
    prop.add_geometry_args(p)
    p.add_argument("--diameter", type=float, default=0.254, help="metres [0.254]")
    p.add_argument("--rpm", type=float, default=6000.0, help="[6000]")
    p.add_argument("--rho", type=float, default=1.225, help="kg/m^3 [1.225]")
    p.add_argument("--nu", type=float, default=1.5e-5, help="m^2/s [1.5e-5]")
    p.add_argument("--j", default="0.2:0.8:0.2",
                   help="advance ratios: 'a,b,c' or 'start:stop:step' [0.2:0.8:0.2]")
    p.add_argument("--out", default="curve.csv", help="[curve.csv]")
    p.add_argument("--case", default="foam_mrf_case", help="case directory")
    p.add_argument("--quality", default="draft", choices=sorted(QUALITY),
                   help="mesh preset (--cells-per-D, --layers, --max-cells; "
                        "an explicit flag wins): "
                        + "; ".join(f"{k} = {v[0]:g}/{v[1]}/{v[2]}" for k, v in
                                    sorted(QUALITY.items(), key=lambda kv: kv[1][0]))
                        + ". draft is 5-10 %% off in CP; medium is "
                          "mesh-independent to ~1 %% (docstring) [draft]")
    p.add_argument("--cells-per-D", type=float, default=None,
                   help="background blockMesh cells per diameter [preset]")
    p.add_argument("--refine", default="5 6",
                   help="blade surface level 'L' or 'Lmin Lmax'; cell = "
                        "D/(cells-per-D 2^L) ['5 6']")
    p.add_argument("--layers", type=int, default=None,
                   help="prism layers on the propeller patch [preset]")
    p.add_argument("--iters", type=int, default=600, help="max iterations per J [600]")
    p.add_argument("--check-every", type=int, default=50,
                   help="iterations per chunk between convergence checks [50]")
    p.add_argument("--window", type=int, default=100,
                   help="iterations the CT/CQ band is measured over [100]")
    p.add_argument("--tol", type=float, default=0.002,
                   help="converged when the --window mean of CT and CQ moved "
                        "by < tol (relative) from the window before [0.002]")
    p.add_argument("--np", type=int, default=8, help="MPI ranks; 1 = serial [8]")
    p.add_argument("--turbulence", default="kOmegaSST",
                   choices=("kOmegaSST", "laminar", "SpalartAllmaras"))
    p.add_argument("--sector", action="store_true",
                   help="write the 1/B sector case (cyclicAMI periodics); "
                        "--dry-run only, the MRF solution is wrong (docstring)")
    p.add_argument("--max-cells", type=int, default=None,
                   help="snappy maxGlobalCells guard [preset]")
    p.add_argument("--timeout", type=float, default=1800.0,
                   help="seconds per OpenFOAM command [1800]")
    p.add_argument("--foam-bashrc", default=os.environ.get(
        "FOAM_BASHRC", "/opt/OpenFOAM/OpenFOAM-v2606/etc/bashrc"))
    p.add_argument("--reuse-mesh", action="store_true",
                   help="skip meshing; sweep on the case's existing mesh")
    p.add_argument("--dry-run", action="store_true",
                   help="write the case and print it, run nothing")
    a = p.parse_args()
    sys.stdout.reconfigure(line_buffering=True)
    for flag, v in zip(("cells_per_D", "layers", "max_cells"), QUALITY[a.quality]):
        if getattr(a, flag) is None:
            setattr(a, flag, v)
    prop.check_geometry(a)
    if a.section == "lens":
        print("foam_mrf: warning: --section lens has knife edges at both ends; "
              "a RANS case wants --section naca or dat", file=sys.stderr)
    if min(a.diameter, a.rpm, a.rho, a.nu, a.cells_per_D) <= 0:
        _die("--diameter, --rpm, --rho, --nu, --cells-per-D must be > 0")
    if a.layers < 0 or a.iters < 1 or a.check_every < 1 or a.window < 2 or a.np < 1:
        _die("need --layers >= 0, --iters/--check-every >= 1, --window >= 2, --np >= 1")
    if a.sector and a.blades < 2:
        _die("--sector needs --blades >= 2 (a 1/1 sector is the full case)")
    if a.sector and not a.dry_run:
        _die("--sector is --dry-run only: the sector meshes, but MRF + rotational "
             "cyclicAMI gives wrong forces (measured CT +6..11 % at B=2, -26 % "
             "at B=3 vs full 360; see the module docstring). TODO: SRFSimpleFoam")
    if not os.path.isfile(a.foam_bashrc) and not a.dry_run:
        _die(f"no OpenFOAM at {a.foam_bashrc} (--foam-bashrc)")
    s = Setup(a)
    if not s.js or any(not math.isfinite(J) or J < 0 for J in s.js):
        _die("--j values must be finite and >= 0")
    case = os.path.abspath(a.case)

    h0 = s.D / a.cells_per_D
    blade = f"planform {a.planform}" if a.planform else f"P/D {a.pitch}"
    print(f"foam_mrf: D {s.D} m, {a.rpm:g} rpm (n {s.n:.4g}/s, omega {s.omega:.5g} "
          f"rad/s), tip speed {s.omega * s.D / 2:.4g} m/s, {a.blades} blades, "
          f"{blade}, {a.turbulence}, "
          f"{f'1/{a.blades} sector' if a.sector else 'full 360'}")
    print(f"  cells: background {h0 * 1e3:.3g} mm, rotor zone (level {s.lzone}) "
          f"{h0 / 2 ** s.lzone * 1e3:.3g} mm, blade {h0 / 2 ** s.lmin * 1e3:.3g}"
          f"-{h0 / 2 ** s.lmax * 1e3:.3g} mm (chord 0.75R {s.c75 * 1e3:.3g} mm)")
    print(f"  rotor zone: r {ZONE_R * s.D:.4g} m, x [{s.zx0:.4g}, {s.zx1:.4g}] m")
    for J in s.js:
        print(f"  J {J:.3f}: V {s.V(J):.4g} m/s, Re75 {s.re75(J):.4g}"
              f"{'  (static: pressure far field)' if J == 0 else ''}")

    os.makedirs(case, exist_ok=True)
    write_mesh_dicts(case, s)
    write_physics(case, s)
    solution(case, s.js[0])
    control(case, s, a.check_every, a.check_every)
    write_stl(case, s)
    if a.dry_run:
        write_zero(os.path.join(case, "0"), s, s.js[0])
        print(f"  dry run: case written to {case}")
        if a.sector:
            print("  sector mesh: blockMesh; surfaceFeatureExtract; "
                  "snappyHexMesh -overwrite; createPatch -overwrite")
        return
    if not a.reuse_mesh:
        t0 = time.monotonic()
        mesh(case, s)
        print(f"  meshing took {time.monotonic() - t0:.0f} s", flush=True)
    rows = sweep(case, s)
    with open(a.out, "w") as f:
        f.write("J,CT,CP,CQ,eta,T_N,Q_Nm,P_W,Re75,converged\n")
        for r in rows:
            f.write(",".join(f"{x:.6g}" for x in r[:9]) + f",{int(r[9])}\n")
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
