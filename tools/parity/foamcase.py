"""OpenFOAM case writers for the dns parity suite (tools/parity/DESIGN.md).

Writes ordinary v2606 cases; runs nothing. suite.py drives the utilities.

Frames: body at the origin, freestream +x, lift +y, reference length L = 1 m
(cylinder D, section chord c, rotor swept diameter D). The box is the dns box,
placed from dns' own --dry-run placement, so blockage is matched exactly.

Three physics:
  rhoPimpleFoam (tier A): SI air, calorically perfect gamma = 1.4 (Cp = 3.5 R),
    constant mu from Re, Pr = dns' --prandtl; adiabatic walls; IC at rest + the
    dns --kick vortex; inlet U ramped by dns' smoothstep over the dns --ramp;
    outlet waveTransmissive p; free-slip laterals (dns' walled laterals).
  simpleFoam (tier B): kinematic, U = 1 (rotor: U = J, n = 1), nu = U L / Re,
    IC = freestream, inlet fixed U, outlet p = 0, slip laterals.
  rhoSimpleFoam (tier Bc): tier A's gas and Mach, steady; IC = freestream,
    inlet fixed U and T, outlet fixed p, slip laterals, adiabatic walls.

U convection is a per-case choice (cases.toml `u_scheme`): "linear" (central,
the parity rule of CLAUDE.md) or "linearUpwind" (the scheme every pre-W5 run
used; stale for parity).
"""
import math
import os
import string
import sys

import numpy as np

TOOLS = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TEMPLATES = os.path.join(TOOLS, "foam_templates")
# the shared template with the refinement cylinders split off the zone (zone study)
ZONE_SNAPPY = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "snappyHexMeshDict.zone")

RR = 8314.46261815324          # OpenFOAM thermodynamic::RR (1e3 * NA k)
MOLW = 28.9644
RGAS = RR / MOLW
GAMMA = 1.4
CP = GAMMA / (GAMMA - 1.0) * RGAS
T_INF = 288.15
P_INF = 101325.0
RHO_INF = P_INF / (RGAS * T_INF)
C_INF = math.sqrt(GAMMA * RGAS * T_INF)

HDR = """FoamFile
{{
    version     2.0;
    format      ascii;
    class       {cls};
    object      {obj};
}}
"""


def put(case, rel, body, cls="dictionary"):
    path = os.path.join(case, rel)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write(HDR.format(cls=cls, obj=os.path.basename(rel)) + body)


def g(x):
    return f"{x:.12g}"


def vec(v):
    return "(" + " ".join(g(x) for x in v) + ")"


def smoothstep(s):
    s = min(max(s, 0.0), 1.0)
    return s * s * (3.0 - 2.0 * s)


# --- meshes -------------------------------------------------------------------

def _geo_count(L, d1, qmax):
    n = 1
    while d1 * (qmax ** n - 1) / (qmax - 1) < L:
        n += 1
    return n


def _geo_total(L, d1, n):
    """Total expansion ratio (last/first) of n cells from d1 spanning L."""
    lo, hi = 1.0, 1.5
    for _ in range(200):
        q = 0.5 * (lo + hi)
        tot = d1 * (q ** n - 1) / (q - 1) if q > 1 else d1 * n
        lo, hi = (lo, q) if tot > L else (q, hi)
    return q ** (n - 1)


def _first_cell_ratio(L, n, d1):
    """Total expansion (last/first) giving first cell d1 over n cells on L."""
    lo, hi = 1.0000001, 1.5
    for _ in range(200):
        q = 0.5 * (lo + hi)
        d = L * (q - 1) / (q ** n - 1)
        lo, hi = (q, hi) if d > d1 else (lo, q)
    return q ** (n - 1)


def cylinder_mesh(case, box, nq, nr):
    """O-H body-fitted 2D mesh (port of runs/verify/L1_ghostwall/foam/mkcase.py,
    asymmetric box). box = dict(xu, xd, yb, yt) in D from the centre; D = 1.
    Patches: inlet outlet bottom top (symmetryPlane), body (wall), frontAndBack."""
    r, s = 0.5, 1.5
    a = r / math.sqrt(2)
    arc = 2 * math.pi * r / (4 * nq)
    gr = _first_cell_ratio(s - r, nr, 0.5 * arc)
    d1o = max(0.5 * arc * gr, 2 * s / nq)
    xu, xd, yb, yt = box["xu"], box["xd"], box["yb"], box["yt"]
    nu_, nd_ = _geo_count(xu - s, d1o, 1.06), _geo_count(xd - s, d1o, 1.04)
    nb_, nt_ = _geo_count(yb - s, d1o, 1.06), _geo_count(yt - s, d1o, 1.06)
    gu, gd = _geo_total(xu - s, d1o, nu_), _geo_total(xd - s, d1o, nd_)
    gb, gt = _geo_total(yb - s, d1o, nb_), _geo_total(yt - s, d1o, nt_)
    X, Y = [-xu, -s, s, xd], [-yb, -s, s, yt]
    verts, idx = [], {}
    for j in range(4):
        for i in range(4):
            idx[f"P{i}{j}"] = len(verts)
            verts.append((X[i], Y[j]))
    for k, (x, y) in enumerate(((-a, -a), (a, -a), (a, a), (-a, a))):
        idx[f"c{k}"] = len(verts)
        verts.append((x, y))
    nv = len(verts)

    def hexv(names):
        lo = [idx[n] for n in names]
        return lo + [k + nv for k in lo]

    blocks = []
    for q in (("c1", "P21", "P22", "c2"), ("c2", "P22", "P12", "c3"),
              ("c3", "P12", "P11", "c0"), ("c0", "P11", "P21", "c1")):
        blocks.append((hexv(q), (nr, nq, 1), f"({gr:.8g} 1 1)"))
    ncx, ncy = [nu_, nq, nd_], [nb_, nq, nt_]
    gx, gy = [1.0 / gu, 1.0, gd], [1.0 / gb, 1.0, gt]
    for j in range(3):
        for i in range(3):
            if i == 1 and j == 1:
                continue
            q = (f"P{i}{j}", f"P{i+1}{j}", f"P{i+1}{j+1}", f"P{i}{j+1}")
            blocks.append((hexv(q), (ncx[i], ncy[j], 1), f"({gx[i]:.8g} {gy[j]:.8g} 1)"))
    arcs = [("c1", "c2", (r, 0)), ("c2", "c3", (0, r)), ("c3", "c0", (-r, 0)),
            ("c0", "c1", (0, -r))]

    def bface(n0, n1):
        a0, a1 = idx[n0], idx[n1]
        return (a0, a1, a1 + nv, a0 + nv)

    patches = [
        ("inlet", "patch", [bface("P00", "P01"), bface("P01", "P02"), bface("P02", "P03")]),
        ("outlet", "patch", [bface("P30", "P31"), bface("P31", "P32"), bface("P32", "P33")]),
        ("bottom", "symmetryPlane", [bface("P00", "P10"), bface("P10", "P20"), bface("P20", "P30")]),
        ("top", "symmetryPlane", [bface("P03", "P13"), bface("P13", "P23"), bface("P23", "P33")]),
        ("body", "wall", [bface("c0", "c1"), bface("c1", "c2"), bface("c2", "c3"), bface("c3", "c0")]),
    ]
    out = ["scale 1;", "vertices", "("]
    for z in (0.0, 1.0):
        out += [f"    ({x:.12g} {y:.12g} {z})" for x, y in verts]
    out += [");", "blocks", "("]
    out += [f"    hex ({' '.join(map(str, v))}) ({n[0]} {n[1]} {n[2]}) simpleGrading {gg}"
            for v, n, gg in blocks]
    out += [");", "edges", "("]
    for p0, p1, (mx, my) in arcs:
        for z, off in ((0.0, 0), (1.0, nv)):
            out.append(f"    arc {idx[p0]+off} {idx[p1]+off} ({mx} {my} {z})")
    out += [");", "boundary", "("]
    for name, typ, fl in patches:
        out.append(f"    {name} {{ type {typ}; faces ( "
                   + " ".join("(" + " ".join(map(str, fc)) + ")" for fc in fl) + " ); }")
    fb = []
    for v, _, _ in blocks:
        fb += [f"({v[0]} {v[3]} {v[2]} {v[1]})", f"({v[4]} {v[5]} {v[6]} {v[7]})"]
    out += [f"    frontAndBack {{ type empty; faces ( {' '.join(fb)} ); }}", ");"]
    put(case, "system/blockMeshDict", "\n".join(out) + "\n")
    cells = 4 * nq * nr + (nu_ + nd_) * (nb_ + nq + nt_) + (nb_ + nt_) * nq
    return dict(cells=cells, first=0.5 * arc, arc=arc, counts=(nu_, nd_, nb_, nt_))


def section_mesh(case, ring, box, nt, nr, d1, s=1.2, tan_ratio=4.0):
    """2D body-fitted O-H mesh about the lens section, wind frame, bounded by
    the dns box. ring = LE, lower crest, TE, upper crest (chord 1, origin = dns
    placement point); box = dict(xu, xd, yb, yt).

    Inner O-ring: 8 blocks from the kite (its 4 vertices + 4 edge midpoints) to
    a square of half-side s; every kite vertex maps to a square side point
    along its outward bisector (LE -> (-s,0), crests -> (xm, -+s), TE ->
    (s,0)), every edge midpoint to a square corner, so no block corner is near
    180 deg. (The 4-block kite-to-box-corner topology of L3's make_case.py box
    variant puts the crests' radial lines almost along the wall: 167 deg corners,
    negative volumes.) Outer: 12 H blocks to the box, the cylinder mesh's way.
    nt cells per square half-side, nr radial in the ring, first cell d1;
    kite-edge spacing clustered toward the vertices by tan_ratio.
    Patches: body (wall), inlet, outlet, sides (both y faces), frontAndBack."""
    le, lc, te, uc = ring
    mid = lambda p, q: (0.5 * (p[0] + q[0]), 0.5 * (p[1] + q[1]))  # noqa: E731
    m = [le, mid(le, lc), lc, mid(lc, te), te, mid(te, uc), uc, mid(uc, le)]
    xm = 0.5 * (lc[0] + uc[0])
    q = [(-s, 0.0), (-s, -s), (xm, -s), (s, -s), (s, 0.0), (s, s), (xm, s), (-s, s)]
    xu, xd, yb, yt = box["xu"], box["xd"], box["yb"], box["yt"]
    X, Y = [-xu, -s, xm, s, xd], [-yb, -s, 0.0, s, yt]
    # tangential counts per O block (each must equal its square side's H count)
    ncl = max(2, round(nt * (xm + s) / s))
    ncr = max(2, round(nt * (s - xm) / s))
    ntan = [nt, ncl, ncr, nt, nt, ncr, ncl, nt]
    rl = [math.hypot(qq[0] - mm[0], qq[1] - mm[1]) for mm, qq in zip(m, q)]
    er = _first_cell_ratio(sum(rl) / len(rl), nr, d1)
    dlast = d1 * er
    d1o = max(dlast, s / nt)
    nu_, nd_ = _geo_count(xu - s, d1o, 1.06), _geo_count(xd - s, d1o, 1.04)
    nb_, nt_ = _geo_count(yb - s, d1o, 1.06), _geo_count(yt - s, d1o, 1.06)
    gu, gd = _geo_total(xu - s, d1o, nu_), _geo_total(xd - s, d1o, nd_)
    gb, gt = _geo_total(yb - s, d1o, nb_), _geo_total(yt - s, d1o, nt_)
    verts, idx = [], {}

    def add(key, p):
        idx[key] = len(verts)
        verts.append(p)
    for k, p in enumerate(m):
        add(("m", k), p)
    for j in range(5):
        for i in range(5):
            add(("P", i, j), (X[i], Y[j]))
    qkey = [("P", 1, 2), ("P", 1, 1), ("P", 2, 1), ("P", 3, 1), ("P", 3, 2), ("P", 3, 3),
            ("P", 2, 3), ("P", 1, 3)]
    nv = len(verts)

    def hexv(keys):
        lo = [idx[k] for k in keys]
        return lo + [k + nv for k in lo]

    blocks = []
    for k in range(8):
        k1 = (k + 1) % 8
        g_in = tan_ratio if k % 2 == 0 else 1.0 / tan_ratio
        eg = (f"edgeGrading ({er:.8g} {er:.8g} {er:.8g} {er:.8g} "
              f"{g_in:.8g} 1 1 {g_in:.8g} 1 1 1 1)")
        blocks.append((hexv([("m", k), qkey[k], qkey[k1], ("m", k1)]), (nr, ntan[k], 1), eg))
    ncx, ncy = [nu_, ncl, ncr, nd_], [nb_, nt, nt, nt_]
    gx, gy = [1.0 / gu, 1.0, 1.0, gd], [1.0 / gb, 1.0, 1.0, gt]
    for j in range(4):
        for i in range(4):
            if i in (1, 2) and j in (1, 2):
                continue
            keys = [("P", i, j), ("P", i + 1, j), ("P", i + 1, j + 1), ("P", i, j + 1)]
            blocks.append((hexv(keys), (ncx[i], ncy[j], 1),
                           f"simpleGrading ({gx[i]:.8g} {gy[j]:.8g} 1)"))

    def bface(k0, k1):
        a0, a1 = idx[k0], idx[k1]
        return f"({a0} {a1} {a1 + nv} {a0 + nv})"

    patches = {
        "body": ("wall", [bface(("m", k), ("m", (k + 1) % 8)) for k in range(8)]),
        "inlet": ("patch", [bface(("P", 0, j), ("P", 0, j + 1)) for j in range(4)]),
        "outlet": ("patch", [bface(("P", 4, j), ("P", 4, j + 1)) for j in range(4)]),
        "sides": ("patch", [bface(("P", i, 0), ("P", i + 1, 0)) for i in range(4)]
                  + [bface(("P", i, 4), ("P", i + 1, 4)) for i in range(4)]),
    }
    out = ["scale 1;", "vertices", "("]
    for z in (0.0, 1.0):
        out += [f"    ({x:.12g} {y:.12g} {z})" for x, y in verts]
    out += [");", "blocks", "("]
    out += [f"    hex ({' '.join(map(str, v))}) ({n[0]} {n[1]} {n[2]}) {gg}" for v, n, gg in blocks]
    out += [");", "edges ();", "boundary", "("]
    for name, (typ, fl) in patches.items():
        out.append(f"    {name} {{ type {typ}; faces ( {' '.join(fl)} ); }}")
    fb = []
    for v, _, _ in blocks:
        fb += [f"({v[0]} {v[3]} {v[2]} {v[1]})", f"({v[4]} {v[5]} {v[6]} {v[7]})"]
    out += [f"    frontAndBack {{ type empty; faces ( {' '.join(fb)} ); }}", ");"]
    put(case, "system/blockMeshDict", "\n".join(out) + "\n")
    cells = nr * sum(ntan) + sum(ncx[i] * ncy[j] for j in range(4) for i in range(4)
                                 if not (i in (1, 2) and j in (1, 2)))
    return dict(cells=cells, first=d1, radial_ratio=er, counts=(nu_, nd_, nb_, nt_))


def render(tmpl, **v):
    with open(os.path.join(TEMPLATES, tmpl)) as f:
        return string.Template(f.read()).substitute(v)


def rotor_mesh(case, stl_src, box, cells_per_d, lmin, lmax, lzone, lwake,
               zone_r, zone_x, wake_len, hub, np_, refine_r=None):
    """Background box (dns box, shaft = +x through the origin) + snappy with the
    foam_mrf snappyHexMeshDict template; MRF faceZone/cellZone 'rotor' =
    cylinder radius zone_r over x in zone_x; wake refinement to wake_len.
    refine_r (zone study only): the lzone/lwake refinement cylinders keep this
    radius while the zone moves, so the study varies the zone, not the mesh;
    None = the shared template, refinement radius = zone_r."""
    x0, x1, y0, y1, z0, z1 = (-box["xu"], box["xd"], -box["yb"], box["yt"],
                              -box["zb"], box["zt"])
    nx = max(round((x1 - x0) * cells_per_d), 4)
    ny = max(round((y1 - y0) * cells_per_d), 4)
    nz = max(round((z1 - z0) * cells_per_d), 4)
    put(case, "system/blockMeshDict", f"""scale 1;
vertices
(
    ({g(x0)} {g(y0)} {g(z0)}) ({g(x1)} {g(y0)} {g(z0)}) ({g(x1)} {g(y1)} {g(z0)}) ({g(x0)} {g(y1)} {g(z0)})
    ({g(x0)} {g(y0)} {g(z1)}) ({g(x1)} {g(y0)} {g(z1)}) ({g(x1)} {g(y1)} {g(z1)}) ({g(x0)} {g(y1)} {g(z1)})
);
blocks ( hex (0 1 2 3 4 5 6 7) ({nx} {ny} {nz}) simpleGrading (1 1 1) );
edges ();
boundary
(
    inlet  {{ type patch; faces ((0 4 7 3)); }}
    outlet {{ type patch; faces ((1 2 6 5)); }}
    sides  {{ type patch; faces ((0 1 5 4) (3 7 6 2) (0 3 2 1) (4 5 6 7)); }}
);
""")
    tri = os.path.join(case, "constant/triSurface")
    os.makedirs(tri, exist_ok=True)
    with open(stl_src, "rb") as f, open(os.path.join(tri, "propeller.stl"), "wb") as o:
        o.write(f.read())
    with open(os.path.join(case, "system/surfaceFeatureExtractDict"), "w") as f:
        f.write(render("system/surfaceFeatureExtractDict"))
    # seed point off every face and the shaft, upstream of the rotor
    loc = vec((0.5 * x0, 0.4137 * y1, 0.5371 * z1))
    keys = dict(addLayers="false", zx0=g(zone_x[0]), zx1=g(zone_x[1]),
                wx1=g(zone_x[1] + wake_len), zr=g(zone_r), maxCells=8000000, lmin=lmin,
                lmax=lmax, lzone=lzone, lwake=lwake, loc=loc, layers="")
    if refine_r is None or refine_r == zone_r:
        sh = render("system/snappyHexMeshDict", **keys)
    else:
        with open(ZONE_SNAPPY) as f:
            sh = string.Template(f.read()).substitute(keys, rr=g(refine_r))
    with open(os.path.join(case, "system/snappyHexMeshDict"), "w") as f:
        f.write(sh)
    hx, ho = 0.9 * hub["half"], 1.02 * hub["half"]
    with open(os.path.join(case, "system/topoSetDict.hubcheck"), "w") as f:
        f.write(render("system/topoSetDict.hubcheck", mhx=g(-hx), hx=g(hx),
                       rin=g(0.9 * hub["r"] * math.cos(math.pi / hub["sides"])),
                       mho=g(-ho), ho=g(ho), hr=g(1.02 * hub["r"]), zone="hubSkin"))
    decompose(case, np_)
    return dict(background=(nx, ny, nz), h0=1.0 / cells_per_d,
                h_blade=(1.0 / cells_per_d / 2 ** lmin, 1.0 / cells_per_d / 2 ** lmax),
                h_zone=1.0 / cells_per_d / 2 ** lzone)


def decompose(case, np_):
    put(case, "system/decomposeParDict",
        f"numberOfSubdomains {np_};\nmethod scotch;\n")


# --- physics ------------------------------------------------------------------

def tier_a_state(mach, re, L=1.0):
    """SI freestream for (M, Re on L)."""
    U = mach * C_INF
    mu = RHO_INF * U * L / re
    return dict(U=U, rho=RHO_INF, p=P_INF, T=T_INF, mu=mu, c=C_INF, L=L)


def physics_a(case, st, pr, mrf=None):
    put(case, "constant/thermophysicalProperties", f"""
thermoType
{{
    type            heRhoThermo;
    mixture         pureMixture;
    transport       const;
    thermo          hConst;
    equationOfState perfectGas;
    specie          specie;
    energy          sensibleInternalEnergy;
}}
mixture
{{
    specie {{ molWeight {g(MOLW)}; }}
    thermodynamics {{ Cp {g(CP)}; Hf 0; }}
    transport {{ mu {g(st['mu'])}; Pr {g(pr)}; }}
}}
""")
    put(case, "constant/turbulenceProperties", "simulationType laminar;\n")
    put(case, "constant/momentumTransport", "simulationType laminar;\n")
    if mrf:
        put(case, "constant/MRFProperties", _mrf(mrf))


def physics_b(case, nu, mrf=None):
    put(case, "constant/transportProperties", f"transportModel Newtonian;\nnu {g(nu)};\n")
    put(case, "constant/turbulenceProperties", "simulationType laminar;\n")
    put(case, "constant/momentumTransport", "simulationType laminar;\n")
    if mrf:
        put(case, "constant/MRFProperties", _mrf(mrf))


def _mrf(omega):
    return f"""rotor
{{
    cellZone    rotor;
    active      yes;
    nonRotatingPatches ();
    origin      (0 0 0);
    axis        (1 0 0);
    omega       {g(omega)};
}}
"""


U_SCHEMES = ("linear", "linearUpwind")


def u_div(u_scheme, bounded):
    """div(phi,U) entry for cases.toml u_scheme; bounded for the steady solvers."""
    if u_scheme not in U_SCHEMES:
        raise ValueError(f"u_scheme '{u_scheme}' (have: {', '.join(U_SCHEMES)})")
    s = "Gauss linear" if u_scheme == "linear" else "Gauss linearUpwind grad(U)"
    return ("bounded " + s) if bounded else s


def schemes_a(case, limited=False, u_scheme="linearUpwind"):
    lap = "Gauss linear limited corrected 0.33" if limited else "Gauss linear corrected"
    sng = "limited corrected 0.33" if limited else "corrected"
    gradu = "cellLimited Gauss linear 1" if limited else "Gauss linear"
    put(case, "system/fvSchemes", f"""
ddtSchemes      {{ default Euler; }}
gradSchemes     {{ default Gauss linear; grad(U) {gradu}; }}
divSchemes
{{
    default         none;
    div(phi,U)      {u_div(u_scheme, False)};
    div(phi,e)      Gauss linearUpwind grad(e);
    div(phi,K)      Gauss linear;
    div(phiv,p)     Gauss linear;
    div(phiMRF,p)   Gauss linear;
    div(((rho*nuEff)*dev2(T(grad(U))))) Gauss linear;
}}
laplacianSchemes {{ default {lap}; }}
interpolationSchemes {{ default linear; }}
snGradSchemes   {{ default {sng}; }}
""")


def solution_a(case, nonorth, nouter=1):
    put(case, "system/fvSolution", f"""
solvers
{{
    "rho.*" {{ solver diagonal; }}
    p {{ solver GAMG; smoother GaussSeidel; tolerance 1e-9; relTol 0.01; }}
    pFinal {{ $p; relTol 0; }}
    "(U|e)" {{ solver smoothSolver; smoother symGaussSeidel; tolerance 1e-10; relTol 0.1; }}
    "(U|e)Final" {{ $U; relTol 0; }}
}}
PIMPLE
{{
    momentumPredictor yes;
    transonic       no;
    nOuterCorrectors {nouter};
    nCorrectors     2;
    nNonOrthogonalCorrectors {nonorth};
}}
relaxationFactors {{ equations {{ ".*" 1; }} }}
""")


def schemes_b(case, limited=False, u_scheme="linearUpwind"):
    lap = "Gauss linear limited corrected 0.33" if limited else "Gauss linear corrected"
    sng = "limited corrected 0.33" if limited else "corrected"
    gradu = "cellLimited Gauss linear 1" if limited else "Gauss linear"
    put(case, "system/fvSchemes", f"""
ddtSchemes      {{ default steadyState; }}
gradSchemes     {{ default Gauss linear; grad(U) {gradu}; }}
divSchemes
{{
    default         none;
    div(phi,U)      {u_div(u_scheme, True)};
    div((nuEff*dev2(T(grad(U))))) Gauss linear;
}}
laplacianSchemes {{ default {lap}; }}
interpolationSchemes {{ default linear; }}
snGradSchemes   {{ default {sng}; }}
""")


# tier Bc energy convection (recorded in the report; the parity rule covers U)
E_DIV_BC = "bounded Gauss linearUpwind grad(e)"


def schemes_bc(case, limited=False, u_scheme="linear"):
    """rhoSimpleFoam: tier B's steady discretization + tier A's energy terms
    (e linearUpwind as tier A, Ekp central as tier A's K)."""
    lap = "Gauss linear limited corrected 0.33" if limited else "Gauss linear corrected"
    sng = "limited corrected 0.33" if limited else "corrected"
    gradu = "cellLimited Gauss linear 1" if limited else "Gauss linear"
    put(case, "system/fvSchemes", f"""
ddtSchemes      {{ default steadyState; }}
gradSchemes     {{ default Gauss linear; grad(U) {gradu}; }}
divSchemes
{{
    default         none;
    div(phi,U)      {u_div(u_scheme, True)};
    div(phi,e)      {E_DIV_BC};
    div(phi,Ekp)    bounded Gauss linear;
    div(phiMRF,p)   Gauss linear;
    div(((rho*nuEff)*dev2(T(grad(U))))) Gauss linear;
}}
laplacianSchemes {{ default {lap}; }}
interpolationSchemes {{ default linear; }}
snGradSchemes   {{ default {sng}; }}
""")


def solution_bc(case, nonorth, consistent, p_relax, u_relax, e_relax):
    """rhoSimpleFoam SIMPLE(C), subsonic (transonic no): the rotor's relative
    Mach stays below 0.5."""
    put(case, "system/fvSolution", f"""
solvers
{{
    p {{ solver GAMG; smoother GaussSeidel; tolerance 1e-10; relTol 0.01; }}
    "(U|e)" {{ solver smoothSolver; smoother symGaussSeidel; tolerance 1e-12; relTol 0.01; }}
}}
SIMPLE
{{
    nNonOrthogonalCorrectors {nonorth};
    consistent      {consistent};
    transonic       no;
    pMinFactor      0.5;
    pMaxFactor      2.0;
}}
relaxationFactors {{ fields {{ p {p_relax}; rho 1; }} equations {{ U {u_relax}; e {e_relax}; }} }}
cache {{ grad(U); }}
""")


def solution_b(case, nonorth, consistent, p_relax, u_relax):
    put(case, "system/fvSolution", f"""
solvers
{{
    p {{ solver GAMG; smoother GaussSeidel; tolerance 1e-10; relTol 0.01; }}
    U {{ solver smoothSolver; smoother symGaussSeidel; tolerance 1e-12; relTol 0.01; }}
}}
SIMPLE
{{
    nNonOrthogonalCorrectors {nonorth};
    consistent      {consistent};
}}
relaxationFactors {{ fields {{ p {p_relax}; }} equations {{ U {u_relax}; }} }}
cache {{ grad(U); }}
""")


# --- fields -------------------------------------------------------------------

def boost_sources(case, U, t_ramp):
    """dns' quiet start is a uniform Galilean boost of the whole field
    (runner_comp.hip boostX), which radiates no acoustics. The foam analogue:
    the uniform acceleration a(t) = U d(smoothstep)/dt as a momentum source
    rho a, plus its work rho a u_x on the internal energy (a boost adds kinetic
    energy at fixed internal energy), while the inlet follows the same ramp.
    An inlet-only ramp from rest drives a compression pulse of order rho c U
    (5x the dynamic pressure at M 0.2) that rings in the box for the whole run."""
    a = (f"(time() < {g(t_ramp)} ? {g(6.0 * U / t_ramp)}*(time()/{g(t_ramp)})"
         f"*(1.0 - time()/{g(t_ramp)}) : 0.0)")
    put(case, "system/fvOptions", f"""
boost
{{
    type            vectorSemiImplicitSource;
    volumeMode      specific;
    selectionMode   all;
    sources
    {{
        U {{ explicit {{ type exprField; expression #{{ vector(rho*{a}, 0, 0) #}}; }} }}
    }}
}}
boostWork
{{
    type            scalarSemiImplicitSource;
    volumeMode      specific;
    selectionMode   all;
    sources
    {{
        e {{ explicit {{ type exprField; expression #{{ rho*U.x()*{a} #}}; }} }}
    }}
}}
""")


def ramp_table(U, t_ramp, n=41):
    """uniformFixedValue table of dns' smoothstep freestream ramp (x only)."""
    rows = [f"({g(t_ramp * k / (n - 1))} ({g(U * smoothstep(k / (n - 1)))} 0 0))"
            for k in range(n)]
    rows.append(f"({g(t_ramp * 1e6)} ({g(U)} 0 0))")
    return "table ( " + " ".join(rows) + " )"


def fields_a(case, st, patches, t_ramp, box_len, mrf=False, internal_u=None):
    """rhoPimpleFoam 0/U, 0/p, 0/T. patches: names -> role (inlet, outlet,
    wall, lateral, empty, symmetryPlane). mrf: the wall U is set by MRF."""
    U, p, T = st["U"], st["p"], st["T"]

    def bc(field):
        out = []
        for name, role in patches.items():
            if role == "inlet":
                v = {"U": f"type uniformFixedValue; uniformValue {ramp_table(U, t_ramp)}; "
                          "value uniform (0 0 0);",
                     "p": "type zeroGradient;",
                     "T": f"type fixedValue; value uniform {g(T)};"}[field]
            elif role == "outlet":
                v = {"U": "type inletOutlet; inletValue uniform (0 0 0); value uniform (0 0 0);",
                     "p": (f"type waveTransmissive; field p; phi phi; rho rho; psi thermo:psi; "
                           f"gamma {GAMMA}; fieldInf {g(p)}; lInf {g(box_len)}; "
                           f"value uniform {g(p)};"),
                     "T": f"type inletOutlet; inletValue uniform {g(T)}; value uniform {g(T)};"}[field]
            elif role == "wall":
                v = {"U": "type fixedValue; value uniform (0 0 0);" if mrf else
                          "type noSlip;",
                     "p": "type zeroGradient;", "T": "type zeroGradient;"}[field]
            elif role == "lateral":
                v = "type slip;" if field == "U" else "type zeroGradient;"
            elif role == "symmetryPlane":
                v = "type symmetryPlane;"
            else:
                v = "type empty;"
            out.append(f"    {name} {{ {v} }}")
        return "\n".join(out)

    ui = internal_u if internal_u is not None else "uniform (0 0 0)"
    put(case, "0/U", f"dimensions [0 1 -1 0 0 0 0];\ninternalField {ui};\n"
        f"boundaryField\n{{\n    #includeEtc \"caseDicts/setConstraintTypes\"\n{bc('U')}\n}}\n",
        "volVectorField")
    put(case, "0/p", f"dimensions [1 -1 -2 0 0 0 0];\ninternalField uniform {g(p)};\n"
        f"boundaryField\n{{\n    #includeEtc \"caseDicts/setConstraintTypes\"\n{bc('p')}\n}}\n",
        "volScalarField")
    put(case, "0/T", f"dimensions [0 0 0 1 0 0 0];\ninternalField uniform {g(T)};\n"
        f"boundaryField\n{{\n    #includeEtc \"caseDicts/setConstraintTypes\"\n{bc('T')}\n}}\n",
        "volScalarField")


def fields_b(case, U, patches, mrf=False):
    def bc(field):
        out = []
        for name, role in patches.items():
            if role == "inlet":
                v = {"U": f"type fixedValue; value uniform ({g(U)} 0 0);",
                     "p": "type zeroGradient;"}[field]
            elif role == "outlet":
                v = {"U": f"type inletOutlet; inletValue uniform (0 0 0); value uniform ({g(U)} 0 0);",
                     "p": "type fixedValue; value uniform 0;"}[field]
            elif role == "wall":
                v = {"U": "type fixedValue; value uniform (0 0 0);" if mrf else "type noSlip;",
                     "p": "type zeroGradient;"}[field]
            elif role == "lateral":
                v = "type slip;"
            elif role == "symmetryPlane":
                v = "type symmetryPlane;"
            else:
                v = "type empty;"
            out.append(f"    {name} {{ {v} }}")
        return "\n".join(out)

    put(case, "0/U", f"dimensions [0 1 -1 0 0 0 0];\ninternalField uniform ({g(U)} 0 0);\n"
        f"boundaryField\n{{\n    #includeEtc \"caseDicts/setConstraintTypes\"\n{bc('U')}\n}}\n",
        "volVectorField")
    put(case, "0/p", "dimensions [0 2 -2 0 0 0 0];\ninternalField uniform 0;\n"
        f"boundaryField\n{{\n    #includeEtc \"caseDicts/setConstraintTypes\"\n{bc('p')}\n}}\n",
        "volScalarField")


def fields_bc(case, st, patches, mrf=False):
    """rhoSimpleFoam 0/U, 0/p, 0/T: tier B's steady BCs in tier A's SI gas
    (outlet p = p_inf, inlet T = T_inf, adiabatic walls); IC = freestream."""
    U, p, T = st["U"], st["p"], st["T"]

    def bc(field):
        out = []
        for name, role in patches.items():
            if role == "inlet":
                v = {"U": f"type fixedValue; value uniform ({g(U)} 0 0);",
                     "p": "type zeroGradient;",
                     "T": f"type fixedValue; value uniform {g(T)};"}[field]
            elif role == "outlet":
                v = {"U": f"type inletOutlet; inletValue uniform (0 0 0); value uniform ({g(U)} 0 0);",
                     "p": f"type fixedValue; value uniform {g(p)};",
                     "T": f"type inletOutlet; inletValue uniform {g(T)}; value uniform {g(T)};"}[field]
            elif role == "wall":
                v = {"U": "type fixedValue; value uniform (0 0 0);" if mrf else "type noSlip;",
                     "p": "type zeroGradient;", "T": "type zeroGradient;"}[field]
            elif role == "lateral":
                v = "type slip;" if field == "U" else "type zeroGradient;"
            elif role == "symmetryPlane":
                v = "type symmetryPlane;"
            else:
                v = "type empty;"
            out.append(f"    {name} {{ {v} }}")
        return "\n".join(out)

    for field, dims, cls, val in (("U", "[0 1 -1 0 0 0 0]", "volVectorField", f"({g(U)} 0 0)"),
                                  ("p", "[1 -1 -2 0 0 0 0]", "volScalarField", g(p)),
                                  ("T", "[0 0 0 1 0 0 0]", "volScalarField", g(T))):
        put(case, f"0/{field}", f"dimensions {dims};\ninternalField uniform {val};\n"
            f"boundaryField\n{{\n    #includeEtc \"caseDicts/setConstraintTypes\"\n{bc(field)}\n}}\n",
            cls)


def read_cell_centres(case):
    """Internal field of 0/C (postProcess -func writeCellCentres), ascii."""
    with open(os.path.join(case, "0/C")) as f:
        txt = f.read()
    i = txt.index("internalField")
    j = txt.index("(", txt.index("\n", i))
    n = int(txt[txt.index("\n", i):j].split()[-1])
    body = txt[j + 1:]
    vals = np.array(body[:body.index("\n)")].replace("(", " ").replace(")", " ").split(),
                    float)
    assert vals.size == 3 * n, (vals.size, n)
    return vals.reshape(n, 3)


def kick_field(C, centre, kw, amp, inside=None):
    """dns' --kick: u += 2 dy/kw e, v -= 2 dx/kw e, e = amp exp(-r^2/kw^2)
    (flow_ic.cpp compressibleBodyIC). C cell centres; fluid cells only."""
    dx, dy = C[:, 0] - centre[0], C[:, 1] - centre[1]
    e = amp * np.exp(-(dx * dx + dy * dy) / (kw * kw))
    u = np.zeros_like(C)
    u[:, 0] = 2 * dy / kw * e
    u[:, 1] = -2 * dx / kw * e
    if inside is not None:
        u[inside] = 0.0
    lines = "\n".join(f"({a:.10g} {b:.10g} 0)" for a, b in u[:, :2])
    return f"nonuniform List<vector>\n{len(u)}\n(\n{lines}\n)"


# --- controlDict --------------------------------------------------------------

STEADY_APPS = ("simpleFoam", "rhoSimpleFoam")    # iteration = "time"


def control(case, app, end, dt0, max_co, write_every, fos, adjustable=True,
            max_dt=None):
    if app in STEADY_APPS:
        head = (f"application {app};\nstartFrom latestTime;\nstartTime 0;\n"
                f"stopAt endTime;\nendTime {end};\ndeltaT 1;\nwriteControl timeStep;\n"
                f"writeInterval {end};\npurgeWrite 1;\n")
    else:
        head = (f"application {app};\nstartFrom latestTime;\nstartTime 0;\n"
                f"stopAt endTime;\nendTime {g(end)};\ndeltaT {g(dt0)};\n"
                f"writeControl adjustableRunTime;\nwriteInterval {g(end)};\npurgeWrite 1;\n"
                f"adjustTimeStep {'yes' if adjustable else 'no'};\nmaxCo {g(max_co)};\n"
                + (f"maxDeltaT {g(max_dt)};\n" if max_dt else ""))
    put(case, "system/controlDict", head + """writeFormat ascii;
writePrecision 12;
timeFormat general;
timePrecision 10;
runTimeModifiable false;
functions
{
""" + fos + "}\n")


def fo_forces(patch, rho, cofr=(0, 0, 0), rho_inf=1.0, p_ref=None):
    """p_ref: subtract a reference pressure (absolute-p solvers) before the
    surface integral; None = the forces default (0)."""
    r = "rho rho;" if rho == "rho" else "rho rhoInf;"
    pr = f" pRef {g(p_ref)};" if p_ref is not None else ""
    return f"""    forces
    {{
        type forces; libs (forces); writeControl timeStep; writeInterval 1;
        patches ({patch}); {r} rhoInf {g(rho_inf)}; CofR {vec(cofr)};{pr} log false;
    }}
"""


def fo_line(name, start, end, n, interval, app):
    ctl = ("writeControl timeStep;" if app in STEADY_APPS
           else "writeControl adjustableRunTime;")
    return f"""    {name}
    {{
        type sets; libs (sampling); {ctl} writeInterval {g(interval)};
        interpolationScheme cellPoint; setFormat raw; fields (U);
        sets ( line {{ type uniform; axis x; start {vec(start)}; end {vec(end)}; nPoints {n}; }} );
    }}
"""
