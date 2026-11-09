#!/usr/bin/env python3
"""Converging-diverging nozzle: one polyMesh + one case directory for dns and rhoCentralFoam (docs/nozzle.md).

  nozzle_mesh.py --contour nc|nc-smooth|a1-smooth|parabola|cone6|rocket|react --nx N --ny N [--beta B]
                 --mode planar|wedge [--dphi DEG] [--pb P] [--exit extrapolate|fixed|riemann] [--pa PA] --out DIR
  react: --eps-c E --eps-e E --lc L --lcon L --ldiv L --scale S (masterplan_combnozzle 2.2). --ny 1: the one-row
  planar half nozzle, the quasi-1D FV scheme (docs/nozzle.md "Quasi-1D").

Writes DIR/constant/polyMesh (the `note` line, else dns refuses the owner file), DIR/bc.toml (astar from the
generator's own throat column), and rhoCentralFoam's 0/, constant/, system/. Wedge: the y = r form, points
(x, r, +-r tan(h)), h = dphi/2, axis nodes merged, prisms on the axis, +-z bit mirrored. Planar: half nozzle, `axis`
a symmetryPlane, `back`, `front` empty. Units: dns p, t, p0, t0 are ratios to the --mach 1 reference, the stagnation
state; foam runs in SI at p0 1e5, T0 300, R 287.1. Library use: tools/fv/nozzle_cases.py.
"""
import argparse
import math
import os

import numpy as np

P0, T0, RGAS, GAMMA = 1e5, 300.0, 287.1, 1.4
# OpenFOAM's RR is not 8314.46261815324 exactly: measured on p/(rho T), once (A3 E1).
RR_FOAM = 8314.46261815324 * (287.1002571930404 / 287.1)


def ncContour(ystar=1.0, yc=2.5, ye=2.0, Ru=2.0, Rd=2.0, thc=30.0, thd=15.0, Ri=2.0, Lc=2.0, smooth=0.0):
    """Wall y(x), x0, x2, x = 0 the throat: chamber, blend arc Ri, cone thc, arcs Ru, Rd, cone thd to ye.
    smooth > 0: curvature 1/Rd to x = smooth, ramped linearly to 0 where the slope reaches tan(thd) (A1's --smooth)."""
    tc, td = math.radians(thc), math.radians(thd)
    x1 = Rd * math.sin(td)
    y1 = ystar + Rd * (1 - math.cos(td))
    x2 = x1 + (ye - y1) / math.tan(td)
    xu1 = -Ru * math.sin(tc)
    yu1 = ystar + Ru * (1 - math.cos(tc))
    yu2 = yc - Ri * (1 - math.cos(tc))
    xu2 = xu1 - (yu2 - yu1) / math.tan(tc)
    xu3 = xu2 - Ri * math.sin(tc)
    x0 = xu3 - Lc

    def y(x):
        x = np.asarray(x, float)
        o = np.empty_like(x)
        m = x <= xu3
        o[m] = yc
        m = (x > xu3) & (x <= xu2)
        o[m] = yc - Ri + np.sqrt(np.maximum(Ri**2 - (x[m] - xu3) ** 2, 0))
        m = (x > xu2) & (x <= xu1)
        o[m] = yu1 + (xu1 - x[m]) * math.tan(tc)
        m = (x > xu1) & (x <= x1)
        R = np.where(x[m] < 0, Ru, Rd)
        o[m] = ystar + R - np.sqrt(np.maximum(R**2 - x[m] ** 2, 0))
        m = x > x1
        o[m] = y1 + (x[m] - x1) * math.tan(td)
        return o

    if smooth <= 0:
        return y, x0, x2
    ta = math.tan(td)
    xk = smooth
    xs = xk + 2 * (ta * Rd - xk)
    assert xs > xk
    xf = np.linspace(0, xs, 20001)
    kap = np.where(xf <= xk, 1 / Rd, (xs - xf) / (xs - xk) / Rd)
    sl = np.concatenate(([0], np.cumsum(0.5 * (kap[1:] + kap[:-1]) * np.diff(xf))))
    yf = ystar + np.concatenate(([0], np.cumsum(0.5 * (sl[1:] + sl[:-1]) * np.diff(xf))))
    xe = xs + (ye - yf[-1]) / ta
    y0 = y

    def ys(x):
        x = np.asarray(x, float)
        o = y0(x)
        o = np.where((x > 0) & (x <= xs), np.interp(x, xf, yf), o)
        return np.where(x > xs, yf[-1] + (x - xs) * ta, o)

    return ys, x0, xe


def contour(a):
    """(y, x0, x2) of the named contour."""
    Rt = a.Rt
    if a.contour == "nc":
        return ncContour(Ru=Rt or 2.0, Rd=Rt or 2.0)
    if a.contour == "nc-smooth":
        return ncContour(Ru=Rt or 2.0, Rd=Rt or 2.0, smooth=0.2)
    if a.contour == "a1-smooth":  # A1's rig: 45 deg, no fillet, contraction 9, the same curvature ramp
        return ncContour(Ru=2.0, Rd=2.0, thc=45.0, yc=3.0, Ri=0.0, smooth=0.2)
    if a.contour == "cone6":
        return ncContour(Rd=2, Ru=2, thd=15, ye=2.449, yc=2.0, Lc=0.5, Ri=1.0)
    if a.contour == "rocket":
        return ncContour(Ru=1.5, Rd=0.382, thc=30, thd=20, ye=2.5, yc=2.5, Lc=1.0, Ri=1.0)
    if a.contour == "parabola":
        R = Rt or 20.0
        xm = 0.2 * R  # |slope| <= 0.2
        return (lambda x: 1 + np.asarray(x, float) ** 2 / (2 * R)), -xm, xm
    if a.contour == "react":
        return reactContour(a.eps_c, a.eps_e, a.lc, a.lcon, a.ldiv, a.scale)
    raise SystemExit("unknown contour " + a.contour)


def reactContour(ec, ee, lc, lcon, ldiv, s=1.0):
    """masterplan_combnozzle 2.2, planar, y* = 1 before the scale s: chamber y = ec on [0, lc], contraction
    1 + (ec - 1)(1 + cos pi xi)/2 over lcon, divergent 1 + (ee - 1)(1 - cos pi xi)/2 over ldiv. C1, exit parallel.
    (y, x0, x2); the throat sits at x = s (lc + lcon) (throatOf)."""
    assert ec >= 1 and ee >= 1 and lc >= 0 and lcon > 0 and ldiv > 0 and s > 0

    def y(x):
        u = np.asarray(x, float) / s
        o = np.full_like(u, ec)
        m = (u > lc) & (u <= lc + lcon)
        o[m] = 1 + (ec - 1) * (1 + np.cos(np.pi * (u[m] - lc) / lcon)) / 2
        m = u > lc + lcon
        o[m] = 1 + (ee - 1) * (1 - np.cos(np.pi * np.minimum(u[m] - lc - lcon, ldiv) / ldiv)) / 2
        return s * o

    return y, 0.0, s * (lc + lcon + ldiv)


def throatOf(a):
    """x of the throat: 0 for every contour but react's."""
    return a.scale * (a.lc + a.lcon) if a.contour == "react" else 0.0


def quadMesh(nx, ny, y, x0, x2, beta=0.0):
    """Nodes (nx + 1, ny + 1): i lines vertical, j lines y = eta y_w(x); beta > 0: tanh clustering to the wall."""
    xs = np.linspace(x0, x2, nx + 1)
    eta = np.linspace(0, 1, ny + 1)
    if beta > 0:
        eta = np.tanh(beta * eta) / math.tanh(beta)
    return np.repeat(xs[:, None], ny + 1, 1), y(xs)[:, None] * eta[None, :]


HDR = """FoamFile
{{
    version 2.0; format ascii; arch "LSB;label=32;scalar=64"; class {cls}; location "constant/polyMesh"; object {obj};{note}
}}
"""


def _list(path, obj, cls, items, note=""):
    with open(path, "w") as f:
        f.write(HDR.format(cls=cls, obj=obj, note=note))
        f.write(f"\n{len(items)}\n(\n")
        f.write("\n".join(items))
        f.write("\n)\n")


def writePolyMesh(case, X, Y, dphi=0.0, depth=0.1):
    """dphi > 0: the wedge. Returns the mesh's A* (the least column area) and its column."""
    nx, ny = X.shape[0] - 1, X.shape[1] - 1
    pm = os.path.join(case, "constant/polyMesh")
    os.makedirs(pm, exist_ok=True)
    wedge = dphi > 0
    N = (nx + 1) * (ny + 1)

    def nid(i, j, k):
        if wedge:
            return i if j == 0 else (nx + 1) + (i + (nx + 1) * (j - 1)) * 2 + k
        return (i + (nx + 1) * j) + k * N

    pts = []
    tanh_ = math.tan(math.radians(dphi) / 2) if wedge else 0.0
    if wedge:
        pts += [(X[i, 0], 0.0, 0.0) for i in range(nx + 1)]
        for j in range(1, ny + 1):
            for i in range(nx + 1):
                z = Y[i, j] * tanh_
                pts += [(X[i, j], Y[i, j], -z), (X[i, j], Y[i, j], z)]
    else:
        for k in range(2):
            pts += [(X[i, j], Y[i, j], k * depth) for j in range(ny + 1) for i in range(nx + 1)]
    _list(pm + "/points", "points", "vectorField", [f"({p[0]:.17g} {p[1]:.17g} {p[2]:.17g})" for p in pts])
    cell = lambda i, j: i + nx * j
    faces, own, nei = [], [], []

    def add(f, o, n=None):
        u = []
        for a in f:
            if a not in u:
                u.append(a)
        if len(u) < 3:
            return
        faces.append("%d(%s)" % (len(u), " ".join(map(str, u))))
        own.append(o)
        if n is not None:
            nei.append(n)

    for j in range(ny):
        for i in range(nx):
            if i < nx - 1:
                add([nid(i + 1, j, 0), nid(i + 1, j + 1, 0), nid(i + 1, j + 1, 1), nid(i + 1, j, 1)], cell(i, j), cell(i + 1, j))
            if j < ny - 1:
                add([nid(i, j + 1, 0), nid(i, j + 1, 1), nid(i + 1, j + 1, 1), nid(i + 1, j + 1, 0)], cell(i, j), cell(i, j + 1))
    nI = len(faces)
    pat = []

    def patch(name, typ, gen):
        st = len(faces)
        for f, o in gen:
            add(f, o)
        pat.append((name, typ, len(faces) - st, st))

    patch("inlet", "patch", [([nid(0, j, 0), nid(0, j, 1), nid(0, j + 1, 1), nid(0, j + 1, 0)], cell(0, j)) for j in range(ny)])
    patch("outlet", "patch", [([nid(nx, j, 0), nid(nx, j + 1, 0), nid(nx, j + 1, 1), nid(nx, j, 1)], cell(nx - 1, j)) for j in range(ny)])
    patch("wall", "wall", [([nid(i, ny, 0), nid(i, ny, 1), nid(i + 1, ny, 1), nid(i + 1, ny, 0)], cell(i, ny - 1)) for i in range(nx)])
    if not wedge:
        patch("axis", "symmetryPlane", [([nid(i, 0, 0), nid(i + 1, 0, 0), nid(i + 1, 0, 1), nid(i, 0, 1)], cell(i, 0)) for i in range(nx)])
    t = "wedge" if wedge else "empty"
    patch("back", t, [([nid(i, j, 0), nid(i, j + 1, 0), nid(i + 1, j + 1, 0), nid(i + 1, j, 0)], cell(i, j)) for j in range(ny) for i in range(nx)])
    patch("front", t, [([nid(i, j, 1), nid(i + 1, j, 1), nid(i + 1, j + 1, 1), nid(i, j + 1, 1)], cell(i, j)) for j in range(ny) for i in range(nx)])
    note = f'\n    note "nPoints:{len(pts)} nCells:{nx * ny} nFaces:{len(faces)} nInternalFaces:{nI}";'
    _list(pm + "/faces", "faces", "faceList", faces)
    _list(pm + "/owner", "owner", "labelList", list(map(str, own)), note)
    _list(pm + "/neighbour", "neighbour", "labelList", list(map(str, nei)), note)
    with open(pm + "/boundary", "w") as f:
        f.write(HDR.format(cls="polyBoundaryMesh", obj="boundary", note=""))
        f.write(f"\n{len(pat)}\n(\n")
        for n, ty, c, s in pat:
            f.write(f"{n}\n{{\n type {ty};\n nFaces {c};\n startFace {s};\n}}\n")
        f.write(")\n")
    # A planar column is a rectangle, a wedge column the triangle r^2 tan(h) of its own wall node.
    area = Y[:, ny] ** 2 * tanh_ if wedge else Y[:, ny] * depth
    i = int(np.argmin(area))
    return float(area[i]), i


def writeBc(case, astar, wedge, pb, exitKind, wall, twall, area, u_force=True, pa=None):
    """DIR/bc.toml. pb: the back pressure over p0 (also the rest state of [nozzle]); exitKind: extrapolate writes no p.
    pa: [thrust] pa (the ambient over p0, docs/nozzle.md "Thrust"), None: no [thrust] table."""
    side = '[patch.back]\ntype = "wedge"\n\n[patch.front]\ntype = "wedge"\n\n' if wedge else ""
    axis = "" if wedge else '[patch.axis]\ntype = "slip"\n\n'
    if wall == "slip":
        w = '[patch.wall]\ntype = "slip"\nforce = true\n'
    elif twall is None:
        w = '[patch.wall]\ntype = "wall"\n'
    else:
        w = f'[patch.wall]\ntype = "wall"\nthermal = "isothermal"\ntwall = {twall:.17g}\n'
    if exitKind == "extrapolate":
        o = '[patch.outlet]\ntype = "outflow"\nexit = "extrapolate"\n'
    else:
        o = f'[patch.outlet]\ntype = "outflow"\nexit = "{exitKind}"\np = {pb:.17g}\ntback = 1\n'
    with open(os.path.join(case, "bc.toml"), "w") as f:
        f.write(f"# tools/fv/nozzle_mesh.py: p, t ratios to the --mach 1 reference (the stagnation state)\n")
        f.write(f"[reference]\nlength = 1\narea = {area:.17g}\ndirection = [1, 0, 0]\nlift = [0, 1, 0]\n\n")
        f.write(f"[nozzle]\np = {pb:.17g}\nt = 1\nastar = {astar:.17g}\n\n")
        f.write(f'[patch.inlet]\ntype = "total"\np0 = 1\nt0 = 1\n\n{o}\n{w}\n{axis}{side}')
        if pa is not None:
            f.write(f"\n[thrust]\npa = {pa:.17g}\n")


FH = "FoamFile {{ version 2.0; format ascii; class {cls}; location \"{loc}\"; object {obj}; }}\n"


def _scalars(v):
    return "nonuniform List<scalar>\n%d\n(\n%s\n)" % (len(v), "\n".join("%.17g" % a for a in v))


def _vectors(u, v):
    return "nonuniform List<vector>\n%d\n(\n%s\n)" % (len(u), "\n".join("(%.17g %.17g 0)" % (a, b) for a, b in zip(u, v)))


def writeFoamCase(case, ic, pback=None, recon="upwind", endTime=1.2, maxCo=0.2, wedge=False, mu=0.0, Pr=0.71, wall="slip",
                  Twall=None, write=None, dt=1e-7):
    """rhoCentralFoam's 0/, constant/, system/. pback: fixed exit pressure [Pa], else zeroGradient. ic: cell-order p, T, u, v."""
    for d in ("0", "system", "constant"):
        os.makedirs(os.path.join(case, d), exist_ok=True)
    axis = "" if wedge else "axis { type symmetryPlane; }\n"
    st = "wedge" if wedge else "empty"
    sides = f"back {{ type {st}; }}\nfront {{ type {st}; }}\n"
    wall_U = "wall { type slip; }" if wall == "slip" else "wall { type noSlip; }"
    wall_T = "wall { type zeroGradient; }" if Twall is None else f"wall {{ type fixedValue; value uniform {Twall}; }}"
    out_p = "outlet { type zeroGradient; }" if pback is None else f"outlet {{ type fixedValue; value uniform {pback}; }}"
    with open(case + "/0/p", "w") as f:
        f.write(FH.format(cls="volScalarField", loc="0", obj="p"))
        f.write(f"dimensions [1 -1 -2 0 0 0 0];\ninternalField {_scalars(ic['p'])};\nboundaryField\n{{\n"
                f"inlet {{ type totalPressure; p0 uniform {P0}; gamma {GAMMA}; psi thermo:psi; rho none; value uniform {P0}; }}\n{out_p}\n"
                f"wall {{ type zeroGradient; }}\n{axis}{sides}}}\n")
    with open(case + "/0/T", "w") as f:
        f.write(FH.format(cls="volScalarField", loc="0", obj="T"))
        f.write(f"dimensions [0 0 0 1 0 0 0];\ninternalField {_scalars(ic['T'])};\nboundaryField\n{{\n"
                f"inlet {{ type totalTemperature; T0 uniform {T0}; gamma {GAMMA}; psi thermo:psi; value uniform {T0}; }}\noutlet {{ type zeroGradient; }}\n"
                f"{wall_T}\n{axis}{sides}}}\n")
    with open(case + "/0/U", "w") as f:
        f.write(FH.format(cls="volVectorField", loc="0", obj="U"))
        f.write(f"dimensions [0 1 -1 0 0 0 0];\ninternalField {_vectors(ic['u'], ic['v'])};\nboundaryField\n{{\n"
                f"inlet {{ type pressureInletOutletVelocity; value uniform (0 0 0); }}\noutlet {{ type zeroGradient; }}\n"
                f"{wall_U}\n{axis}{sides}}}\n")
    W = RR_FOAM / RGAS
    Cp = GAMMA / (GAMMA - 1) * RGAS
    with open(case + "/constant/thermophysicalProperties", "w") as f:
        f.write(FH.format(cls="dictionary", loc="constant", obj="thermophysicalProperties"))
        f.write("thermoType { type hePsiThermo; mixture pureMixture; transport const; thermo hConst; equationOfState perfectGas; specie specie; energy sensibleInternalEnergy; }\n")
        f.write(f"mixture {{ specie {{ molWeight {W:.17g}; }} thermodynamics {{ Cp {Cp:.17g}; Hf 0; }} transport {{ mu {mu}; Pr {Pr}; }} }}\n")
    with open(case + "/constant/turbulenceProperties", "w") as f:
        f.write(FH.format(cls="dictionary", loc="constant", obj="turbulenceProperties"))
        f.write("simulationType laminar;\n")
    rec = {"upwind": ("upwind", "upwind", "upwind"), "vanLeer": ("vanLeer", "vanLeerV", "vanLeer"), "Minmod": ("Minmod", "MinmodV", "Minmod")}[recon]
    with open(case + "/system/fvSchemes", "w") as f:
        f.write(FH.format(cls="dictionary", loc="system", obj="fvSchemes"))
        f.write(f"""fluxScheme Kurganov;
ddtSchemes {{ default Euler; }}
gradSchemes {{ default Gauss linear; }}
divSchemes {{ default none; div(tauMC) Gauss linear; }}
laplacianSchemes {{ default Gauss linear corrected; }}
interpolationSchemes {{ default linear; reconstruct(rho) {rec[0]}; reconstruct(U) {rec[1]}; reconstruct(T) {rec[2]}; }}
snGradSchemes {{ default corrected; }}
""")
    with open(case + "/system/fvSolution", "w") as f:
        f.write(FH.format(cls="dictionary", loc="system", obj="fvSolution"))
        f.write('solvers { "(rho|rhoU|rhoE)" { solver diagonal; } U { solver smoothSolver; smoother GaussSeidel; nSweeps 2; tolerance 1e-12; relTol 0; } e { solver smoothSolver; smoother GaussSeidel; nSweeps 2; tolerance 1e-12; relTol 0; } }\n')
    with open(case + "/system/controlDict", "w") as f:
        f.write(FH.format(cls="dictionary", loc="system", obj="controlDict"))
        f.write(f"""application rhoCentralFoam;
startFrom startTime; startTime 0; stopAt endTime; endTime {endTime}; deltaT {dt}; writeControl adjustableRunTime; writeInterval {write or endTime};
purgeWrite 4; writeFormat ascii; writePrecision 17; writeCompression off; timeFormat general; timePrecision 12; runTimeModifiable false;
adjustTimeStep yes; maxCo {maxCo}; maxDeltaT 1;
""")


def areaMach(ar, sup=True, g=GAMMA):
    """M of A/A* >= 1."""
    f = lambda M: (1 / M) * ((2 / (g + 1)) * (1 + (g - 1) / 2 * M * M)) ** ((g + 1) / (2 * (g - 1))) - ar
    lo, hi = (1.0, 60.0) if sup else (1e-6, 1.0)
    for _ in range(200):
        m = 0.5 * (lo + hi)
        if (f(m) > 0) != sup:
            lo = m
        else:
            hi = m
    return 0.5 * (lo + hi)


def quasi1dIc(X, Y, delta, pback=None, p0=P0, T0_=T0, R=RGAS, g=GAMMA, xt=0.0):
    """Cell-order isentropic state, supersonic past the throat at x = xt; pback [Pa]: the normal-shock branch. delta 1:
    axisymmetric."""
    nx, ny = X.shape[0] - 1, X.shape[1] - 1
    xm = 0.5 * (X[:-1, 0] + X[1:, 0])
    ym = 0.5 * (Y[:-1, -1] + Y[1:, -1])
    yc = 0.25 * (Y[:-1, :-1] + Y[1:, :-1] + Y[:-1, 1:] + Y[1:, 1:])
    ar = (ym / Y[:, -1].min()) ** (1 + delta)
    M = np.array([areaMach(max(a, 1 + 1e-9), xx > xt, g) for a, xx in zip(ar, xm)])
    pr = (1 + (g - 1) / 2 * M * M) ** (-g / (g - 1))
    Tr = 1 / (1 + (g - 1) / 2 * M * M)
    if pback is not None:
        ae = (ym[-1] / Y[:, -1].min()) ** (1 + delta)

        def peOf(As):
            Ms = areaMach(As, True, g)
            p02 = ((g + 1) * Ms * Ms / ((g - 1) * Ms * Ms + 2)) ** (g / (g - 1)) * ((g + 1) / (2 * g * Ms * Ms - (g - 1))) ** (1 / (g - 1))
            Me = areaMach(ae * p02, False, g)
            return p02 * (1 + (g - 1) / 2 * Me * Me) ** (-g / (g - 1)), Ms, p02

        lo, hi = 1.0001, ae
        for _ in range(100):
            m = 0.5 * (lo + hi)
            if peOf(m)[0] > pback / p0:
                lo = m
            else:
                hi = m
        As = 0.5 * (lo + hi)
        p02 = peOf(As)[2]
        xs = next(i for i in range(nx) if xm[i] > xt and ar[i] >= As)
        for i in range(xs, nx):
            Mi = areaMach(max(ar[i] * p02, 1 + 1e-9), False, g)
            M[i], pr[i], Tr[i] = Mi, p02 * (1 + (g - 1) / 2 * Mi * Mi) ** (-g / (g - 1)), 1 / (1 + (g - 1) / 2 * Mi * Mi)
    p = (pr * p0)[:, None] * np.ones(ny)
    T = (Tr * T0_)[:, None] * np.ones(ny)
    U = M[:, None] * np.sqrt(g * R * T)
    slope = np.gradient(ym, xm)
    fl = lambda a: a.T.ravel().tolist()
    return dict(p=fl(p), T=fl(T), u=fl(U), v=fl(U * slope[:, None] * (yc / ym[:, None])), xm=xm, M=M, pr=pr)


def generate(a):
    """The whole case. a: parsed arguments (nozzle_cases.py builds them too)."""
    global GAMMA
    GAMMA = a.gamma
    y, x0, x2 = contour(a)
    X, Y = quadMesh(a.nx, a.ny, y, x0, x2, a.beta)
    wedge = a.mode == "wedge"
    astar, col = writePolyMesh(a.out, X, Y, a.dphi if wedge else 0.0)
    pb = a.pb if a.pb > 0 else 0.0
    exitKind = a.exit
    writeBc(a.out, astar, wedge, pb if pb > 0 else 0.1, exitKind, a.wall, a.twall, astar, pa=a.pa)
    ic = quasi1dIc(X, Y, 1 if wedge else 0, pb * P0 if (pb > 0 and exitKind == "fixed") else None, g=a.gamma, xt=throatOf(a))
    Tw = a.twall * T0 if a.twall is not None else None
    writeFoamCase(a.out, ic, pback=pb * P0 if exitKind == "fixed" else None, recon=a.recon, endTime=a.end, maxCo=a.co, wedge=wedge,
                  mu=a.mu, Pr=a.Pr, wall=a.wall, Twall=Tw, write=a.write)
    np.savez(a.out + "/mesh.npz", X=X, Y=Y, astar=astar, col=col)
    return astar, col, X, Y


def parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--contour", default="nc")
    p.add_argument("--nx", type=int, required=True)
    p.add_argument("--ny", type=int, required=True)
    p.add_argument("--beta", type=float, default=0.0, help="tanh clustering to the wall")
    p.add_argument("--mode", default="planar", choices=["planar", "wedge"])
    p.add_argument("--dphi", type=float, default=20.0, help="wedge angle, degrees")
    p.add_argument("--Rt", type=float, default=0.0, help="throat arc radius (0: the contour's)")
    p.add_argument("--pb", type=float, default=0.0, help="back pressure / p0")
    p.add_argument("--exit", default="extrapolate", choices=["extrapolate", "fixed", "riemann"])
    p.add_argument("--wall", default="slip", choices=["slip", "noslip"])
    p.add_argument("--twall", type=float, default=None, help="isothermal wall T / T0")
    p.add_argument("--mu", type=float, default=0.0, help="foam mu, SI")
    p.add_argument("--Pr", type=float, default=0.71)
    p.add_argument("--recon", default="upwind", choices=["upwind", "vanLeer", "Minmod"])
    p.add_argument("--end", type=float, default=1.2)
    p.add_argument("--write", type=float, default=None)
    p.add_argument("--co", type=float, default=0.2, help="foam maxCo")
    p.add_argument("--gamma", type=float, default=1.4)
    p.add_argument("--pa", type=float, default=None, help="[thrust] pa / p0 (absent: no [thrust] table)")
    # --contour react (masterplan_combnozzle 2.2): chamber, cosine contraction, cosine divergent.
    p.add_argument("--eps-c", type=float, default=6.25, help="react: chamber area / A*")
    p.add_argument("--eps-e", type=float, default=6.25, help="react: exit area / A*")
    p.add_argument("--lc", type=float, default=0.5, help="react: chamber length / y*")
    p.add_argument("--lcon", type=float, default=4.0, help="react: contraction length / y*")
    p.add_argument("--ldiv", type=float, default=12.0, help="react: divergent length / y*")
    p.add_argument("--scale", type=float, default=1.0, help="react: y* (the mesh's length unit)")
    p.add_argument("--out", required=True)
    return p


def main():
    a = parser().parse_args()
    astar, col, X, Y = generate(a)
    print(f"{a.contour} {a.mode}: {a.nx}x{a.ny}, x in [{X[0, 0]:.5f}, {X[-1, 0]:.5f}], A* {astar:.12g} at column {col}, A_e/A* "
          f"{(Y[-1, -1] / Y[:, -1].min()) ** (2 if a.mode == 'wedge' else 1):.5f}, throat x {throatOf(a):.6g} -> {a.out}")


if __name__ == "__main__":
    main()
