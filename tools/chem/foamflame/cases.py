"""Case specs of the dns-vs-reactingFoam flame comparison: ONE dict per case
-> the dns argv, the OpenFOAM case directory and the simjob job specs.

A case is a dict (CASES below). Keys every case has:
  geometry   "1d" (one row of cells), "bunsen2d" (--mode bunsen), "wrinkle2d"
             (planar flame + --wrinkle modes, --ny periodic); 2D cases are
             measured by observe2d.py and run decomposed (foam "np")
  model      sigma, beta, Pr, mach (the dns flags of the same name)
  species    OpenFOAM species, solved first, inert LAST; reactions
             [(equation, order)] -- A, Ta come from the SI map
  ladder     dns --cells-per-delta values (the foam mesh is the same cells)
  dns        extra dns argv (protocol flags)
  foam       time step in tau, PIMPLE correctors, variants
             {name: {yscheme, hscheme, ...overrides}}
  window     averaging window in tau
A geometry provides five functions (GEOMETRY): the dns grid (`mesh_1d`:
cells, h, x centres, the ref column), the IC in SI (`ic_1d`), the patch
conditions (`foam_bcs_1d`), the blockMeshDict (`block_1d`) and the probe
points (`probes_1d`). Add a case = add a dict; a new geometry = those five
(observe.py's measure() is the 1D measure; a 2D case adds its own).
"""
import math

import numpy as np

from . import writer
from .model import GAMMA

CASES = {
    "G1": {
        "title": "1D laminar premixed flame, sliding (docs/chem.md Measured)",
        "geometry": "1d",
        "model": {"sigma": 5.0, "beta": 8.0, "Pr": 0.72, "mach": 0.01},
        "species": ["F", "P"],
        "reactions": ["F = P"],
        # dns: the docs/chem.md "Measured" protocol, its default box
        # (12 delta_th fresh + 10 burnt + 16 sink cells per open face)
        "box": {"upstream": 12, "downstream": 10, "sink": 16},
        "inflow": 0.98,
        "tt": 20.0,
        "window": (10.0, 20.0),
        "ladder": [7.5, 15, 30],
        # dns profile cadence: 0.1 tau averages the box acoustics (period
        # ~0.8-1.7 tau, ~1% of dp at the ref column) out of the pressure
        # jump; foam has none (implicit, damped), its fields every 0.5 tau
        "profile_dt": 0.1,
        "foam_write_dt": 0.5,
        "dns": [],
        "foam": {
            "dt": 1e-3,          # tau; Co <= 0.1 on every rung (u_b = 4.9 S_L)
            "n_outer": 3, "n_corr": 2,
            "variants": {
                # dns-like: MC-limited MUSCL on Y, central h (dns: central E)
                "mclin": {"yscheme": "MUSCL", "hscheme": "linear"},
                # both on the MC limiter
                "mcmc": {"yscheme": "MUSCL", "hscheme": "MUSCL"},
            },
            # time-step / iteration checks (one rung)
            "checks": {
                "mclin_dt2": {"base": "mclin", "N": 15, "dt": 5e-4},
                "mclin_o6": {"base": "mclin", "N": 15, "n_outer": 6},
                # stock reactingFoam species diffusion (rho D = mu: Sc = 1,
                # Le = 1/Pr = 1.39): what the unityLewis fvOption buys
                "mclin_sc1": {"base": "mclin", "N": 15, "le1": False},
            },
        },
        "si": {"Tu": 300.0, "p0": 101325.0, "W": 28.014, "dD": 1e-4},
    },
    "G2a": {
        "title": "2D slot Bunsen, burnt coflow (docs/chem.md 2D slot Bunsen)",
        "geometry": "bunsen2d",
        "model": {"sigma": 5.0, "beta": 8.0, "Pr": 0.72, "mach": 0.02},
        "species": ["F", "P"],
        "reactions": ["F = P"],
        # dns defaults: W 12 delta_th, U_jet 5, burnt coflow (T/Tu = sigma)
        # at 1 S_L, coflow 12 delta_th per side, edge 0.25 delta_th, length
        # ceil(1.25 cone + 8) = 45 delta_th past the 16-cell bands
        "burner": {"slot": 12, "jet": 5},
        "box": {"sink": 16},
        "tt": 30.0,                 # dns (job 659); foam stops at 25 (variant)
        # the cone rings with a ~10 tau convective period (dns tt 30: the
        # tip 38 -> 45 -> 44.8); [20, 25]: dns means within 1.5e-3 of [20,
        # 30] (xTip 1.7e-3 relative); [15, 20] is 8.7e-3 off in sinRatio
        "window": (20.0, 25.0),
        "ladder": [8],
        "fdt": 1.0,
        "foam_write_dt": 1.0,
        "write_precision": 10,
        "dns": [],
        # foam cost (scout jobs 660, 661, 113k cells): nOuter 3 at dt 2e-3
        # tau = 0.135 s/step on 14 ranks (25 tau = 1690 s, over budget);
        # nOuter 1 (PISO, 2 pressure correctors) at dt 4e-3 tau = 0.042 s/step
        # on 12 ranks (8 ranks 0.047; the hybrid i5 does not scale past ~8).
        # The steady cone is the fixed point of either; the chemistry source
        # lags one step (dt k_chem ~ 0.7 at the peak rate, stable)
        "foam": {
            "dt": 4e-3, "n_outer": 1, "n_corr": 2, "np": 12,
            "variants": {"mclin": {"yscheme": "MUSCL", "hscheme": "linear",
                                   "tt": 25.0}},
        },
        "si": {"Tu": 300.0, "p0": 101325.0, "W": 28.014, "dD": 1e-4},
    },
    "G2b": {
        "title": "Darrieus-Landau growth of one wrinkle mode, sigma 5",
        "geometry": "wrinkle2d",
        "model": {"sigma": 5.0, "beta": 8.0, "Pr": 0.72, "mach": 0.02},
        "species": ["F", "P"],
        "reactions": ["F = P"],
        # the lesgap dispersion box (docs/chem.md "Why TFLES misses"):
        # upstream / downstream 16, stationary inflow, M 0.02
        "box": {"upstream": 16, "downstream": 16, "sink": 16},
        "inflow": 1.0,
        # lambda = 30 delta_th = 48.05 delta_D (2.1 lambda_c of Matalon-
        # Matkowsky, ~2.3 of the lesgap scan's ~21). 8 cells: the
        # 6-cell run (job 663/665) wobbles in its local rate (0.04-0.12 over
        # t 0-16, pinning/IC transient) before it settles; 8 cells is a clean
        # exponential from t ~ 8 (local rate 0.085-0.089 per 2 tau to t 32)
        "ny": 240,
        "wrinkle": 0.02, "wrinkle_modes": 1,
        "tt": 32.0,                 # dns (job 665); foam stops at 20 (variant)
        "fit": (8.0, 20.0),
        "window": (8.0, 20.0),
        "ladder": [8],
        # dns-only runs read by compare: (name, dir, cells, ny, sigma,
        # delta_th/delta_D) -- the 6-cell twin (job 665) and sigma 2 at
        # lambda = 36 delta_th = 64.6 delta_D ~ 2 lambda_c (job 666)
        "extra_dns": [("dns_c6_tt32", "diag_dns_c6_tt32", 6, 180, 5.0, 1.6016509),
                      ("dns_sigma2_c8", "dns_s2_c8", 8, 288, 2.0, 1.7953)],
        "fdt": 0.5,
        "foam_write_dt": 0.5,
        "write_precision": 10,
        "dns": [],
        "foam": {
            "dt": 4e-3, "n_outer": 1, "n_corr": 2, "np": 8,
            "variants": {"mclin": {"yscheme": "MUSCL", "hscheme": "linear",
                                   "tt": 20.0}},
        },
        "si": {"Tu": 300.0, "p0": 101325.0, "W": 28.014, "dD": 1e-4},
    },
    "G3a": {
        "title": "2D slot-jet diffusion flame, dns defaults (docs/chem.md "
                 "2D slot-jet diffusion flame)",
        "geometry": "diffusion2d",
        # model C1 at the --mode diffusion defaults (dns/config.cpp)
        "model": {"s": 1.0, "yF1": 1.0, "yO2": 1.0, "sigma": 5.0, "beta": 5.0,
                  "Da": 1000.0, "Pr": 0.72, "mach": 0.05},
        # foam species: the two reactants, a PASSIVE mixture-fraction tracer
        # Zt (no reaction, Le 1, inlet Z(y): the carried check of Z(Y_F,
        # Y_O)), the products/inert LAST
        "species": ["F", "O", "Zt", "P"],
        # defaults: W 16 l, coflow 16 l per side at 1 U, edge 3 l, length
        # ceil(min(1.25 x 70.34 + 8, 320)) = 96 l past the 16-cell bands
        "burner": {"slot": 16.0, "coflow_width": 16.0, "coflow_speed": 1.0,
                   "edge": 3.0},
        "box": {"sink": 16},
        "ladder": [4],              # --cells-per-unit
        # dns tt 80 = gpu_diffusion_jet (its window [60, 80] printed too);
        # the comparison window [60, 75]: foam stops at 75 (CPU budget). The
        # IC's extruded sheet flushes by t ~ 27 (first closed tip), the box
        # mode (pMean +-3% at t 30, period ~12 l/U) decays to +-0.3% by 60
        "tt": 80.0,
        "window": (60.0, 75.0),
        "fdt": 1.0,
        "foam_write_dt": 1.0,
        "write_precision": 10,
        "dns": [],
        # dns-only twins read by compare (label, run dir; the case argv + the
        # flag): the inflow relax stiffness (the dns-foam fuel-flux gap is
        # dns' inflow overshoot at the default etaIn 100, issues.md) and the
        # temperature-held inflow entropy target (G2a's §66)
        "extra_dns": [("dns etaIn 1000", "dns_c4_etain1000", "--nscbc-etain 1000"),
                      ("dns etaIn 20", "dns_c4_etain20", "--nscbc-etain 20"),
                      ("dns tinflow", "dns_c4_tinflow", "--nscbc-tinflow")],
        # foam cost (scout job 670, 80k cells, 8 ranks): 0.056 s/step, PISO
        # nOuter 1 / nCorr 2. dt 0.025 l/U: Co max 0.52, and dt x the
        # chemistry rate J (dns' dtChem = 0.5/J = 0.020) = 0.63, the G2
        # regime (0.7); the explicit (lagged) source is checked by dt/2.
        # Both variants run CONCURRENTLY, 7 ranks each
        "foam": {
            "dt": 0.025, "n_outer": 1, "n_corr": 2, "np": 7,
            "variants": {
                # dns' central E, but each species its OWN MC limiter
                "mclin": {"yscheme": "MUSCL", "hscheme": "linear", "tt": 75.0},
                # ONE limiter for every field (min over the voters F, O; the
                # tracer, the inert and h ride it without voting: blended 2)
                "mvsel": {"yscheme": "MUSCL", "hscheme": "linear", "tt": 75.0,
                          "mv": "multivariateSelection",
                          "per": {"F": "MUSCL", "O": "MUSCL", "Zt": "blended 2",
                                  "P": "blended 2", "h": "blended 2"}},
            },
            # time-step / corrector checks, restarted from the base run's
            # t = 60 field, 8 l/U, compared over [64, 68] with the base
            "checks": {
                "mclin_dt2": {"base": "mclin", "N": 4, "dt": 0.0125,
                              "restart": 60.0, "tt": 68.0},
                "mclin_o3": {"base": "mclin", "N": 4, "n_outer": 3,
                             "restart": 60.0, "tt": 68.0},
            },
        },
        "si": {"T0": 300.0, "p0": 101325.0, "W": 28.014, "l": 1e-3},
    },
    "G3b": {
        "title": "2D candle (dns flame --mode candle defaults, 4 cells/l)",
        "geometry": "candle2d",
        # the candle defaults (dns/config.cpp): paraffin/air s 3.4, Y_O2
        # 0.233, sigma 5, beta 5, Da 5000, M 0.005
        "model": {"s": 3.4, "yF1": 1.0, "yO2": 0.233, "sigma": 5.0, "beta": 5.0,
                  "Da": 5000.0, "Pr": 0.72, "mach": 0.005},
        "species": ["F", "O", "Zt", "P"],
        # W 1, coflow 1 U x 12 l per side, edge 0.5, length 32, ignition 4,
        # g 14 U^2/l along -x
        "burner": {"slot": 1.0, "coflow_width": 12.0, "coflow_speed": 1.0,
                   "edge": 0.5, "length": 32.0, "ignition": 4.0,
                   "gravity": 14.0},
        "box": {"sink": 16},
        # 4 cells/l, NOT the default 8: foam at 8 (57.6k cells, the plume's
        # 34 U at Co 0.5: dt 1.8e-3 l/U, 16.5k steps) is ~12 min of CPU;
        # both solvers on the same coarse mesh
        "ladder": [4],
        "tt": 30.0,                 # docs/chem.md: measure in t in [20, 30]
        "window": (20.0, 30.0),
        "fdt": 0.5,
        "foam_write_dt": 0.5,
        "write_precision": 10,
        "dns": [],
        "foam": {
            # dns dtChem = 0.5/J = 1.8e-3 at 4 cells (job 673): dt 2e-3 l/U
            # puts J dt at 0.55 (G3a: 0.63), Co 0.31 at the plume's 39 U
            "dt": 2e-3, "n_outer": 1, "n_corr": 2, "np": 7,
            "variants": {"mvsel": {"yscheme": "MUSCL", "hscheme": "linear",
                                   "mv": "multivariateSelection",
                                   "per": {"F": "MUSCL", "O": "MUSCL",
                                           "Zt": "blended 2", "P": "blended 2",
                                           "h": "blended 2"}}},
        },
        "si": {"T0": 300.0, "p0": 101325.0, "W": 28.014, "l": 1e-3},
    },
    "G3b_rest": {
        "title": "2D candle box at rest (--mode candle --quiescent, 4 cells/l): "
                 "the hydrostatic boundary twin",
        "geometry": "candle2d",
        "model": {"s": 3.4, "yF1": 1.0, "yO2": 0.233, "sigma": 5.0, "beta": 5.0,
                  "Da": 5000.0, "Pr": 0.72, "mach": 0.005},
        "species": ["F", "O", "Zt", "P"],
        # the candle geometry, every inflow row air at rest (u 0, T0)
        "burner": {"slot": 1.0, "coflow_width": 12.0, "coflow_speed": 1.0,
                   "edge": 0.5, "length": 32.0, "ignition": 4.0,
                   "gravity": 14.0, "quiescent": True},
        "box": {"sink": 16},
        "ladder": [4],
        "tt": 10.0,                 # gpu_candle_rest: tt 10 (37 free-fall times)
        "window": (0.0, 10.0),
        "fdt": 0.5,
        "foam_write_dt": 0.5,
        "write_precision": 12,
        "dns": [],
        "foam": {
            # no flow, no chemistry: dt bounded by the acoustic-free PISO
            # only; 0.02 l/U = 500 steps
            "dt": 0.02, "n_outer": 1, "n_corr": 2, "np": 7,
            "variants": {"mvsel": {"yscheme": "MUSCL", "hscheme": "linear",
                                   "mv": "multivariateSelection",
                                   "per": {"F": "MUSCL", "O": "MUSCL",
                                           "Zt": "blended 2", "P": "blended 2",
                                           "h": "blended 2"}}},
        },
        "si": {"T0": 300.0, "p0": 101325.0, "W": 28.014, "l": 1e-3},
    },
}

# the two-reactant geometries (model C1: no premixed oracle, si_map_diff)
DIFF_GEOMS = ("diffusion2d", "candle2d")

# The tracer specie carries ZT_EPS Z: reactingFoam's multiComponentMixture
# divides every Y by sum Y at construction (correctMassFractions), and the
# pure fuel stream (Y_F1 = 1) has no inert to displace -- a full-amplitude
# Zt halved the fuel core (scout, measured). At 1e-3 the construction
# rescales the IC's fuel cells by 1/(1 + 1e-3) once (flushed in one
# flow-through); afterwards sum Y <= 1 + 1e-3, the inert clamps at 0 there
# and the mixture properties (normalised averages of identical species) do
# not see it. Read back as Zt / ZT_EPS.
ZT_EPS = 1e-3


# --- geometry: 1d -------------------------------------------------------------

def mesh_1d(spec, N, dth):
    """dns' 1D flame grid (dns/config.cpp): h = delta_th/N, nx = ceil((up +
    down) N) + 2 sink, x_f0 = sink h + up delta_th, the stats' fresh
    reference column iRef = sink + round(2 delta_th/h)."""
    b = spec["box"]
    h = dth / N
    nb = b["sink"]
    nx = int(math.ceil((b["upstream"] + b["downstream"]) * N - 1e-9)) + 2 * nb
    return {"nx": nx, "h": h, "x": (np.arange(nx) + 0.5) * h,
            "xf0": nb * h + b["upstream"] * dth,
            "iRef": min(nx - 1, nb + int(round(2 * dth / h))), "nb": nb}


def ic_1d(spec, N, oracle, si):
    """dns premixedState (dns/flame1d.cpp) of the oracle at x - x_f0, in SI:
    rho0 = 1/t, p = p0 - m^2 (1/rho0 - 1), rho = p/T, u = m/rho (box)."""
    m = mesh_1d(spec, N, oracle.dth)
    t = oracle.profile(m["x"] - m["xf0"])["T"]
    Tu = si["TuBox"]
    mf = spec["inflow"]
    p = Tu - mf * mf * (t - 1.0)
    rho = p / (t * Tu)
    u = mf / rho
    Y = (oracle.sigma - t) / (oracle.sigma - 1)
    U = np.zeros((m["nx"], 3))
    U[:, 0] = u * si["SL"]
    sp = spec["species"]
    return {"T": t * si["Tu"], "p": p * si["pUnit"], "U": U,
            sp[0]: Y, sp[1]: 1.0 - Y}


def foam_bcs_1d(spec, oracle, si, m):
    """inlet: fixed U (u_in), T (Tu), Y (fresh), zeroGradient p.
    outlet: waveTransmissive p -> p_b (dns' outflow target p0 - u_in^2
    (sigma - 1)), lInf = box length; zeroGradient U, T, Y."""
    sp = spec["species"]
    mf = spec["inflow"]
    pB = (si["TuBox"] - mf * mf * (oracle.sigma - 1)) * si["pUnit"]
    L = m["nx"] * m["h"] * si["dD"]
    zg = writer.bc("zeroGradient")
    inlet = {"U": writer.bc("fixedValue", (mf * si["SL"], 0, 0)),
             "T": writer.bc("fixedValue", si["Tu"]),
             "p": zg, sp[0]: writer.bc("fixedValue", 1.0),
             sp[1]: writer.bc("fixedValue", 0.0)}
    outlet = {"U": zg, "T": zg, sp[0]: zg, sp[1]: zg,
              "p": writer.bc("waveTransmissive", gamma=GAMMA, pInf=pB, lInf=L)}
    return {"inlet": inlet, "outlet": outlet}


def block_1d(case, m, si):
    writer.block_mesh_1d(case, m["nx"], m["h"] * si["dD"])


def probes_1d(m, si):
    """The fresh reference column's cell centre [m]."""
    hz = m["h"] * si["dD"]
    return [((m["iRef"] + 0.5) * hz, 0.5 * hz, 0.5 * hz)]


# --- geometry: bunsen2d (dns flame --mode bunsen, nz 1) -------------------------

def bunsen_geom(spec, dth):
    """dns/config.cpp bunsenGeometry() with the case's burner flags (lengths
    delta_th in the spec, box units out): W, uJet, uCo, wCo, tCo, pilot 0,
    edge, yc = half h, coneH; the box nx x ny."""
    b = spec["burner"]
    N = spec["N"]
    h = dth / N
    W, U = b["slot"], b["jet"]
    wco = b.get("coflow_width", W)
    coneH = 0.5 * W * math.sqrt(U * U - 1.0)
    length = b.get("length", math.ceil(1.25 * coneH + 8))
    half = int(math.ceil((0.5 * W + wco) * N - 1e-9))
    nb = spec["box"]["sink"]
    return {"W": W * dth, "uJet": U, "uCo": b.get("coflow_speed", 1.0),
            "tCo": b.get("coflow_temp", spec["model"]["sigma"]),
            "edge": b.get("edge", 0.25) * dth, "yc": half * h,
            "coneH": coneH * dth, "nx": int(math.ceil(length * N - 1e-9)) + 2 * nb,
            "ny": 2 * half, "h": h, "nb": nb}


def bunsen_inflow(b, y):
    """dns::bunsenInflow (dns/bunsen.cpp), no pilot strip: (u, t, Y, jet)."""
    r = np.abs(y - b["yc"])
    jet = 0.5 * (1.0 - np.tanh((r - 0.5 * b["W"]) / b["edge"]))
    co = 0.5 * (1.0 + np.tanh((r - 0.5 * b["W"]) / b["edge"]))
    return b["uJet"] * jet + b["uCo"] * co, jet + b["tCo"] * co, jet, jet


def bunsen_cone_distance(b, x, y):
    """dns::bunsenConeDistance: signed distance to the ideal cone."""
    r = np.abs(y - b["yc"])
    ax, ar = b["coneH"], -0.5 * b["W"]
    L2 = ax * ax + ar * ar
    s = np.clip((x * ax + (r - 0.5 * b["W"]) * ar) / L2, 0.0, 1.0)
    d = np.hypot(x - s * ax, r - (0.5 * b["W"] + s * ar))
    inside = (x >= 0) & (x <= b["coneH"]) & (r < 0.5 * b["W"] * (1 - x / b["coneH"]))
    return np.where(inside, -d, d)


def mesh_bunsen2d(spec, N, dth):
    b = bunsen_geom({**spec, "N": N}, dth)
    nx, ny, h = b["nx"], b["ny"], b["h"]
    x = (np.arange(nx) + 0.5) * h
    y = (np.arange(ny) + 0.5) * h
    return {**b, "x": x, "y": y, "geom": b}


def ic_bunsen2d(spec, N, oracle, si):
    """dns bunsenIC (dns/flame1d.cpp): t = tf J + tCo C on the ideal cone,
    Y = (sigma - tf)/(sigma - 1) J, p = p0, rho = 1/t, u = m/rho with m the
    row's inflow rho u (box), v = 0. Cells i fastest (blockMesh order)."""
    m = mesh_bunsen2d(spec, N, oracle.dth)
    b = m["geom"]
    X, Y = np.meshgrid(m["x"], m["y"])            # (ny, nx), i fastest
    uin, tin, _, jet = bunsen_inflow(b, Y)
    co = 1.0 - jet                                # pilot 0: C = 1 - J
    tf = oracle.profile(bunsen_cone_distance(b, X, Y))["T"]
    t = tf * jet + b["tCo"] * co
    Yf = (oracle.sigma - tf) / (oracle.sigma - 1) * jet
    u = uin / tin * t                             # m / rho, rho = 1/t
    n = m["nx"] * m["ny"]
    U = np.zeros((n, 3))
    U[:, 0] = u.ravel() * si["SL"]
    sp = spec["species"]
    return {"T": t.ravel() * si["Tu"], "p": np.full(n, si["p0"]), "U": U,
            sp[0]: Yf.ravel(), sp[1]: 1.0 - Yf.ravel()}


def foam_bcs_bunsen2d(spec, oracle, si, m):
    """inlet: the per-row bunsenInflow profile at the face centres (fixed U,
    T, Y), zeroGradient p. outlet: waveTransmissive p -> p0 (dns' bunsen
    outflow target pB = p0), zeroGradient U, T, Y. y: cyclic."""
    sp = spec["species"]
    b = m["geom"]
    uin, tin, Yin, _ = bunsen_inflow(b, m["y"])
    U = np.zeros((m["ny"], 3))
    U[:, 0] = uin * si["SL"]
    L = m["nx"] * m["h"] * si["dD"]
    zg = writer.bc("zeroGradient")
    cy = writer.bc("cyclic")
    inlet = {"U": writer.bc("fixedValue", U),
             "T": writer.bc("fixedValue", tin * si["Tu"]),
             "p": zg, sp[0]: writer.bc("fixedValue", Yin),
             sp[1]: writer.bc("fixedValue", 1.0 - Yin)}
    outlet = {"U": zg, "T": zg, sp[0]: zg, sp[1]: zg,
              "p": writer.bc("waveTransmissive", gamma=GAMMA, pInf=si["p0"],
                             lInf=L)}
    cyc = {q: cy for q in ("U", "T", "p", sp[0], sp[1])}
    return {"inlet": inlet, "outlet": outlet, "lower": cyc, "upper": cyc}


def block_2d(case, m, si):
    writer.block_mesh_2d(case, m["nx"], m["ny"], m["h"] * si["dD"])


def no_probes(m, si):
    return []


# --- geometry: wrinkle2d (planar flame + one wrinkle mode, --ny periodic) -----

GOLDEN = 0.6180339887498949


def flame_wrinkle(A, n, y, Ly):
    """dns::flameWrinkle (dns/flame1d.cpp): sum_k (A/sqrt n) cos(2 pi k y/Ly +
    phi_k), phi_k = 2 pi frac(k^2 golden). A, y, Ly in delta_D."""
    s = np.zeros_like(y)
    for k in range(1, n + 1):
        phi = 2 * math.pi * math.fmod(float(k) * k * GOLDEN, 1.0)
        s = s + A / math.sqrt(n) * np.cos(2 * math.pi * k * y / Ly + phi)
    return s


def mesh_wrinkle2d(spec, N, dth):
    m = mesh_1d(spec, N, dth)
    ny = spec["ny"]
    m.update({"ny": ny, "y": (np.arange(ny) + 0.5) * m["h"]})
    return m


def ic_wrinkle2d(spec, N, oracle, si):
    """dns flameIC: premixedState of the oracle at x - x_f0 - w(y), w =
    flameWrinkle(A dth, modes): p = p0 - m^2 (1/rho0 - 1), rho = p/T, u =
    m/rho per cell (x momentum only), m = the inflow speed."""
    m = mesh_wrinkle2d(spec, N, oracle.dth)
    X, Yc = np.meshgrid(m["x"], m["y"])
    Ly = m["ny"] * m["h"]
    w = flame_wrinkle(spec["wrinkle"] * oracle.dth, spec["wrinkle_modes"], Yc, Ly)
    t = oracle.profile(X - m["xf0"] - w)["T"].ravel()
    Tu = si["TuBox"]
    mf = spec["inflow"]
    p = Tu - mf * mf * (t - 1.0)
    rho = p / (t * Tu)
    u = mf / rho
    Y = (oracle.sigma - t) / (oracle.sigma - 1)
    U = np.zeros((t.size, 3))
    U[:, 0] = u * si["SL"]
    sp = spec["species"]
    return {"T": t * si["Tu"], "p": p * si["pUnit"], "U": U,
            sp[0]: Y, sp[1]: 1.0 - Y}


def foam_bcs_wrinkle2d(spec, oracle, si, m):
    """The 1D patch conditions + the cyclic y pair."""
    d = foam_bcs_1d(spec, oracle, si, m)
    cy = writer.bc("cyclic")
    sp = spec["species"]
    cyc = {q: cy for q in ("U", "T", "p", sp[0], sp[1])}
    d.update({"lower": cyc, "upper": cyc})
    return d


# --- geometry: diffusion2d / candle2d (dns flame --mode diffusion | candle) -----

def erfc_inv(y):
    """dns::erfcInv (dns/diffusion.cpp): bisection to 1e-3, then Newton."""
    lo, hi = -6.0, 6.0
    while hi - lo > 1e-3:
        mid = 0.5 * (lo + hi)
        if math.erfc(mid) > y:
            lo = mid
        else:
            hi = mid
    x = 0.5 * (lo + hi)
    for _ in range(50):
        dx = (math.erfc(x) - y) / (-2.0 / math.sqrt(math.pi) * math.exp(-x * x))
        x -= dx
        if abs(dx) <= 1e-16 * max(1.0, abs(x)):
            break
    return x


def zst(md):
    return md["yO2"] / (md["s"] * md["yF1"] + md["yO2"])


def q_over_cpT0(md):
    return (md["sigma"] - 1.0) / (md["yF1"] * zst(md))


def theta_diff(md):
    """Ta/T0 = beta sigma^2/(sigma - 1)."""
    s = md["sigma"]
    return md["beta"] * s * s / (s - 1.0)


def burke_schumann(md, Z):
    """dns::burkeSchumann: (Y_F, Y_O, t = T/T0) at mixture fraction Z."""
    Z = np.asarray(Z, float)
    zs = zst(md)
    YF = np.where(Z <= zs, 0.0, md["yF1"] * (Z - zs) / (1.0 - zs))
    YO = np.where(Z <= zs, md["yO2"] * (1.0 - Z / zs), 0.0)
    t = 1.0 + q_over_cpT0(md) * (md["yF1"] * Z - YF)
    return YF, YO, t


def mixture_fraction(md, YF, YO):
    return (md["s"] * np.asarray(YF, float) - np.asarray(YO, float) + md["yO2"]) / (
        md["s"] * md["yF1"] + md["yO2"])


def diff_geom(spec):
    """dns/config.cpp's diffusion/candle grid and BunsenGeom (box units, l):
    h = 1/N, ny = 2 ceil((W/2 + wco) N), nx = ceil(L N) + 2 sink, yc = half
    h; length default ceil(min(1.25 x the constant-density sheet tip + 8,
    20 W)) (the jet; the candle's is 32)."""
    b = spec["burner"]
    md = spec["model"]
    N = spec["N"]
    h = 1.0 / N
    W = b["slot"]
    wco = b.get("coflow_width", W)
    tip = W * W / (16.0 * erfc_inv(1.0 - zst(md)) ** 2)
    length = b.get("length", math.ceil(min(1.25 * tip + 8, 20.0 * W)))
    half = int(math.ceil((0.5 * W + wco) * N - 1e-9))
    nb = spec["box"]["sink"]
    return {"W": W, "uCo": b.get("coflow_speed", 1.0), "edge": b["edge"],
            "yc": half * h, "nx": int(math.ceil(length * N - 1e-9)) + 2 * nb,
            "ny": 2 * half, "h": h, "nb": nb, "length": length, "tipCold": tip,
            "ignition": b.get("ignition", 0.0), "gravity": b.get("gravity", 0.0),
            "quiescent": b.get("quiescent", False)}


def diffusion_inflow(md, b, y):
    """dns::diffusionInflow (pilot 0): bunsenInflow's J, C = 1 - J (tanh
    blends), Z = J, the Burke-Schumann state; u = J + uCo C. Returns
    (u, Z, Y_F, Y_O, t, jet)."""
    r = np.abs(np.asarray(y, float) - b["yc"])
    jet = 0.5 * (1.0 - np.tanh((r - 0.5 * b["W"]) / b["edge"]))
    co = 0.5 * (1.0 + np.tanh((r - 0.5 * b["W"]) / b["edge"]))
    if b.get("quiescent"):          # dns::quiescentInflow: air at rest, T0
        jet = np.zeros_like(r)
        co = np.zeros_like(r)
    Z = jet
    YF, YO, t = burke_schumann(md, Z)
    return jet + b["uCo"] * co, Z, YF, YO, t, jet


def candle_ambient_p(p0, g, RT0, x):
    """dns::candleAmbientP: p0 exp(-g x/(R T0)) (g 0: p0)."""
    x = np.asarray(x, float)
    return p0 * np.exp(-g * x / RT0) if g > 0 else np.full_like(x, p0)


def mesh_diffusion2d(spec, N, _dth=None):
    b = diff_geom({**spec, "N": N})
    x = (np.arange(b["nx"]) + 0.5) * b["h"]
    y = (np.arange(b["ny"]) + 0.5) * b["h"]
    return {**b, "x": x, "y": y, "geom": b}


def ic_diffusion2d(spec, N, _oracle, si):
    """dns diffusionIC (dns/flame1d.cpp): the inflow row extruded (ignition
    0) or the burning column Z(y) w(x), w = (1 - tanh((x - L_ign)/edge))/2;
    p = p0 (candle: the ambient column p_a(x)), rho = p/(p0 t), u = the
    row's (column: uCo + (u - uCo) w), v = 0. Cells i fastest. SI out."""
    md = spec["model"]
    m = mesh_diffusion2d(spec, N)
    b = m["geom"]
    X, Yc = np.meshgrid(m["x"], m["y"])            # (ny, nx), i fastest
    u, Z, YF, YO, t, _ = diffusion_inflow(md, b, Yc)
    if b["ignition"] > 0 and not b["quiescent"]:
        w = 0.5 * (1.0 - np.tanh((X - b["ignition"]) / b["edge"]))
        u = b["uCo"] + (u - b["uCo"]) * w
        Z = Z * w
        YF, YO, t = burke_schumann(md, Z)
    T0 = si["T0Box"]
    p = candle_ambient_p(T0, b["gravity"], T0, X)  # R T0 = p0 = T0 (box)
    n = m["nx"] * m["ny"]
    U = np.zeros((n, 3))
    U[:, 0] = u.ravel() * si["U"]
    sp = spec["species"]
    out = {"T": t.ravel() * si["T0"], "p": p.ravel() * si["pUnit"], "U": U,
           sp[0]: YF.ravel(), sp[1]: YO.ravel()}
    if "Zt" in sp:
        out["Zt"] = ZT_EPS * Z.ravel()
    out[sp[-1]] = np.maximum(0.0, 1.0 - sum(out[s] for s in sp[:-1]))
    return out


def inlet_state(spec, m):
    """The inflow row at the face centres (ascending y): u, Z, Y_F, Y_O, t."""
    return diffusion_inflow(spec["model"], m["geom"], m["y"])


def foam_bcs_diffusion2d(spec, _oracle, si, m):
    """inlet: the dns inflow row (diffusionInflow at the face centres) as
    fixedValue U, T, Y_F, Y_O, Zt = Z, inert = the balance; zeroGradient p.
    dns PINS rho Y_k of the first CELL row to these values after its NSCBC
    overwrite and relaxes that cell's u, rho (etaIn 100): foam holds them on
    the FACE, half a cell upstream, and conducts / diffuses through it.
    outlet: waveTransmissive p -> p0 (lInf = box), zeroGradient U, T, Y.
    y: cyclic."""
    sp = spec["species"]
    u, Z, YF, YO, t, _ = inlet_state(spec, m)
    U = np.zeros((m["ny"], 3))
    U[:, 0] = u * si["U"]
    vals = {sp[0]: YF, sp[1]: YO}
    if "Zt" in sp:
        vals["Zt"] = ZT_EPS * Z
    vals[sp[-1]] = np.maximum(0.0, 1.0 - sum(vals[s] for s in sp[:-1]))
    L = m["nx"] * m["h"] * si["l"]
    zg = writer.bc("zeroGradient")
    cy = writer.bc("cyclic")
    inlet = {"U": writer.bc("fixedValue", U), "T": writer.bc("fixedValue", t * si["T0"]),
             "p": zg}
    inlet.update({s: writer.bc("fixedValue", vals[s]) for s in sp})
    outlet = {q: zg for q in ["U", "T"] + list(sp)}
    outlet["p"] = writer.bc("waveTransmissive", gamma=GAMMA, pInf=si["p0"], lInf=L)
    cyc = {q: cy for q in ["U", "T", "p"] + list(sp)}
    return {"inlet": inlet, "outlet": outlet, "lower": cyc, "upper": cyc}


def block_diff(case, m, si):
    writer.block_mesh_2d(case, m["nx"], m["ny"], m["h"] * si["l"])


def ic_candle2d(spec, N, _oracle, si):
    """diffusionIC with gravity (p = p_a(x) on every cell, rho = p/(p0 t))
    + p_rgh = p - rho g.C = p + rho g x (rhoReactingBuoyantFoam re-forms it
    from p at start: p is the master)."""
    out = ic_diffusion2d(spec, N, None, si)
    m = mesh_diffusion2d(spec, N)
    X, _ = np.meshgrid(m["x"], m["y"])
    rho = out["p"] / (si["R"] * out["T"])
    out["p_rgh"] = out["p"] + rho * si["g"] * X.ravel() * si["l"]
    return out


def foam_bcs_candle2d(spec, _oracle, si, m):
    """inlet: the dns inflow row as fixedValue U, T, Y (foam_bcs_diffusion2d),
    p_rgh fixedFluxPressure (the buoyant solvers' wall/inlet condition: the
    face gradient balancing the fixed flux), p calculated.
    outlet (the top): p_rgh prghPressure at the AMBIENT static p_a(L)
    (dns: NSCBC p -> p_a(x_out) at the last row, anchor sigma 4, with the
    incoming acoustic wave carrying the ambient hydrostatic gradient);
    U pressureInletOutletVelocity (zeroGradient out, the flux-normal velocity
    in: tangential 0 -- dns: v -> 0 for air drawn in); T, Y inletOutlet with
    the ambient air (T0, Y_O2; dns: backflow relaxes rho -> rho_a and the
    species face carries air in). y: cyclic."""
    d = foam_bcs_diffusion2d(spec, None, si, m)
    sp = spec["species"]
    md = spec["model"]
    b = m["geom"]
    L = m["nx"] * m["h"]
    T0 = si["T0Box"]
    pin = float(candle_ambient_p(T0, b["gravity"], T0, 0.0)) * si["pUnit"]
    pout = float(candle_ambient_p(T0, b["gravity"], T0, L)) * si["pUnit"]
    rho_a = pout / (si["R"] * si["T0"])
    zg_out = {"T": si["T0"], sp[0]: 0.0, sp[1]: md["yO2"], sp[-1]: 1.0 - md["yO2"]}
    if "Zt" in sp:
        zg_out["Zt"] = 0.0
    d["inlet"]["p"] = writer.bc("calculated", pin)
    d["inlet"]["p_rgh"] = writer.bc("fixedFluxPressure", pin)
    out = {q: writer.bc("inletOutlet", v) for q, v in zg_out.items()}
    out["U"] = writer.bc("pressureInletOutletVelocity", (0, 0, 0))
    out["p"] = writer.bc("calculated", pout)
    out["p_rgh"] = writer.bc("prghPressure", pout,
                             rgh=pout + rho_a * si["g"] * L * si["l"])
    d["outlet"] = out
    for pt in ("lower", "upper"):
        d[pt]["p_rgh"] = writer.bc("cyclic")
    return d


# geometry -> (mesh, ic, bcs, blockMesh writer, probe points)
GEOMETRY = {"1d": (mesh_1d, ic_1d, foam_bcs_1d, block_1d, probes_1d),
            "bunsen2d": (mesh_bunsen2d, ic_bunsen2d, foam_bcs_bunsen2d,
                         block_2d, no_probes),
            "wrinkle2d": (mesh_wrinkle2d, ic_wrinkle2d, foam_bcs_wrinkle2d,
                          block_2d, no_probes),
            "diffusion2d": (mesh_diffusion2d, ic_diffusion2d,
                            foam_bcs_diffusion2d, block_diff, no_probes),
            "candle2d": (mesh_diffusion2d, ic_candle2d, foam_bcs_candle2d,
                         block_diff, no_probes)}


# --- dns ----------------------------------------------------------------------

def dns_argv_diff(spec, N, out):
    """--mode diffusion | candle with every model / burner flag explicit."""
    md, b = spec["model"], spec["burner"]
    mode = "candle" if spec["geometry"] == "candle2d" else "diffusion"
    a = ["flame", "--mode", mode, "--mach", f"{md['mach']:g}",
         "--stoich", f"{md['s']:g}", "--fuel-y", f"{md['yF1']:g}",
         "--ox-y", f"{md['yO2']:g}", "--expansion", f"{md['sigma']:g}",
         "--zeldovich", f"{md['beta']:g}", "--damkohler", f"{md['Da']:g}",
         "--prandtl", f"{md['Pr']:g}", "--cells-per-unit", f"{N:g}"]
    if b.get("quiescent"):          # the burner flags are rejected: its defaults
        a += ["--quiescent"]
    else:
        a += ["--slot-width", f"{b['slot']:g}",
              "--coflow-width", f"{b.get('coflow_width', b['slot']):g}",
              "--coflow-speed", f"{b.get('coflow_speed', 1.0):g}",
              "--slot-edge", f"{b['edge']:g}"]
    a += ["--length", f"{diff_geom({**spec, 'N': N})['length']:g}",
          "--nscbc-sink", f"{spec['box']['sink']:d}", "--tt", f"{spec['tt']:g}",
          "--out", out]
    if b.get("ignition") and not b.get("quiescent"):
        a += ["--ignition", f"{b['ignition']:g}"]
    if mode == "candle":
        a += ["--gravity", f"{b['gravity']:g}"]
    if spec.get("fdt"):
        a += ["--emit", "stats", "--emit", "bin", "--fdt", f"{spec['fdt']:g}"]
    return a + list(spec["dns"])


def dns_argv(spec, N, out):
    md = spec["model"]
    if spec["geometry"] in DIFF_GEOMS:
        return dns_argv_diff(spec, N, out)
    a = ["flame", "--mach", f"{md['mach']:g}", "--expansion", f"{md['sigma']:g}",
         "--zeldovich", f"{md['beta']:g}", "--prandtl", f"{md['Pr']:g}",
         "--cells-per-delta", f"{N:g}", "--tt", f"{spec['tt']:g}",
         "--out", out]
    if spec.get("profile_dt"):
        a += ["--profile-dt", f"{spec['profile_dt']:g}"]
    if spec["geometry"] in ("1d", "wrinkle2d"):
        b = spec["box"]
        a += ["--upstream", f"{b['upstream']:g}", "--downstream",
              f"{b['downstream']:g}", "--nscbc-sink", f"{b['sink']:d}",
              "--inflow-speed", f"{spec['inflow']:g}"]
    if spec["geometry"] == "wrinkle2d":
        a += ["--ny", f"{spec['ny']:d}", "--wrinkle", f"{spec['wrinkle']:g}",
              "--wrinkle-modes", f"{spec['wrinkle_modes']:d}"]
    if spec["geometry"] == "bunsen2d":
        b = spec["burner"]
        a += ["--mode", "bunsen", "--slot-width", f"{b['slot']:g}",
              "--jet-speed", f"{b['jet']:g}", "--nscbc-sink",
              f"{spec['box']['sink']:d}"]
    if spec.get("fdt"):
        a += ["--emit", "stats", "--emit", "bin", "--fdt", f"{spec['fdt']:g}"]
    return a + list(spec["dns"])


# --- OpenFOAM -----------------------------------------------------------------

def foam_runs(spec):
    """(run name, N, settings) of every foam run of the case."""
    f = spec["foam"]
    base = {"dt": f["dt"], "n_outer": f["n_outer"], "n_corr": f["n_corr"]}
    if "np" in f:
        base["np"] = f["np"]
    runs = []
    for v, o in f["variants"].items():
        for N in spec["ladder"]:
            runs.append((f"foam_{v}_c{N:g}", N, {**base, **o, "variant": v}))
    for name, o in f.get("checks", {}).items():
        b = f["variants"][o["base"]]
        s = {**base, **b, **{k: v for k, v in o.items() if k not in ("base", "N")},
             "variant": name}
        if "restart" in o:      # started from the base variant's field at t0
            s["restart_base"] = o["base"]
        runs.append((f"foam_{name}_c{o['N']:g}", o["N"], s))
    return runs


def diff_reaction(md, sp):
    """F + s O -> (1+s) P, first order in EACH reactant (O^1: OpenFOAM's
    order defaults to the stoichiometric coefficient)."""
    s = md["s"]
    so = "" if s == 1 else f"{s:g}"
    return f"{sp[0]} + {so}{sp[1]}^1 = {1 + s:g}{sp[-1]}"


def write_foam_diff(spec, N, st, si, case):
    """The two-reactant case: model C1 in reactingFoam (jet) or
    rhoReactingBuoyantFoam (candle: rhoThermo, p_rgh, constant/g)."""
    mesh_f, ic_f, bcs_f, block_f, _ = GEOMETRY[spec["geometry"]]
    m = mesh_f(spec, N)
    sp = spec["species"]
    solved = sp[:-1]
    buoyant = spec["geometry"] == "candle2d"
    block_f(case, m, si)
    writer.thermo_files(case, si, sp, {sp[0]: si["q"]},
                        thermo_type="heRhoThermo" if buoyant else "hePsiThermo")
    writer.reactions_file(case, sp, [(diff_reaction(spec["model"], sp),
                                      si["A"], si["Ta"])])
    writer.chemistry_files(case, integrate=st.get("integrate", False))
    writer.fv_options(case, solved, le1=st.get("le1", True),
                      buoyancy_work=buoyant)
    writer.schemes(case, st["yscheme"], st["hscheme"], sp,
                   mv=st.get("mv", "multivariateIndependent"), per=st.get("per"),
                   flux_required="p_rgh" if buoyant else None)
    writer.solution(case, st["n_outer"], st["n_corr"], multi_d=True,
                    p_name="p_rgh" if buoyant else "p")
    writer.decompose(case, st.get("np", 1))
    tu = si["tUnit"]
    t0 = st.get("restart", 0.0)
    writer.control(case, st["dt"] * tu, st.get("tt", spec["tt"]) * tu,
                   spec["foam_write_dt"] * tu, [],
                   precision=spec.get("write_precision", 14),
                   application="rhoReactingBuoyantFoam" if buoyant else "reactingFoam",
                   start_time=t0 * tu)
    if buoyant:
        writer.gravity(case, si["g"])
    writer.fields_0(case, ic_f(spec, N, None, si), si, bcs_f(spec, None, si, m),
                    solved, sp)
    return m


def write_foam(spec, N, st, oracle, si, case):
    if spec["geometry"] in DIFF_GEOMS:
        return write_foam_diff(spec, N, st, si, case)
    mesh_f, ic_f, bcs_f, block_f, probes_f = GEOMETRY[spec["geometry"]]
    m = mesh_f(spec, N, oracle.dth)
    sp = spec["species"]
    solved = sp[:-1]
    block_f(case, m, si)
    writer.thermo_files(case, si, sp, {sp[0]: si["q"]})
    writer.reactions_file(case, sp, [(eq, si["A"], si["Ta"])
                                     for eq in spec["reactions"]])
    writer.chemistry_files(case, integrate=st.get("integrate", False))
    writer.fv_options(case, solved, le1=st.get("le1", True))
    writer.schemes(case, st["yscheme"], st["hscheme"], sp)
    multi_d = spec["geometry"] != "1d"
    writer.solution(case, st["n_outer"], st["n_corr"], multi_d=multi_d)
    if multi_d:
        writer.decompose(case, st.get("np", 1))
    tau = si["tau"]
    writer.control(case, st["dt"] * tau, st.get("tt", spec["tt"]) * tau,
                   spec["foam_write_dt"] * tau, probes_f(m, si),
                   precision=spec.get("write_precision", 14))
    writer.fields_0(case, ic_f(spec, N, oracle, si), si,
                    bcs_f(spec, oracle, si, m), solved, sp)
    return m
