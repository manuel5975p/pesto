"""The C0 premixed model of `dns flame` (docs/chem.md): the P0 phase-plane
oracle (a numpy port of dns/chem.cpp, same start, same logit RK4, same
Hermite profile) and the ONE box-units -> SI mapping of the OpenFOAM side.

Box units (docs/chem.md "Units (box)"): rho_u = S_L = R = 1, delta_D =
lambda_u/(rho_u cp S_L) = 1, tau = delta_D/S_L = 1, cp = 3.5, mu = Pr,
A = Lambda, q = cp (sigma - 1) Tu, Tu = p0 = 1/(gamma M^2).
"""
import math

import numpy as np

GAMMA = 1.4
# Universal gas constant [J/kmol/K], CODATA 2018 exact. NOT OpenFOAM's:
# v2606 thermodynamic::RR = 1e3 NA k with NA 6.0221417930e23, k 1.38065e-23
# (etc/controlDict) = 8314.4700665, 9.0e-7 above. foam's R = RR_foam/W is
# then 9e-7 larger than si_map's at the written Cp: densities at a given
# (p, T) differ by 9e-7 -- below every quoted comparison digit; kept so
# the recorded case files and compare outputs stay byte-reproducible.
RR = 8314.46261815324

_S0, _S1 = 1e-9, 1 - 1e-7   # dns/chem.cpp kS0, kS1


def _shoot(m, theta, sigma, n, record=False):
    """dns/chem.cpp shoot() in the non-dimensional problem (Tu = 1, rho = 1/T,
    cp = lambda = A = 1), vectorized over an array of mass fluxes m.
    Returns (tooFast bool array, table or None)."""
    m = np.atleast_1d(np.asarray(m, float))
    Tu, Tb = 1.0, sigma
    dT = Tb - Tu
    q = dT
    B = 1.0 / Tb * math.exp(-theta / Tb)
    a = 0.5 * (-m + np.sqrt(m * m + 4 * B))
    u0 = math.log(_S0 / (1 - _S0))
    u1 = math.log(_S1 / (1 - _S1))
    h = (u1 - u0) / n

    def rhs(u, p):
        s = 1 / (1 + math.exp(-u))
        T = Tb - dT * s
        dTdu = -dT * s * (1 - s)
        om = (1 / T) * s * math.exp(-theta / T)
        return (m - q * om / p) * dTdu, dTdu / p

    p = a * dT * _S0
    x = np.zeros_like(m)
    dead = np.zeros(m.shape, bool)
    tab = [] if record else None
    if record:
        tab.append((Tb - dT * _S0, 0.0, p[0]))
    for k in range(n):
        u = u0 + k * h
        with np.errstate(all="ignore"):
            k1p, k1x = rhs(u, p)
            p2 = p + 0.5 * h * k1p
            dead |= ~(p2 > 0)
            k2p, k2x = rhs(u + 0.5 * h, p2)
            p3 = p + 0.5 * h * k2p
            dead |= ~(p3 > 0)
            k3p, k3x = rhs(u + 0.5 * h, p3)
            p4 = p + h * k3p
            dead |= ~(p4 > 0)
            k4p, k4x = rhs(u + h, p4)
            p = p + h / 6 * (k1p + 2 * k2p + 2 * k3p + k4p)
            x = x + h / 6 * (k1x + 2 * k2x + 2 * k3x + k4x)
            dead |= ~(p > 0)
        p = np.where(dead, 1.0, p)   # keep dead lanes finite
        if record and dead[0]:       # dns shoot() returns here: table ends
            break
        if record:
            s = 1 / (1 + math.exp(-(u + h)))
            tab.append((Tb - dT * s, x[0], p[0]))
    Tend = Tb - dT * _S1
    resid = p - m * (Tend - Tu)
    return dead | (resid < 0), tab


def hermite(x, xs, Ts, Gs):
    """dns::flameProfile: cubic Hermite on (x_k, T_k, dT/dx_k), clamped."""
    x = np.asarray(x, float)
    k = np.clip(np.searchsorted(xs, x, side="right") - 1, 0, len(xs) - 2)
    hh = xs[k + 1] - xs[k]
    t = (x - xs[k]) / hh
    t2, t3 = t * t, t * t * t
    T = ((2 * t3 - 3 * t2 + 1) * Ts[k] + (t3 - 2 * t2 + t) * hh * Gs[k] +
         (-2 * t3 + 3 * t2) * Ts[k + 1] + (t3 - t2) * hh * Gs[k + 1])
    T = np.where(x <= xs[0], Ts[0], T)
    return np.where(x >= xs[-1], Ts[-1], T)


class Oracle:
    """premixedEigen(sigma, beta) of dns/chem.cpp, constant lambda (no
    Sutherland). Lambda, dth (delta_th/delta_D) and the profile table in box
    units (x in delta_D, T in Tu, x = 0 at the mid temperature)."""

    def __init__(self, sigma=5.0, beta=8.0, n=20000):
        self.sigma, self.beta = sigma, beta
        self.theta = beta * sigma * sigma / (sigma - 1.0)
        lo = hi = 1.0
        if _shoot(1.0, self.theta, sigma, n)[0][0]:
            while _shoot(lo, self.theta, sigma, n)[0][0]:
                lo *= 0.5
        else:
            while not _shoot(hi, self.theta, sigma, n)[0][0]:
                hi *= 2
        # multisection: 63 interior points per pass (dns bisects; same limit)
        while hi - lo > 1e-15 * hi:
            ms = np.linspace(lo, hi, 65)[1:-1]
            fast = _shoot(ms, self.theta, sigma, n)[0]
            j = int(np.argmax(fast)) if fast.any() else len(ms)
            nlo = ms[j - 1] if j > 0 else lo
            nhi = ms[j] if j < len(ms) else hi
            if (nlo, nhi) == (lo, hi):
                break
            lo, hi = nlo, nhi
        m1 = 0.5 * (lo + hi)
        _, tab = _shoot(m1, self.theta, sigma, n, record=True)
        T, x, p = (np.array(c[::-1]) for c in zip(*tab))
        G = p.copy()                  # lambda = 1: dT/dx = p
        Tm = 0.5 * (1 + sigma)
        k = int(np.searchsorted(T, Tm, side="right"))
        w = (Tm - T[k - 1]) / (T[k] - T[k - 1])
        x = x - (x[k - 1] + w * (x[k] - x[k - 1]))
        for _ in range(4):
            x = x - (Tm - float(hermite(0.0, x, T, G))) / G[k]
        self.Lambda = 1 / (m1 * m1)
        self.x = x * m1              # A -> Lambda similarity (S_L = 1)
        self.G = G / m1
        self.T = T
        self.dth = (sigma - 1) / self.G.max()

    def profile(self, xi, Tu=1.0):
        """T(x)/Tu * Tu at x (delta_D, mid temperature at 0); isobaric
        low-Mach companions: Y = (sigma - t)/(sigma - 1), rho = 1/t, u = t
        (S_L = 1, rho_u = 1), omega = Lambda rho Y exp(-theta/t)."""
        t = hermite(xi, self.x, self.T, self.G)
        Y = (self.sigma - t) / (self.sigma - 1)
        om = self.Lambda / t * np.maximum(Y, 0) * np.exp(-self.theta / t)
        return {"T": t * Tu, "Y": Y, "rho": 1 / t, "u": t, "omega": om}


def si_map_diff(md, T0=300.0, p0=101325.0, W=28.014, l=1e-3, gravity=0.0):
    """Box units of the two-reactant model C1 (docs/chem.md 2D slot-jet
    diffusion flame: rho0 = R = U_jet = 1, l = D0/U_jet, mu = Pr, T0 = p0 =
    1/(gamma M^2)) -> SI, the ONE mapping of the diffusion / candle cases.
    Inputs: md (s, yF1, yO2, sigma, beta, Da, Pr, mach), the stream state
    (T0 [K], p0 [Pa]), the common W [kg/kmol], the length unit l [m] (sets
    mu = Pr rho0 U l), the box gravity g [U^2/l].
    Rate: dns omega = A rho^2 Y_F Y_O e^(-Ta/T) [fuel mass/vol/time]; OpenFOAM
    F + s O^1: RR_F = -W kf (rho Y_F/W)(rho Y_O/W) -> kf = A_SI W [m^3/kmol/s],
    A_SI = A_box (U/l)/rho0, A_box = e^(ln Da + Ta/T_ad)."""
    s, yF1, yO2, sig = md["s"], md["yF1"], md["yO2"], md["sigma"]
    mach, Pr = md["mach"], md["Pr"]
    R = RR / W
    cp = GAMMA / (GAMMA - 1) * R
    c0 = math.sqrt(GAMMA * R * T0)
    U = mach * c0
    rho0 = p0 / (R * T0)
    mu = Pr * rho0 * U * l
    tu = l / U
    zst = yO2 / (s * yF1 + yO2)
    qcp = (sig - 1.0) / (yF1 * zst)
    theta = md["beta"] * sig * sig / (sig - 1.0)
    lnA = math.log(md["Da"]) + theta / sig
    A_si = math.exp(lnA) / (rho0 * tu)
    T0Box = 1.0 / (GAMMA * mach * mach)
    return {
        "W": W, "R": R, "cp": cp, "T0": T0, "p0": p0, "rho0": rho0, "c0": c0,
        "U": U, "l": l, "tUnit": tu, "mu": mu, "kappa": mu * cp / Pr, "Pr": Pr,
        "A": A_si * W, "A_box": math.exp(lnA), "lnA": lnA, "Ta": theta * T0,
        "q": qcp * cp * T0, "qOverCpT0": qcp, "zst": zst, "theta": theta,
        "g": gravity * U * U / l, "gBox": gravity,
        "T0Box": T0Box, "pUnit": p0 / T0Box, "TUnit": T0 / T0Box,
        "omegaUnit": rho0 / tu,
    }


def si_map(mach, sigma, beta, Pr, Lambda, Tu=300.0, p0=101325.0, W=28.014,
           dD=1e-4):
    """Box units -> SI, the ONE mapping of the OpenFOAM cases.
    Inputs: flame Mach M = S_L/c_u, sigma, beta, Pr, the oracle Lambda, the
    fresh state (Tu [K], p0 [Pa]), the common molecular weight W [kg/kmol]
    and the Damkoehler-free length delta_D [m] (sets mu).
    Box quantity * unit = SI: lengths * dD, times * tau, velocities * SL,
    densities * rho_u, pressures * (p0/p0_box), temperatures * (Tu/Tu_box),
    omega * rho_u/tau, Qdot * rho_u q_SI/tau."""
    R = RR / W
    cp = GAMMA / (GAMMA - 1) * R
    cu = math.sqrt(GAMMA * R * Tu)
    SL = mach * cu
    rho_u = p0 / (R * Tu)
    lam = rho_u * cp * SL * dD
    mu = Pr * lam / cp
    tau = dD / SL
    theta = beta * sigma * sigma / (sigma - 1)
    TuBox = 1.0 / (GAMMA * mach * mach)
    return {
        "W": W, "R": R, "cp": cp, "Tu": Tu, "p0": p0, "rho_u": rho_u,
        "c_u": cu, "SL": SL, "dD": dD, "tau": tau, "mu": mu, "kappa": lam,
        "Pr": Pr, "A": Lambda / tau, "Ta": theta * Tu, "q": cp * (sigma - 1) * Tu,
        "sigma": sigma, "beta": beta, "mach": mach, "Lambda": Lambda,
        # box -> SI factors
        "TuBox": TuBox, "pUnit": p0 / TuBox, "TUnit": Tu / TuBox,
        "omegaUnit": rho_u / tau,
    }
