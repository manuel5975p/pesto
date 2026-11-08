// --shock (docs/shock.md): the pressure sensor and the two-point HLL correction
// of the structured compressible flux. Header-only, DNS_HD, T = float on the
// device, float or double in the host oracle (tests/shock_ref.hpp): the oracle
// and the kernels run these lines. Normal + area form with per-side gamma, so a
// curvilinear or FV caller can reuse it. No GPU include.
//
// The flux written is F_KG + sigma_f (F_HLL - F_KG) on the inviscid part of the
// face; viscous, conduction and SGS terms stay in kCompStage on top of D.
#pragma once

#include "dns/hd.hpp"

#include <cmath>
#include <cstdint>

namespace dns {

// --shock: the central solver, the sensed HLL correction, or the sensor alone
// (an instrument: the central solver bit for bit).
enum class ShockMode { None, Hll, Sense };

// The sensor ramp on v = |pp - 2 p + pm| / (pp + 2 p + pm) at stride 2.
// Bracket: live flame max 3.12e-3 at 4 cells/delta_th, M 0.02 (below kShockTh1
// x 6.4); a Mach 1.1 moving shock engages. Measured, not a tuning knob.
inline constexpr double kShockTh1 = 0.02, kShockTh2 = 0.06;
// Flame Mach bound of --shock hll. Bracket: flame v 1.17e-2 at M 0.05 (silent),
// 3.9e-2 sustained at M 0.1 (fires).
inline constexpr double kShockMachMax = 0.05;
// nDim * cfl bound of --shock hll, stableDt being the max over axes.
// Bracket: axis sum 1.2 stable on every 2D and 3D case, 1.4 diverges.
inline constexpr double kShockCflAxisSum = 1.2;
// FV --shock hll's background term, F += eps (1 - sigma) |u.S| D3 along the mesh line
// (dns/fvflux.hpp fvD4Row, docs/nozzle.md). Bracket: source flow 256x64, L1 p 3.25e-5,
// order 1.98, against bare KG's 1.26e-3, order 1.00. The cap is masterplan_cdnozzle's,
// twice the default; no stability bound was measured at it.
inline constexpr double kFvShockEps4 = 1.0 / 16.0, kFvShockEps4Max = 0.125;

// One cell as the face correction reads it. e = E / rho, gam = the cell's gamma.
template <class T>
struct ShockState
{
    T rho, u[3], p, e, gam;
};

// D of one face: mass, momentum, total energy.
template <class T>
struct ShockDelta
{
    T v[5];
};

// Stride-2 pressure jump of one axis: 0 on constant, linear and odd-even
// fields, so the sensor sees a shock, not a resolved wave or mesh noise.
// Pre: pm, p, pp > 0 on a live state (else the result reads as sensed).
template <class T>
DNS_HD inline T shockJump(T pm, T p, T pp)
{
    using std::fabs;
    return fabs(pp - T(2) * p + pm) / (pp + T(2) * p + pm);
}

// sigma as a byte, sigma = byte / 255: 0 at or below kShockTh1 (exactly: no
// write downstream), nonzero above it, 255 at or above kShockTh2 and on NaN (a
// diverged cell reads as sensed; stableDt reports it). Post: never 0 above th1.
template <class T>
DNS_HD inline std::uint8_t shockByte(T v)
{
    const T th1 = T(kShockTh1), th2 = T(kShockTh2);
    if (v <= th1) return 0;
    if (!(v < th2)) return 255;
    return std::uint8_t(1 + int(T(254) * (v - th1) / (th2 - th1)));
}

// sigma (F_HLL - F_KG) through the unit normal n: F_HLL first order, Davis wave
// speeds; F_KG in faceFlux's expression and order (comp_flux.hpp).
// Pre: L, R finite with rho, p > 0; |n| = 1. sigma == 0 returns exact zeros.
template <class T>
DNS_HD inline ShockDelta<T> shockFaceDelta(T sigma, const ShockState<T>& L, const ShockState<T>& R, const T n[3])
{
    using std::sqrt;
    const T unL = L.u[0] * n[0] + L.u[1] * n[1] + L.u[2] * n[2];
    const T unR = R.u[0] * n[0] + R.u[1] * n[1] + R.u[2] * n[2];
    const T cL = sqrt(L.gam * L.p / L.rho), cR = sqrt(R.gam * R.p / R.rho);
    const T a = unL - cL, b = unR - cR, c = unL + cL, d = unR + cR;
    const T SL = a < b ? a : b, SR = c > d ? c : d;
    const T bm = SL < T(0) ? SL : T(0), bp = SR > T(0) ? SR : T(0);
    const T mL = L.rho * unL, mR = R.rho * unR;

    const T rm = T(0.5) * (L.rho + R.rho);
    const T um[3] = {T(0.5) * (L.u[0] + R.u[0]), T(0.5) * (L.u[1] + R.u[1]), T(0.5) * (L.u[2] + R.u[2])};
    const T em = T(0.5) * (L.e + R.e);
    const T pm = T(0.5) * (L.p + R.p);
    const T un = um[0] * n[0] + um[1] * n[1] + um[2] * n[2];
    const T mk = rm * un;

    // (bp FL - bm FR + bp bm dU) / (bp - bm) in its centred form: one divide,
    // equal to the Euler flux bit for bit when L == R, and a mirror (L <-> R,
    // n -> -n) negates every term in the same order.
    const T iw = T(1) / (bp - bm), s = T(0.5) * (bp + bm) * iw, q = bp * bm * iw;
    auto hll = [&](T FL, T FR, T UL, T UR) { return T(0.5) * (FL + FR) + s * (FL - FR) + q * (UR - UL); };
    ShockDelta<T> D;
    D.v[0] = sigma * (hll(mL, mR, L.rho, R.rho) - mk);
    for (int k = 0; k < 3; ++k) {
        const T FL = mL * L.u[k] + L.p * n[k], FR = mR * R.u[k] + R.p * n[k];
        D.v[1 + k] = sigma * (hll(FL, FR, L.rho * L.u[k], R.rho * R.u[k]) - (mk * um[k] + pm * n[k]));
    }
    const T FL = mL * L.e + L.p * unL, FR = mR * R.e + R.p * unR;
    D.v[4] = sigma * (hll(FL, FR, L.rho * L.e, R.rho * R.e) - (mk * em + pm * un));
    return D;
}

} // namespace dns
