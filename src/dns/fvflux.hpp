// The FV path's face physics (docs/mesh.md, masterplan_meshsolver §2.3): the
// Kennedy-Gruber mean flux plus the face viscous terms, the corrected face
// gradient, and every boundary kind as a FACE FLUX (never a state or rhs
// overwrite). Header-only, DNS_HD, templated on the scalar T (double for the
// CPU oracle dns/fvref, float for the GPU path, Dual<N> for implicit
// Jacobians) and on the geometry scalar G (double on the host, float on the
// device; geometry is never differentiated).
//
// This RESTATES faceFluxCurvOf's algebra (gpu/internal/comp_flux.hpp) with an
// arbitrary face: the structured bodies stay as they are (FMA rule). On a
// uniform Cartesian FaceMesh the scheme IS the structured one: the corrected
// face gradient with least-squares cell gradients reproduces "normal compact,
// tangential averaged central" (fvref G1, 1e-13 relative).
//
// Mesh-agnostic: a face is (S, d, corr) and two cell states. Eddy viscosity
// (RANS/LES) and a rotating frame enter through the caller's mu, kappa and
// the advected frame area flux wS; scalars and species ride F[0] (later
// phases).
#pragma once

#include "dns/dual.hpp"
#include "dns/hd.hpp"
#include "dns/transport.hpp"

#include <cmath>
#include <cstdint>

namespace dns {

// rho, rho u, rho v, rho w, E: the mean-flow rows of the FV state [nV][nC].
inline constexpr int kFvMeanVars = 5;

// One cell's (or one boundary state's) primitives as the face flux reads
// them; e = E / rho is the total specific energy KG averages.
template <class T> struct FvPrim
{
    T rho, u[3], e, p, temp;
};

// A face flux through the area vector S: mass, momentum, total energy.
template <class T> struct FvFlux
{
    T rho, m[3], E;
};

// Primitives off the conserved state: Comp::pressure's operand order, so the
// oracle's primitives equal dns::Comp's bit for bit. Pre: rho > 0.
template <class T>
DNS_HD inline FvPrim<T> fvPrimOf(T rho, T mx, T my, T mz, T E, double gamma, double Rgas)
{
    FvPrim<T> q;
    q.rho = rho;
    q.u[0] = mx / rho;
    q.u[1] = my / rho;
    q.u[2] = mz / rho;
    q.e = E / rho;
    q.p = T(gamma - 1.0) * (E - T(0.5) * (mx * mx + my * my + mz * mz) / rho);
    q.temp = q.p / (rho * T(Rgas));
    return q;
}

// mu and kappa at the face temperature Tf: Transport::mu / kappa restated for
// any T (one Sutherland evaluation per FACE, never per cell). Bit-identical
// to Transport for T = double. Pre: Tf > 0.
template <class T>
DNS_HD inline void fvTransportAt(const Transport& tr, T Tf, T& mu, T& kappa)
{
    using std::sqrt;
    T r = T(1.0);
    if (tr.sutherland) {
        const T t = Tf / T(tr.Tref);
        const T s = T(tr.S / tr.Tref);
        r = t * sqrt(t) * (T(1.0) + s) / (t + s);
    }
    mu = T(tr.muRef) * r;
    kappa = mu * T(tr.cp()) / T(tr.Pr);
}

// How the face gradient corrects the average of the two cell gradients:
//   g_f = gbar + (phi_N - phi_P - gbar . d) corr,
//   AlongD:      corr = alpha d / |d|^2      (alpha 1: the "corrected" form;
//                g_f . d == phi_N - phi_P, the structured compact normal)
//   AlongNormal: corr = alpha S / (S . d)    (Nishikawa's alpha-damping;
//                alpha 1 is OpenFOAM's `corrected` snGrad on n)
// The rule is a geometry precomputation (dns/fvgrad fvFaceGeom); the kernel
// below is the same for both.
enum class FvFaceGradRule : std::uint8_t { AlongD, AlongNormal };

// Internal face gradient from the two cell gradients and values.
template <class T, class G>
DNS_HD inline void fvFaceGrad(const T gP[3], const T gN[3], T phiP, T phiN, const G d[3],
                              const G corr[3], T g[3])
{
    T gm[3];
    for (int k = 0; k < 3; ++k) gm[k] = T(0.5) * (gP[k] + gN[k]);
    const T r = (phiN - phiP) - (gm[0] * T(d[0]) + gm[1] * T(d[1]) + gm[2] * T(d[2]));
    for (int k = 0; k < 3; ++k) g[k] = gm[k] + r * T(corr[k]);
}

// Boundary face gradient: the owner's gradient corrected toward the face
// value phiB over d = Cf - C_own.
template <class T, class G>
DNS_HD inline void fvFaceGradBnd(const T gP[3], T phiP, T phiB, const G d[3], const G corr[3],
                                 T g[3])
{
    const T r = (phiB - phiP) - (gP[0] * T(d[0]) + gP[1] * T(d[1]) + gP[2] * T(d[2]));
    for (int k = 0; k < 3; ++k) g[k] = gP[k] + r * T(corr[k]);
}

// Viscous traction ts = tau . S (Newtonian, Stokes hypothesis) from the face
// velocity gradient gu[c][x] = d u_c / d x_x. faceFluxCurvOf's operand order.
template <class T, class G>
DNS_HD inline void fvTraction(const T gu[3][3], T mu, const G S[3], T ts[3])
{
    const T divu = gu[0][0] + gu[1][1] + gu[2][2];
    for (int c = 0; c < 3; ++c) {
        T s = T(0.0);
        for (int d = 0; d < 3; ++d)
            s = s + (mu * (gu[c][d] + gu[d][c]) - (c == d ? T(2.0 / 3.0) * mu * divu : T(0.0))) * T(S[d]);
        ts[c] = s;
    }
}

// Total Kennedy-Gruber convective minus viscous flux through the internal
// face S between cells A (owner) and B: product of face means, so the
// expression is identical from both sides (exactly conservative) and the
// convective part conserves discrete kinetic energy (fvref G2). gu, gT: the
// face gradients (fvFaceGrad of u, v, w, T). mu, kappa: the caller's face
// values (molecular at the face-mean T, plus any eddy part). wS: the frame
// velocity's area flux (rotating frame; 0 = inertial, bit-for-bit), which
// changes the ADVECTING normal velocity only -- pressure work stays on the
// absolute Un. mu == 0 returns the Euler flux.
template <class T, class G>
DNS_HD inline FvFlux<T> fvMeanFlux(const FvPrim<T>& A, const FvPrim<T>& B, const G S[3],
                                   const T gu[3][3], const T gT[3], T mu, T kappa, T wS = T(0.0))
{
    const T rm = T(0.5) * (A.rho + B.rho);
    const T um[3] = {T(0.5) * (A.u[0] + B.u[0]), T(0.5) * (A.u[1] + B.u[1]),
                     T(0.5) * (A.u[2] + B.u[2])};
    const T em = T(0.5) * (A.e + B.e);
    const T pm = T(0.5) * (A.p + B.p);
    const T Un = um[0] * T(S[0]) + um[1] * T(S[1]) + um[2] * T(S[2]);
    const T Ua = Un - wS;
    FvFlux<T> f{rm * Ua,
                {rm * Ua * um[0] + pm * T(S[0]), rm * Ua * um[1] + pm * T(S[1]),
                 rm * Ua * um[2] + pm * T(S[2])},
                rm * Ua * em + pm * Un};
    if (value(mu) == 0.0) return f;
    T ts[3];
    fvTraction(gu, mu, S, ts);
    T eflux = kappa * (gT[0] * T(S[0]) + gT[1] * T(S[1]) + gT[2] * T(S[2]));
    for (int c = 0; c < 3; ++c) {
        f.m[c] = f.m[c] - ts[c];
        eflux = eflux + um[c] * ts[c];
    }
    f.E = f.E - eflux;
    return f;
}

// The Euler flux of ONE state through S (the open boundary kinds): KG with
// A == B, so a uniform state crosses every face with the same expression.
template <class T, class G>
DNS_HD inline FvFlux<T> fvEulerFlux(const FvPrim<T>& b, const G S[3])
{
    const T Un = b.u[0] * T(S[0]) + b.u[1] * T(S[1]) + b.u[2] * T(S[2]);
    return {b.rho * Un,
            {b.rho * Un * b.u[0] + b.p * T(S[0]), b.rho * Un * b.u[1] + b.p * T(S[1]),
             b.rho * Un * b.u[2] + b.p * T(S[2])},
            b.rho * Un * b.e + b.p * Un};
}

// Boundary kinds of the FV path (masterplan §2.3). Cyclic is an internal
// face; Empty carries no flux (the 2D restriction).
enum class FvBcKind : std::uint8_t { Wall, Slip, Inflow, Outflow, Far, Empty };

// One patch's boundary condition. Trivially copyable (a device copy is a
// memcpy). Only the members of `kind` are read.
struct FvBc
{
    FvBcKind kind = FvBcKind::Wall;
    // Wall: impermeable no-slip, u_w(x) = uWall + omega x (x - origin) at the
    // face centre; adiabatic (zero heat flux, EXACTLY) unless isothermal at tWall.
    bool isothermal = false;
    double tWall = 0;
    double uWall[3] = {0, 0, 0}, omega[3] = {0, 0, 0}, origin[3] = {0, 0, 0};
    // Inflow: velocity and temperature given; p from the outgoing acoustic
    // invariant p + rho c u_n of the owner. No viscous flux.
    double uIn[3] = {0, 0, 0}, tIn = 0;
    // Outflow: static pressure given; u, T zero-gradient. Backflow (u.S < 0)
    // keeps only the normal velocity and takes tBack. No viscous flux.
    double pOut = 0, tBack = 0;
    // Far: the freestream as a CONSERVED state (rho, rho u, E), so its
    // primitives come out of fvPrimOf exactly as an interior cell's do and a
    // freestream crosses the face bit for bit. No viscous flux.
    double qFar[5] = {1, 0, 0, 0, 0};
};

// Wall velocity at a point. omega x (x - origin).
template <class G>
DNS_HD inline void fvWallVelocity(const FvBc& bc, const G x[3], double u[3])
{
    const double r[3] = {double(x[0]) - bc.origin[0], double(x[1]) - bc.origin[1],
                         double(x[2]) - bc.origin[2]};
    u[0] = bc.uWall[0] + (bc.omega[1] * r[2] - bc.omega[2] * r[1]);
    u[1] = bc.uWall[1] + (bc.omega[2] * r[0] - bc.omega[0] * r[2]);
    u[2] = bc.uWall[2] + (bc.omega[0] * r[1] - bc.omega[1] * r[0]);
}

// The boundary state of face (S, Cf) of owner P: what the boundary flux and
// the gradient's boundary face values read.
//   Wall:    rho_P, u_w(Cf), p_P (zero gradient, no two-sample
//            extrapolation), T_w (isothermal) or T_P (adiabatic).
//   Slip:    P with the normal velocity removed.
//   Inflow:  u_in, T_in, p = p_P + rho_P c_P (u_n,P - u_n,in).
//   Outflow: p_out, u_P and T_P; backflow: normal u only, T_back.
//   Far:     1D characteristic state between P and the freestream F (Riemann
//            invariants u_n +- 2c/(gamma-1); entropy and tangential velocity
//            from the upwind side; supersonic: all from the upwind side),
//            written as DIFFERENCES from the upwind state so P == F returns
//            F exactly.
// Pre: P.rho, P.p > 0; S nonzero.
template <class T, class G>
DNS_HD inline FvPrim<T> fvBoundaryState(const FvBc& bc, const FvPrim<T>& P, const G S[3],
                                        const G Cf[3], double gamma, double Rgas)
{
    using std::pow;
    using std::sqrt;
    const double S2 = double(S[0]) * double(S[0]) + double(S[1]) * double(S[1]) +
                      double(S[2]) * double(S[2]);
    const double magS = sqrt(S2);
    const T n[3] = {T(double(S[0]) / magS), T(double(S[1]) / magS), T(double(S[2]) / magS)};
    auto un = [&](const T u[3]) { return u[0] * n[0] + u[1] * n[1] + u[2] * n[2]; };
    auto ke = [](const T u[3]) { return T(0.5) * (u[0] * u[0] + u[1] * u[1] + u[2] * u[2]); };
    FvPrim<T> b = P;
    switch (bc.kind) {
    case FvBcKind::Wall: {
        double uw[3];
        fvWallVelocity(bc, Cf, uw);
        for (int k = 0; k < 3; ++k) b.u[k] = T(uw[k]);
        if (bc.isothermal) b.temp = T(bc.tWall);
        return b;
    }
    case FvBcKind::Slip: {
        const T unP = un(P.u);
        for (int k = 0; k < 3; ++k) b.u[k] = P.u[k] - unP * n[k];
        return b;
    }
    case FvBcKind::Inflow: {
        const T cP = sqrt(T(gamma) * P.p / P.rho);
        const T uin[3] = {T(bc.uIn[0]), T(bc.uIn[1]), T(bc.uIn[2])};
        b.p = P.p + P.rho * cP * (un(P.u) - un(uin));
        b.temp = T(bc.tIn);
        b.rho = b.p / (T(Rgas) * b.temp);
        for (int k = 0; k < 3; ++k) b.u[k] = uin[k];
        b.e = b.p / (T(gamma - 1.0) * b.rho) + ke(b.u);
        return b;
    }
    case FvBcKind::Outflow: {
        if (value(un(P.u)) < 0.0) {
            const T unP = un(P.u);
            for (int k = 0; k < 3; ++k) b.u[k] = unP * n[k];
            b.temp = T(bc.tBack);
        }
        b.p = T(bc.pOut);
        b.rho = b.p / (T(Rgas) * b.temp);
        b.e = b.p / (T(gamma - 1.0) * b.rho) + ke(b.u);
        return b;
    }
    case FvBcKind::Far: {
        const FvPrim<T> F = fvPrimOf(T(bc.qFar[0]), T(bc.qFar[1]), T(bc.qFar[2]), T(bc.qFar[3]),
                                     T(bc.qFar[4]), gamma, Rgas);
        const T cP = sqrt(T(gamma) * P.p / P.rho), cF = sqrt(T(gamma) * F.p / F.rho);
        const T unP = un(P.u), unF = un(F.u);
        if (value(unP) >= value(cP)) return P; // supersonic outflow
        if (value(unP) <= -value(cP)) return F; // supersonic inflow
        const T delta = (unP - unF) + T(2.0 / (gamma - 1.0)) * (cP - cF);
        const T unb = unF + T(0.5) * delta;
        const T cb = cF + T(0.25 * (gamma - 1.0)) * delta;
        const FvPrim<T>& up = value(unb) >= 0.0 ? P : F;
        const T unUp = value(unb) >= 0.0 ? unP : unF;
        const T cUp = value(unb) >= 0.0 ? cP : cF;
        const T r = cb / cUp;
        b.rho = up.rho * pow(r, 2.0 / (gamma - 1.0));
        b.p = up.p * pow(r, 2.0 * gamma / (gamma - 1.0));
        for (int k = 0; k < 3; ++k) b.u[k] = up.u[k] + (unb - unUp) * n[k];
        b.temp = b.p / (b.rho * T(Rgas));
        b.e = up.e + (b.p / b.rho - up.p / up.rho) / T(gamma - 1.0) + (ke(b.u) - ke(up.u));
        return b;
    }
    case FvBcKind::Empty: return b;
    }
    return b;
}

// The boundary face flux, given the owner P, its boundary state B
// (fvBoundaryState) and, for Wall/Slip, the face gradients (fvFaceGradBnd of
// u, v, w toward B.u and of T toward B.temp) with the caller's face mu,
// kappa:
//   Wall:  {0, p_P S - tau.S, p_P u_w.S - u_w.(tau.S) - q}, q = kappa grad T.S
//          isothermal, EXACTLY 0 adiabatic: no leak by construction (G7).
//   Slip:  {0, p_P S - (n.tau.S) n, 0}: zero shear, zero heat, no work.
//   Inflow, Outflow, Far: the Euler flux of B (no viscous flux).
//   Empty: zero.
template <class T, class G>
DNS_HD inline FvFlux<T> fvBoundaryFlux(const FvBc& bc, const FvPrim<T>& P, const FvPrim<T>& B,
                                       const G S[3], const T gu[3][3], const T gT[3], T mu,
                                       T kappa)
{
    const T zero = T(0.0);
    T ts[3] = {zero, zero, zero};
    switch (bc.kind) {
    case FvBcKind::Wall: {
        if (value(mu) != 0.0) fvTraction(gu, mu, S, ts);
        const T q = bc.isothermal
                        ? kappa * (gT[0] * T(S[0]) + gT[1] * T(S[1]) + gT[2] * T(S[2]))
                        : zero;
        const T uwS = B.u[0] * T(S[0]) + B.u[1] * T(S[1]) + B.u[2] * T(S[2]);
        const T work = B.u[0] * ts[0] + B.u[1] * ts[1] + B.u[2] * ts[2];
        return {zero,
                {P.p * T(S[0]) - ts[0], P.p * T(S[1]) - ts[1], P.p * T(S[2]) - ts[2]},
                P.p * uwS - work - q};
    }
    case FvBcKind::Slip: {
        if (value(mu) != 0.0) fvTraction(gu, mu, S, ts);
        const double S2 = double(S[0]) * double(S[0]) + double(S[1]) * double(S[1]) +
                          double(S[2]) * double(S[2]);
        const T tn = (ts[0] * T(S[0]) + ts[1] * T(S[1]) + ts[2] * T(S[2])) / T(S2);
        return {zero,
                {(P.p - tn) * T(S[0]), (P.p - tn) * T(S[1]), (P.p - tn) * T(S[2])},
                zero};
    }
    case FvBcKind::Inflow:
    case FvBcKind::Outflow:
    case FvBcKind::Far: return fvEulerFlux(B, S);
    case FvBcKind::Empty: break;
    }
    return {zero, {zero, zero, zero}, zero};
}

} // namespace dns
