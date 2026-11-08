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
// the advected frame area flux wS; species ride F[0] (fvFaceY, fvSpeciesFlux),
// RANS scalars later.
#pragma once

#include "dns/chem.hpp"
#include "dns/dual.hpp"
#include "dns/hd.hpp"
#include "dns/shockflux.hpp"
#include "dns/transport.hpp"

#include <cmath>
#include <cstdint>

namespace dns {

// rho, rho u, rho v, rho w, E: the mean-flow rows of the FV state [nV][nC].
inline constexpr int kFvMeanVars = 5;
// Species rows the FV state can carry after the mean-flow rows.
inline constexpr int kFvMaxSpecies = 4;

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

// Largest diffusivity on mu/rho the FV checkerboard sees: energy gamma/Pr, momentum
// 4/3 (the normal stress of a longitudinal checkerboard; 1 grew x1.8e6 in 60 steps at
// Pr 2 on 100:1 cells, E6). FV-only: the structured solvers keep diffusivityFactor.
// Pr > 0: the molecular Pr, or Pr_t for the eddy part.
DNS_HD inline double fvDiffusivityFactor(double gamma, double Pr)
{
    const double a = 4.0 / 3.0, b = gamma / Pr;
    return a > b ? a : b;
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

// Monotonized-central slope, gpu/internal/species_face.hpp's mcSlope for any T:
// 0 unless d1, d2 share a sign, else sign * min(2|d1|, 2|d2|, |d1 + d2|/2).
// Exactly 0 on a uniform field.
template <class T>
DNS_HD inline T fvMcSlope(T d1, T d2)
{
    if (!(value(d1 * d2) > 0.0)) return T(0.0);
    const T a1 = value(d1) > 0.0 ? d1 : T(0.0) - d1, a2 = value(d2) > 0.0 ? d2 : T(0.0) - d2;
    const T s = d1 + d2;
    const T a3 = T(0.5) * (value(s) > 0.0 ? s : T(0.0) - s);
    T m = T(2.0) * a1;
    const T m2 = T(2.0) * a2;
    if (value(m2) < value(m)) m = m2;
    if (value(a3) < value(m)) m = a3;
    return value(d1) > 0.0 ? m : T(0.0) - m;
}

// Species face value of an internal face: upwind on mdot, Y_up + MC(d1, d2)/2
// with d2 = Y_down - Y_up and d1 the VIRTUAL upwind difference 2 grad Y_up . r -
// d2, r = up -> down (Darwish-Moukalled). On a uniform mesh LSQ's grad Y_up . r
// is (Y_down - Y_upup)/2, so d1 = Y_up - Y_upup: the structured faceY (rounding,
// docs/fvcomp.md "Species"). d, gP, gN as fvFaceGrad's. Uniform Y returns Y bit
// for bit (g == 0, d1 == d2 == 0).
template <class T, class G>
DNS_HD inline T fvFaceY(T mdot, T YP, T YN, const T gP[3], const T gN[3], const G d[3])
{
    if (value(mdot) >= 0.0) {
        const T d2 = YN - YP;
        const T d1 = T(2.0) * (gP[0] * T(d[0]) + gP[1] * T(d[1]) + gP[2] * T(d[2])) - d2;
        return YP + T(0.5) * fvMcSlope(d1, d2);
    }
    const T d2 = YP - YN;
    const T d1 = T(-2.0) * (gN[0] * T(d[0]) + gN[1] * T(d[1]) + gN[2] * T(d[2])) - d2;
    return YN + T(0.5) * fvMcSlope(d1, d2);
}

// A species votes on the shared limiter iff |d2| >= 2^-24 (the FP32 resolution of a
// unit mass fraction): gpu/internal/species_face.hpp's kVoteJump, its reason.
inline constexpr double kFvVoteJump = 0x1p-24;

// One field's vote on the shared limiter of an internal face: min(th, MC(d1, d2)/
// d2), (d1, d2) fvFaceY's. Start from th = 2 (none votes), fold over the fields
// (species; the low-Mach face adds T, so T + (q/cp) Y keeps one face formula).
template <class T, class G>
DNS_HD inline T fvThetaVote(T th, T mdot, T YP, T YN, const T gP[3], const T gN[3], const G d[3])
{
    const bool up = value(mdot) >= 0.0;
    const T d2 = up ? YN - YP : YP - YN;
    const T s = up ? T(2.0) * (gP[0] * T(d[0]) + gP[1] * T(d[1]) + gP[2] * T(d[2]))
                   : T(-2.0) * (gN[0] * T(d[0]) + gN[1] * T(d[1]) + gN[2] * T(d[2]));
    const double a = value(d2) > 0.0 ? value(d2) : -value(d2);
    if (a < kFvVoteJump) return th;
    const T r = fvMcSlope(s - d2, d2) / d2;
    return value(r) < value(th) ? r : th;
}

// The shared-limiter face value Y_up + th/2 (Y_down - Y_up): LINEAR in Y across
// species, so every affine combination (coupling functions, mixture fraction)
// rides the same formula whatever th is.
template <class T>
DNS_HD inline T fvFaceYShared(T mdot, T YP, T YN, T th)
{
    return value(mdot) >= 0.0 ? YP + T(0.5) * th * (YN - YP) : YN + T(0.5) * th * (YP - YN);
}

// Species flux through S on an internal face: the face's own mass flux times
// the face value, minus Fick with rho D, g the corrected face gradient.
template <class T, class G>
DNS_HD inline T fvSpeciesFlux(T mdot, T Yf, T rhoD, const T g[3], const G S[3])
{
    return mdot * Yf - rhoD * (g[0] * T(S[0]) + g[1] * T(S[1]) + g[2] * T(S[2]));
}

// Species rho D of an internal face under LES (docs/fvcomp.md "Species LES and
// TFLES"): gpu/species.hip effRhoD<Eddy> restated, rho D + nut_f / Sc_t, nut_f the
// face mean of mu_t (the mean flux's own), invSct = 1/Pr_t (Sc_t = Pr_t).
template <class T>
DNS_HD inline T fvEddyRhoD(T rhoD, T nutf, T invSct)
{
    return rhoD + nutf * invSct;
}

// The thickened flame's (effRhoD<Thick>): FE_f rho D, plus (1 - Om_f) nut_f / Sc_t
// with LES (no SGS flux where the sensor thickens). FE_f, Om_f arithmetic face means.
template <class T>
DNS_HD inline T fvThickRhoD(T rhoD, T FEf, T Omf, T nutf, T invSct, bool les)
{
    T r = FEf * rhoD;
    if (les) r = r + (T(1.0) - Omf) * nutf * invSct;
    return r;
}

// The energy flux the mean face flux lacks under TFLES (thickEnergyFlux restated):
// the conductivity becomes FE kappa + (1 - Om) mu_t cp/Pr_t, so add
//   -(FE_f - 1) kappa_f g_T.S + Om_f nut_f (cp/Pr_t) g_T.S   [LES term with les]
// to F_E: additive, conservative, the mean flux untouched. kappa_f: the face's own
// molecular value (its Sutherland ratio), gTS the corrected face gradient of T . S.
template <class T>
DNS_HD inline T fvThickEnergyFlux(T FEf, T Omf, T nutf, T kapf, T kapLes, T gTS, bool les)
{
    T fl = (T(0.0) - (FEf - T(1.0))) * kapf * gTS;
    if (les) fl = fl + Omf * nutf * kapLes * gTS;
    return fl;
}

// Boundary kinds of the FV path (masterplan §2.3). Cyclic is an internal
// face; Empty carries no flux (the 2D restriction). Total: OpenFOAM's inlet
// fixed point (docs/nozzle.md).
enum class FvBcKind : std::uint8_t { Wall, Slip, Inflow, Outflow, Far, Empty, Total };

// What an Outflow's state is (docs/nozzle.md): Fixed the shipped row, Extrapolate
// zeroGradient, Riemann the half-Riemann problem of the owner against pOut.
enum class FvExit : std::uint8_t { Fixed, Extrapolate, Riemann };

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
    // Inflow: a fixed FACE: u, T and Y diffuse through it toward the row (OpenFOAM's
    // fixedValue; the jet's sheet sends 5% of its fuel in by Fick, docs/fvcomp.md).
    bool diffusive = false;
    // Outflow: static pressure given; u, T zero-gradient. Backflow (u.S < 0)
    // keeps only the normal velocity and takes tBack. No viscous flux.
    double pOut = 0, tBack = 0;
    // Outflow: subsonic characteristics while the flow leaves: entropy and the outgoing
    // Riemann invariant of the owner, rho and u_n follow p_out. Off: rho_b = p_out/(R
    // T_P), u_P: a cell below p_out is drained at p_out's density (a jet's outflow
    // evacuated on a sheared mesh, docs/fvcomp.md "Diffusion flame").
    bool characteristic = false;
    // Far: the freestream as a CONSERVED state (rho, rho u, E), so its
    // primitives come out of fvPrimOf exactly as an interior cell's do and a
    // freestream crosses the face bit for bit. No viscous flux.
    double qFar[5] = {1, 0, 0, 0, 0};
    // Inflow, Outflow, Far: the mass fractions an INCOMING face carries; an
    // outgoing face carries the owner's (fvBoundaryY). Other kinds: unread.
    double y[kFvMaxSpecies] = {0, 0, 0, 0};
    // Outflow: a backflow face carries the owner's Y too, never y (the jet's outflow:
    // docs/chem.md item 4, y would be the jet's fuel).
    bool ownY = false;
    // Total: stagnation pressure and temperature (fvBoundaryState). No viscous flux.
    double pTotal = 0, tTotal = 0;
    // Outflow: the state rule; Fixed reads the members above bit for bit.
    FvExit exit = FvExit::Fixed;
    // Slip: a `wedge` side, read by the tables only (entS, diffCoef); no kernel reads it.
    bool wedge = false;
};

// The species rows and the one irreversible step of model C0 (docs/chem.md):
// omega = arrheniusRate(lnA, Ta, rho, Y_fuel, T) consumes species `fuel` and
// releases q omega into E; other species are passive. Le = 1: rho D = invPr mu.
// nY == 0: no species row, no species pass, the solver bit for bit.
// lnA == -inf: no reaction (omega exactly 0).
// ox >= 0: model C1, F + s O -> (1+s) P: omega = arrheniusRate2, oxidizer -stoich
// omega on the same omega, one shared limiter per face (fvThetaVote). ox < 0: off.
// law.n > 0 (ox >= 0 only): model C2b, ln A + phiRateLn(law, Y_F, Y_O) per cell (the
// structured kChemCellPass2Phi / ChemRate2Phi); n == 0: the C1 rate bit for bit.
struct FvChem
{
    int nY = 0, fuel = 0;
    double lnA = 0, Ta = 0, q = 0, invPr = 1;
    int ox = -1;
    double stoich = 1;
    PhiRateLaw<double> law{};
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

// The species value of a boundary face: the BC row's y[s] where the open face's
// mass flux mdot comes in, the owner's elsewhere. So a boundary species flux is
// mdot Y_b with no diffusion: a Wall or Slip has mdot 0 and zero gradient, an
// open face no viscous flux (the structured ScalarBoundary rule). A `diffusive`
// Inflow: the row's on every face, plus Fick (fvSpeciesFlux of fvFaceGradBnd).
template <class T>
DNS_HD inline T fvBoundaryY(const FvBc& bc, int s, T YP, T mdot)
{
    const bool open = bc.kind == FvBcKind::Inflow || bc.kind == FvBcKind::Outflow || bc.kind == FvBcKind::Far ||
                      bc.kind == FvBcKind::Total;
    if (bc.diffusive) return T(bc.y[s]);
    return open && !bc.ownY && value(mdot) < 0.0 ? T(bc.y[s]) : YP;
}

// The half-Riemann problem of owner P against the ambient pressure a = bc.pOut, sampled
// at the face, n outward (Toro's star state with the right wave absent): the exit of a
// nozzle, per face, so a subsonic layer under a supersonic core takes its own branch.
//   a >= p: shock. Leaves (S >= 0): P. Enters: rho*, u*, p = a, tangential of P.
//   a < p:  fan. Supersonic: P. Subsonic: isentropic at a. Transonic: the sonic state.
//   u* < 0: backflow, p = a, T = bc.tBack, normal velocity u*.
// Continuous across every branch but the contact at u* = 0. Pre: P.rho, P.p, a > 0.
template <class T>
DNS_HD inline FvPrim<T> fvExitState(const FvBc& bc, const FvPrim<T>& P, const T n[3], double gamma, double Rgas)
{
    using std::pow;
    using std::sqrt;
    const T a = T(bc.pOut), un = P.u[0] * n[0] + P.u[1] * n[1] + P.u[2] * n[2];
    const T c = sqrt(T(gamma) * P.p / P.rho);
    const double m = (gamma - 1.0) / (gamma + 1.0), gm1 = gamma - 1.0;
    auto stateAt = [&](T rho, T p, T unb) {
        FvPrim<T> b = P;
        b.rho = rho;
        b.p = p;
        for (int k = 0; k < 3; ++k) b.u[k] = P.u[k] + (unb - un) * n[k];
        b.temp = p / (rho * T(Rgas));
        const T ke = T(0.5) * (b.u[0] * b.u[0] + b.u[1] * b.u[1] + b.u[2] * b.u[2]);
        b.e = p / (T(gm1) * rho) + ke;
        return b;
    };
    auto backflow = [&](T us) {
        FvPrim<T> b = P;
        for (int k = 0; k < 3; ++k) b.u[k] = us * n[k];
        b.temp = T(bc.tBack);
        b.p = a;
        b.rho = a / (T(Rgas) * b.temp);
        const T ke = T(0.5) * (b.u[0] * b.u[0] + b.u[1] * b.u[1] + b.u[2] * b.u[2]);
        b.e = a / (T(gm1) * b.rho) + ke;
        return b;
    };
    const T r = a / P.p;
    if (value(r) >= 1.0) {
        const T S = un - c * sqrt(T((gamma + 1.0) / (2.0 * gamma)) * r + T(gm1 / (2.0 * gamma)));
        if (value(S) >= 0.0) return P;
        const T us = un - (a - P.p) * sqrt(T(2.0 / (gamma + 1.0)) / (P.rho * (a + T(m) * P.p)));
        if (value(us) < 0.0) return backflow(us);
        return stateAt(P.rho * (r + T(m)) / (T(m) * r + T(1.0)), a, us);
    }
    if (value(un) - value(c) >= 0.0) return P;
    const T k = pow(r, gm1 / (2.0 * gamma));
    const T us = un + T(2.0 / gm1) * c * (T(1.0) - k), cs = c * k;
    if (value(us) < 0.0) return backflow(us);
    if (value(us) - value(cs) <= 0.0) return stateAt(P.rho * pow(r, 1.0 / gamma), a, us);
    const T cq = T(m) * (un + T(2.0 / gm1) * c), q = cq / c;
    return stateAt(P.rho * pow(q, 2.0 / gm1), P.p * pow(q, 2.0 * gamma / gm1), cq);
}

// The boundary state of face (S, Cf) of owner P: what the boundary flux and
// the gradient's boundary face values read.
//   Wall:    rho_P, u_w(Cf), p_P (zero gradient, no two-sample
//            extrapolation), T_w (isothermal) or T_P (adiabatic).
//   Slip:    P with the normal velocity removed.
//   Inflow:  u_in, T_in, p = p_P + rho_P c_P (u_n,P - u_n,in).
//   Outflow: p_out, u_P and T_P; backflow: normal u only, T_back. `characteristic`,
//            leaving: p_out, the owner's entropy and outgoing invariant. Extrapolate:
//            P. Riemann: fvExitState.
//   Total:   OpenFOAM's totalPressure + totalTemperature + pressureInletOutletVelocity
//            fixed point. Leaving (u_P.n >= 0): u_P, T0, p0. Entering: normal velocity
//            of P, T = T0 - |u|^2/(2 cp), p = p0 (T/T0)^(gamma/(gamma-1)).
//   Far:     1D characteristic state between P and the freestream F (Riemann
//            invariants u_n +- 2c/(gamma-1); entropy and tangential velocity
//            from the upwind side; supersonic: all from the upwind side).
//            The outgoing invariant reads the ISENTROPIC sound speed of p_P,
//            cF (p_P/p_F)^((gamma-1)/(2 gamma)), never c_P: an entropy wave
//            carries no acoustic amplitude (c_P made a hot wake a pressure
//            source; docs/fvcomp.md "Far gap"). Written as DIFFERENCES from
//            the upwind state so P == F returns F exactly.
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
        if (bc.exit == FvExit::Extrapolate) return P;
        if (bc.exit == FvExit::Riemann) return fvExitState(bc, P, n, gamma, Rgas);
        if (bc.characteristic && value(un(P.u)) >= 0.0) {
            const T r = pow(T(bc.pOut) / P.p, 1.0 / gamma), cP = sqrt(T(gamma) * P.p / P.rho);
            const T dun = T(2.0 / (gamma - 1.0)) * cP * (T(1.0) - pow(r, 0.5 * (gamma - 1.0)));
            b.p = T(bc.pOut);
            b.rho = P.rho * r;
            b.temp = b.p / (b.rho * T(Rgas));
            for (int k = 0; k < 3; ++k) b.u[k] = P.u[k] + dun * n[k];
            b.e = b.p / (T(gamma - 1.0) * b.rho) + ke(b.u);
            return b;
        }
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
        const T cPs = cF * pow(P.p / F.p, (gamma - 1.0) / (2.0 * gamma));
        const T delta = (unP - unF) + T(2.0 / (gamma - 1.0)) * (cPs - cF);
        const T unb = unF + T(0.5) * delta;
        const T cb = cF + T(0.25 * (gamma - 1.0)) * delta;
        const FvPrim<T>& up = value(unb) >= 0.0 ? P : F;
        const T unUp = value(unb) >= 0.0 ? unP : unF;
        b.p = F.p * pow(cb / cF, 2.0 * gamma / (gamma - 1.0));
        b.rho = up.rho * pow(b.p / up.p, 1.0 / gamma);
        for (int k = 0; k < 3; ++k) b.u[k] = up.u[k] + (unb - unUp) * n[k];
        b.temp = b.p / (b.rho * T(Rgas));
        b.e = up.e + (b.p / b.rho - up.p / up.rho) / T(gamma - 1.0) + (ke(b.u) - ke(up.u));
        return b;
    }
    case FvBcKind::Empty: return b;
    case FvBcKind::Total: {
        const T unP = un(P.u);
        if (value(unP) >= 0.0) {
            b.p = T(bc.pTotal);
            b.temp = T(bc.tTotal);
        } else {
            const double cp = gamma * Rgas / (gamma - 1.0);
            for (int k = 0; k < 3; ++k) b.u[k] = unP * n[k];
            b.temp = T(bc.tTotal) - ke(b.u) / T(cp);
            b.p = T(bc.pTotal) * pow(b.temp / T(bc.tTotal), gamma / (gamma - 1.0));
        }
        b.rho = b.p / (T(Rgas) * b.temp);
        b.e = b.p / (T(gamma - 1.0) * b.rho) + ke(b.u);
        return b;
    }
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
//   Inflow, Outflow, Far, Total: the Euler flux of B (no viscous flux; a `diffusive`
//   Inflow adds -tau.S, -u_b.(tau.S) - q as an internal face).
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
    case FvBcKind::Inflow: {
        FvFlux<T> f = fvEulerFlux(B, S);
        if (!bc.diffusive || value(mu) == 0.0) return f;
        fvTraction(gu, mu, S, ts);
        T eflux = kappa * (gT[0] * T(S[0]) + gT[1] * T(S[1]) + gT[2] * T(S[2]));
        for (int c = 0; c < 3; ++c) {
            f.m[c] = f.m[c] - ts[c];
            eflux = eflux + B.u[c] * ts[c];
        }
        f.E = f.E - eflux;
        return f;
    }
    case FvBcKind::Outflow:
    case FvBcKind::Far:
    case FvBcKind::Total: return fvEulerFlux(B, S);
    case FvBcKind::Empty: break;
    }
    return {zero, {zero, zero, zero}, zero};
}

// The third difference of one row along a mesh line (P-, P, N, N+), the background
// term of FV --shock hll (docs/nozzle.md): the grouping makes a uniform field exactly 0.
template <class T>
DNS_HD inline T fvD4Row(T pm, T p, T n, T np)
{
    return (np - pm) - T(3.0) * (n - p);
}

// |S| D of an internal face (docs/nozzle.md): shockFaceDelta of the two cells on S's unit
// normal, the flux F_KG + sigma (F_HLL - F_KG) minus F_KG, rows rho, m, E. Pre: sigma > 0.
template <class T, class G>
DNS_HD inline void fvShockFlux(T sigma, const FvPrim<T>& A, const FvPrim<T>& B, const G S[3], double gamma, T o[5])
{
    using std::sqrt;
    const double mag = sqrt(double(S[0]) * double(S[0]) + double(S[1]) * double(S[1]) + double(S[2]) * double(S[2]));
    const T n[3] = {T(double(S[0]) / mag), T(double(S[1]) / mag), T(double(S[2]) / mag)};
    const ShockState<T> L{A.rho, {A.u[0], A.u[1], A.u[2]}, A.p, A.e, T(gamma)};
    const ShockState<T> R{B.rho, {B.u[0], B.u[1], B.u[2]}, B.p, B.e, T(gamma)};
    const ShockDelta<T> D = shockFaceDelta(sigma, L, R, n);
    for (int k = 0; k < 5; ++k) o[k] = T(mag) * D.v[k];
}

// The rate of the background term of an internal face: eps (1 - sigma) |u_f.S|, u_f the
// face mean. Zero on a wall-normal face of a boundary layer by construction.
template <class T, class G>
DNS_HD inline T fvD4Rate(T eps, T sigma, const FvPrim<T>& A, const FvPrim<T>& B, const G S[3])
{
    using std::fabs;
    const T us = T(0.5) * ((A.u[0] + B.u[0]) * T(S[0]) + (A.u[1] + B.u[1]) * T(S[1]) + (A.u[2] + B.u[2]) * T(S[2]));
    return eps * (T(1.0) - sigma) * fabs(us);
}

// ln(a/b)/(2 xi), xi = (a - b)/(a + b): log mean L(a, b) = (a - b)/ln(a/b) =
// (a + b)/(2 fvLogFactor(a, b)). A series below xi^2 1e-2, exactly 1 at a == b
// (Ismail-Roe). Pre: a, b > 0.
template <class T>
DNS_HD inline T fvLogFactor(T a, T b)
{
    using std::log;
    const T xi = (a - b) / (a + b), u = xi * xi;
    if (value(u) < 1e-2)
        return T(1.0) + u * (T(1.0 / 3.0) + u * (T(1.0 / 5.0) + u * (T(1.0 / 7.0) + u * (T(1.0 / 9.0) + u * (T(1.0 / 11.0) + u * T(1.0 / 13.0))))));
    return log(a / b) / (T(2.0) * xi);
}

// The face density of gravity: L(p_A, p_B) / (R L(T_A, T_B)). With it p_B - p_A =
// rho_f g.d holds EXACTLY whenever T is linear in the height and ln p its
// hydrostatic integral, isothermal included: a path-independent relation, so the
// state is an equilibrium of ANY mesh (docs/fvcomp.md "Gravity"). Pre: p, T > 0.
template <class T>
DNS_HD inline T fvHydroRho(const FvPrim<T>& A, const FvPrim<T>& B, double Rgas)
{
    return (A.p + B.p) / ((A.temp + B.temp) * T(Rgas)) * fvLogFactor(A.temp, B.temp) / fvLogFactor(A.p, B.p);
}

// Gravity as a FACE term of an internal face, added by BOTH cells, unsigned (a
// source, not a flux): o[0..2] = (g.d/2) rho_f S, o[3] = (g.d/2) mdot, d = C_nei -
// C_own, mdot = F[0]. It stands against the pressure sum's own form, (p_N - p_P)/2
// S of a closed cell, so the rest residual is (p_N - p_P - rho_f g.d)/2 S, and
// sum V (E + rho phi) of a closed box telescopes. A Wall or Slip face adds nothing:
// its hydrostatic image and its share of this term are one number, opposite.
template <class T, class G>
DNS_HD inline void fvGravityFace(T rhoF, T mdot, const double g[3], const G d[3], const G S[3], T o[4])
{
    const T gd = T(0.5) * (T(g[0]) * T(d[0]) + T(g[1]) * T(d[1]) + T(g[2]) * T(d[2]));
    for (int k = 0; k < 3; ++k) o[k] = rhoF * gd * T(S[k]);
    o[3] = mdot * gd;
}

} // namespace dns
