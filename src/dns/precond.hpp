// Weiss-Smith low-Mach preconditioning of the explicit --steady pseudo-time
// march (`--precond K`, masterplan_levers M2): the pressure rate of a cell's
// total residual is scaled by eps = U_r^2 / c^2, velocity and entropy rates
// untouched. In conservative variables that is a RANK-ONE correction
//   R' = R - ((1 - eps) / c^2) a (b^T R),
//   a = (1, u, v, w, H),  b^T = (gamma - 1)(|u|^2 / 2, -u, -v, -w, 1),
// with b^T R = dp/dtau and a dp / c^2 the isentropic, velocity-free
// direction (b^T a = c^2). P = I - ((1 - eps)/c^2) a b^T is invertible for
// eps > 0, so R' = 0 iff R = 0: the fixed point is unchanged, only the
// pseudo-time path moves. Acoustic speeds become u' +- c',
//   u' = |u_rel| (1 + eps) / 2,  c' = sqrt(((1 - eps)/2)^2 |u_rel|^2 + U_r^2),
// so at low Mach every eigenvalue is O(U_r) instead of O(c).
//   U_r^2 = min(c^2, max(K^2 U_inf^2, |u_rel|^2, (nu aov_max)^2)):
// a global floor K U_inf (stagnation points), the local speed, and a viscous
// floor nu / h (Weiss-Smith 1995; the cell Reynolds number below 1 needs it).
// Header-only, host+device (DNS_HD), so tests/test_precond.cpp checks the
// algebra on the CPU in FP64.
#pragma once

#include "dns/hd.hpp"

#include <cmath>

namespace dns {

// Largest --precond K the validator accepts (K U_inf >= c turns it off).
inline constexpr double kPrecondKMax = 10.0;

// eps = U_r^2 / c^2 in (0, 1]. kU2 = K^2 U_inf^2, urel2 = |u_rel|^2, visc2 =
// (nu aov_max)^2. Pre: c2 > 0 (a NaN c2 returns NaN; callers test the
// state's finiteness themselves).
template <class T>
DNS_HD inline T precondEps(T c2, T urel2, T kU2, T visc2)
{
    T ur2 = urel2 > kU2 ? urel2 : kU2;
    if (visc2 > ur2) ur2 = visc2;
    if (ur2 > c2) ur2 = c2;
    return ur2 / c2;
}

// R (rho, mx, my, mz, E) -> R' in place, at velocity u (ABSOLUTE: the
// conserved variables' own), total enthalpy H and sound speed^2 c2.
// eps == 1 leaves R bit for bit (the correction is a multiple of 0).
template <class T>
DNS_HD inline void precondApply(T R[5], const T u[3], T H, T gamma, T c2, T eps)
{
    const T q2 = u[0] * u[0] + u[1] * u[1] + u[2] * u[2];
    const T rp = (gamma - T(1)) * (T(0.5) * q2 * R[0] - u[0] * R[1] -
                                   u[1] * R[2] - u[2] * R[3] + R[4]);
    const T f = (T(1) - eps) / c2 * rp;
    R[0] -= f;
    R[1] -= f * u[0];
    R[2] -= f * u[1];
    R[3] -= f * u[2];
    R[4] -= f * H;
}

// The preconditioned acoustic speed |u'| + c' at |u_rel| = urel.
template <class T>
DNS_HD inline T precondWave(T urel, T c2, T eps)
{
    const T h = (T(1) - eps) * T(0.5);
    return urel * (T(1) + eps) * T(0.5) + std::sqrt(h * h * urel * urel + eps * c2);
}

} // namespace dns
