// The flow Newton-Krylov PRECONDITIONER's face flux (docs/rans.md, "Newton-
// Krylov pseudo-time"): first-order Rusanov plus the COMPACT viscous terms,
//   F_P(qL, qR) = (F(qL) + F(qR)) . S / 2 - lambda_f (qR - qL) / 2
//                 - [tau . S, kappa grad T . S + u_f . tau . S]_compact,
// with lambda_f, mu_f, kappa_f FROZEN by the caller and the velocity and
// temperature gradients the normal index difference alone (dxi[axis] of
// faceFluxCurv; the tangential terms dropped). It is only ever the
// preconditioner: the residual is always the GPU's central KG stage, so the
// dissipation here never reaches the fixed point. Templated on the scalar
// (double, Dual<N>); host- and device-compilable, so the N2 test restates it
// by FP64 central differences on the host.
#pragma once

#include "dns/dual.hpp"
#include "dns/hd.hpp"

namespace dns::nk {

// The frozen face data: area vector, row `axis` of dxi/dx at the face, the
// Rusanov rate, the molecular + eddy viscosity and conductivity, gamma, R.
struct FaceFrozen
{
    double S[3];
    double Ji[3];   // d xi_axis / d x_c at the face (row `axis` of Ji)
    double lambda;  // max over L, R of |u . S| + c |S|
    double mu, kappa;
    double gamma, Rgas;
};

// q = (rho, rho u, rho v, rho w, E) of one side; F = the flux of 5 components.
template <class T>
DNS_HD inline void precondFlux(const T qL[5], const T qR[5],
                               const FaceFrozen& f, T F[5])
{
    auto euler = [&](const T q[5], T out[5], T u[3], T& Tt) {
        const T rho = q[0];
        u[0] = q[1] / rho;
        u[1] = q[2] / rho;
        u[2] = q[3] / rho;
        const T ke = 0.5 * (u[0] * q[1] + u[1] * q[2] + u[2] * q[3]);
        const T p = (f.gamma - 1.0) * (q[4] - ke);
        Tt = p / (rho * f.Rgas);
        const T U = u[0] * f.S[0] + u[1] * f.S[1] + u[2] * f.S[2];
        out[0] = rho * U;
        out[1] = q[1] * U + p * f.S[0];
        out[2] = q[2] * U + p * f.S[1];
        out[3] = q[3] * U + p * f.S[2];
        out[4] = (q[4] + p) * U;
    };
    T FL[5], FR[5], uL[3], uR[3], TL, TR;
    euler(qL, FL, uL, TL);
    euler(qR, FR, uR, TR);
    for (int v = 0; v < 5; ++v)
        F[v] = 0.5 * (FL[v] + FR[v]) - 0.5 * f.lambda * (qR[v] - qL[v]);
    if (f.mu == 0.0) return;
    // Compact gradients: d phi / d x_c = (phi_R - phi_L) Ji[c].
    T g[3][3];
    for (int c = 0; c < 3; ++c) {
        const T du = uR[c] - uL[c];
        for (int x = 0; x < 3; ++x) g[c][x] = du * f.Ji[x];
    }
    const T divu = g[0][0] + g[1][1] + g[2][2];
    const T dT = TR - TL;
    T eflux = f.kappa * dT * (f.Ji[0] * f.S[0] + f.Ji[1] * f.S[1] +
                              f.Ji[2] * f.S[2]);
    for (int c = 0; c < 3; ++c) {
        T ts = T(0.0);
        for (int d = 0; d < 3; ++d) {
            T tcd = f.mu * (g[c][d] + g[d][c]);
            if (c == d) tcd = tcd - (2.0 / 3.0) * f.mu * divu;
            ts = ts + tcd * f.S[d];
        }
        F[1 + c] = F[1 + c] - ts;
        eflux = eflux + 0.5 * (uL[c] + uR[c]) * ts;
    }
    F[4] = F[4] - eflux;
}

// The 5x5 derivative blocks dF/dqL and dF/dqR of precondFlux at (qL, qR) by
// forward-mode duals: row = flux component, column = conserved variable.
DNS_HD inline void precondBlocks(const double qL[5], const double qR[5],
                                 const FaceFrozen& f, double AL[5][5],
                                 double AR[5][5])
{
    Dual<10> L[5], R[5], F[5];
    for (int v = 0; v < 5; ++v) {
        L[v] = Dual<10>::var(qL[v], v);
        R[v] = Dual<10>::var(qR[v], 5 + v);
    }
    precondFlux(L, R, f, F);
    for (int r = 0; r < 5; ++r)
        for (int c = 0; c < 5; ++c) {
            AL[r][c] = F[r].d[c];
            AR[r][c] = F[r].d[5 + c];
        }
}

} // namespace dns::nk
