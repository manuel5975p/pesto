// Device-inline compressible face fluxes (3D; nz==1 degenerates exactly to
// 2D) shared by the solver rhs and the ghost-wall surface-force tally.
// GPU-internal header (HIP TUs only).
#pragma once

#include "dns/gpu/index.hpp"

namespace dns::gpu {

struct Flux
{
    Real rho, mx, my, mz, E;
};

struct PrimView
{
    const Real *rho, *u, *v, *w, *e, *p, *T;
};

// Total Kennedy-Gruber convective minus viscous flux through the +axis face
// of cell (i,j,k); identical expression from both sides of a face keeps the
// scheme exactly conservative.
// nut (optional, per-cell dynamic eddy viscosity for LES): when non-null the
// face-averaged value is added to mu and, scaled by kapScale = cp/Pr_t, to
// kappa. nut==nullptr reproduces the molecular DNS flux bit-for-bit, so the
// solver rhs (LES off) and the ghost-wall force tally (molecular stress only)
// are unchanged.
__device__ inline Flux faceFlux(const GpuGrid& g, Real mu, Real kappa,
                                PrimView q, int i, int j, int k, int axis,
                                const Real* nut = nullptr,
                                Real kapScale = Real(0))
{
    const std::size_t a = gidx(g, i, j, k);
    const int ox = axis == 0, oy = axis == 1, oz = axis == 2;
    const int i2 = i + ox, j2 = j + oy, k2 = k + oz;
    const std::size_t b = gidx(g, i2, j2, k2);
    const Real h = axis == 0 ? g.hx : axis == 1 ? g.hy : g.hz;

    const Real rm = Real(0.5) * (q.rho[a] + q.rho[b]);
    const Real um[3] = {Real(0.5) * (q.u[a] + q.u[b]),
                        Real(0.5) * (q.v[a] + q.v[b]),
                        Real(0.5) * (q.w[a] + q.w[b])};
    const Real em = Real(0.5) * (q.e[a] + q.e[b]);
    const Real pm = Real(0.5) * (q.p[a] + q.p[b]);
    const Real un = um[axis];
    Flux f{rm * un, rm * un * um[0], rm * un * um[1], rm * un * um[2],
           rm * un * em + pm * un};
    (axis == 0 ? f.mx : axis == 1 ? f.my : f.mz) += pm;
    Real muf = mu, kapf = kappa;
    if (nut) {
        const Real nf = Real(0.5) * (nut[a] + nut[b]);
        muf += nf;
        kapf += nf * kapScale;
    }
    if (muf == Real(0)) return f;

    // du[c][d] = d u_c / d x_d at the face: compact along the normal,
    // averaged central along the tangents.
    const Real* uc[3] = {q.u, q.v, q.w};
    Real du[3][3];
    for (int c = 0; c < 3; ++c) {
        du[c][axis] = (uc[c][b] - uc[c][a]) / h;
        for (int d = 0; d < 3; ++d) {
            if (d == axis) continue;
            const int px = d == 0, py = d == 1, pz = d == 2;
            const Real hd = d == 0 ? g.hx : d == 1 ? g.hy : g.hz;
            du[c][d] = Real(0.25) *
                (uc[c][gidx(g, i + px, j + py, k + pz)] -
                 uc[c][gidx(g, i - px, j - py, k - pz)] +
                 uc[c][gidx(g, i2 + px, j2 + py, k2 + pz)] -
                 uc[c][gidx(g, i2 - px, j2 - py, k2 - pz)]) / hd;
        }
    }
    const Real divu = du[0][0] + du[1][1] + du[2][2];
    const Real dTn = (q.T[b] - q.T[a]) / h;
    Real eflux = kapf * dTn;
    Real* fm[3] = {&f.mx, &f.my, &f.mz};
    for (int c = 0; c < 3; ++c) {
        const Real tau = muf * (du[c][axis] + du[axis][c]) -
                         (c == axis ? Real(2.0 / 3.0) * muf * divu : Real(0));
        *fm[c] -= tau;
        eflux += um[c] * tau;
    }
    f.E -= eflux;
    return f;
}

} // namespace dns::gpu
