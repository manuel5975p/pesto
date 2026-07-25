// Device-inline compressible face fluxes (3D; nz==1 degenerates exactly to
// 2D) shared by the solver rhs and the ghost-wall surface-force tally.
// GPU-internal header (HIP TUs only).
#pragma once

#include "dns/gpu/index.hpp"

#include <cmath>

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

// Device pointer table of the curvilinear metrics (dns/curvmesh.hpp), one
// entry per cell: face area vectors, face inverse Jacobians, 1/volume.
struct CurvView
{
    const Real* S[3][3];  // S[a][c]: component c of the +a-face area vector
    const Real* Ji[3][9]; // Ji[a][d*3+c]: d xi_d / d x_c at the +a face
    const Real* invVol;
};

// d u_c / d x_e at the CENTRE of cell (i,j,k) on a curvilinear mesh, for the
// SGS operators. Index-space central differences (dxi = 1) contracted with the
// cell-centre inverse Jacobian; row d of dXi/dx is the average of the two
// faces normal to axis d, which is where CurvMetrics stores it. On the
// identity map this is exactly the uniform (u[+1] - u[-1]) / (2 h), and at
// nz == 1 the k stencil wraps onto itself so every z column is exactly zero.
// `vel(cell, comp)` supplies the velocity (primitives or m/rho).
template <class T, class Vel>
__host__ __device__ inline void curvVelGradCentre(const GpuGrid& g,
                                                  const CurvView& m, int i,
                                                  int j, int k, Vel vel,
                                                  T gd[3][3])
{
    const std::size_t nb[3][2] = {
        {gidx(g, i + 1, j, k), gidx(g, i - 1, j, k)},
        {gidx(g, i, j + 1, k), gidx(g, i, j - 1, k)},
        {gidx(g, i, j, k + 1), gidx(g, i, j, k - 1)}};
    T dxi[3][3]; // dxi[c][d] = d vel_c / d xi_d
    for (int c = 0; c < 3; ++c)
        for (int d = 0; d < 3; ++d)
            dxi[c][d] = T(0.5) * (T(vel(nb[d][0], c)) - T(vel(nb[d][1], c)));
    const std::size_t here = gidx(g, i, j, k);
    T Jc[3][3]; // Jc[d][e] = d xi_d / d x_e at the cell centre
    for (int d = 0; d < 3; ++d) {
        const std::size_t back =
            gidx(g, i - (d == 0), j - (d == 1), k - (d == 2));
        for (int e = 0; e < 3; ++e)
            Jc[d][e] = T(0.5) * (T(m.Ji[d][d * 3 + e][here]) +
                                 T(m.Ji[d][d * 3 + e][back]));
    }
    for (int c = 0; c < 3; ++c)
        for (int e = 0; e < 3; ++e)
            gd[c][e] = dxi[c][0] * Jc[0][e] + dxi[c][1] * Jc[1][e] +
                       dxi[c][2] * Jc[2][e];
}

// SGS filter width squared of a curvilinear cell: the isotropic (dx dy dz)^(2/3)
// written through the exact cell volume, and vol/hz at nz == 1 (the unit
// z-depth carries no filter width). Both reduce to the uniform-mesh Delta^2 of
// DeviceComp's constructor when the map is the identity.
__host__ __device__ inline Real curvDelta2(const GpuGrid& g, Real invVol)
{
    const double vol = 1.0 / double(invVol);
    if (g.nz > 1) {
        const double c = cbrt(vol);
        return Real(c * c);
    }
    return Real(vol / double(g.hz));
}

// Curvilinear counterpart of faceFlux: Un = u_avg . S, viscous gradients in
// index space mapped through the face inverse Jacobian, flux carries the
// face area (the rhs divides by the cell volume, not the spacing).
// nut (optional, per-cell dynamic eddy viscosity for LES) enters exactly as in
// faceFlux: face-averaged into mu, and scaled by kapScale = cp/Pr_t into
// kappa. nut == nullptr reproduces the molecular curvilinear flux bit-for-bit.
__device__ inline Flux faceFluxCurv(const GpuGrid& g, Real mu, Real kappa,
                                    PrimView q, CurvView m, int i, int j,
                                    int k, int axis,
                                    const Real* nut = nullptr,
                                    Real kapScale = Real(0))
{
    const std::size_t a = gidx(g, i, j, k);
    const int ox = axis == 0, oy = axis == 1, oz = axis == 2;
    const int i2 = i + ox, j2 = j + oy, k2 = k + oz;
    const std::size_t b = gidx(g, i2, j2, k2);
    const Real Sv[3] = {m.S[axis][0][a], m.S[axis][1][a], m.S[axis][2][a]};

    const Real rm = Real(0.5) * (q.rho[a] + q.rho[b]);
    const Real um[3] = {Real(0.5) * (q.u[a] + q.u[b]),
                        Real(0.5) * (q.v[a] + q.v[b]),
                        Real(0.5) * (q.w[a] + q.w[b])};
    const Real em = Real(0.5) * (q.e[a] + q.e[b]);
    const Real pm = Real(0.5) * (q.p[a] + q.p[b]);
    const Real Un = um[0] * Sv[0] + um[1] * Sv[1] + um[2] * Sv[2];
    Flux f{rm * Un, rm * Un * um[0] + pm * Sv[0], rm * Un * um[1] + pm * Sv[1],
           rm * Un * um[2] + pm * Sv[2], rm * Un * em + pm * Un};
    Real muf = mu, kapf = kappa;
    if (nut) {
        const Real nf = Real(0.5) * (nut[a] + nut[b]);
        muf += nf;
        kapf += nf * kapScale;
    }
    if (muf == Real(0)) return f;

    // d{u,v,w,T}/dxi at the face (dxi = 1): normal compact, tangential the
    // same two-cell average of central differences as the Cartesian flux.
    const Real* fld[4] = {q.u, q.v, q.w, q.T};
    Real dxi[4][3];
    for (int c = 0; c < 4; ++c) {
        dxi[c][axis] = fld[c][b] - fld[c][a];
        for (int d = 0; d < 3; ++d) {
            if (d == axis) continue;
            const int px = d == 0, py = d == 1, pz = d == 2;
            dxi[c][d] = Real(0.25) *
                (fld[c][gidx(g, i + px, j + py, k + pz)] -
                 fld[c][gidx(g, i - px, j - py, k - pz)] +
                 fld[c][gidx(g, i2 + px, j2 + py, k2 + pz)] -
                 fld[c][gidx(g, i2 - px, j2 - py, k2 - pz)]);
        }
    }
    Real gr[4][3]; // d{u,v,w,T}/dx_c
    for (int c = 0; c < 4; ++c)
        for (int cx = 0; cx < 3; ++cx)
            gr[c][cx] = dxi[c][0] * m.Ji[axis][0 + cx][a] +
                        dxi[c][1] * m.Ji[axis][3 + cx][a] +
                        dxi[c][2] * m.Ji[axis][6 + cx][a];
    const Real divu = gr[0][0] + gr[1][1] + gr[2][2];
    Real eflux = kapf * (gr[3][0] * Sv[0] + gr[3][1] * Sv[1] + gr[3][2] * Sv[2]);
    Real* fm[3] = {&f.mx, &f.my, &f.mz};
    for (int c = 0; c < 3; ++c) {
        Real ts = Real(0); // (tau . S)_c
        for (int d = 0; d < 3; ++d)
            ts += (muf * (gr[c][d] + gr[d][c]) -
                   (c == d ? Real(2.0 / 3.0) * muf * divu : Real(0))) * Sv[d];
        *fm[c] -= ts;
        eflux += um[c] * ts;
    }
    f.E -= eflux;
    return f;
}

} // namespace dns::gpu
