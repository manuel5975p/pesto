#include "dns/operators.hpp"

#include <algorithm>
#include <cassert>
#include <cmath>

namespace dns {

namespace {
inline double avg(double a, double b) { return 0.5 * (a + b); }
}

void divergence(const Grid& g, const VectorField& U, Scalar& out)
{
    assert(out.size() == g.size());
    for (int k = 0; k < g.nz; ++k)
        for (int j = 0; j < g.ny; ++j)
            for (int i = 0; i < g.nx; ++i)
                out[g.idx(i, j, k)] =
                    (U.u[g.idx(i + 1, j, k)] - U.u[g.idx(i, j, k)]) / g.hx
                  + (U.v[g.idx(i, j + 1, k)] - U.v[g.idx(i, j, k)]) / g.hy
                  + (U.w[g.idx(i, j, k + 1)] - U.w[g.idx(i, j, k)]) / g.hz;
}

void subtractGradient(const Grid& g, const Scalar& p, double scale, VectorField& U)
{
    assert(p.size() == g.size());
    for (int k = 0; k < g.nz; ++k)
        for (int j = 0; j < g.ny; ++j)
            for (int i = 0; i < g.nx; ++i)
            {
                const auto c = g.idx(i, j, k);
                U.u[c] -= scale * (p[c] - p[g.idx(i - 1, j, k)]) / g.hx;
                U.v[c] -= scale * (p[c] - p[g.idx(i, j - 1, k)]) / g.hy;
                U.w[c] -= scale * (p[c] - p[g.idx(i, j, k - 1)]) / g.hz;
            }
}

void laplacian(const Grid& g, const VectorField& U, VectorField& out)
{
    const double ax = 1.0 / (g.hx * g.hx), ay = 1.0 / (g.hy * g.hy), az = 1.0 / (g.hz * g.hz);
    auto lap = [&](const Scalar& q, Scalar& o) {
        for (int k = 0; k < g.nz; ++k)
            for (int j = 0; j < g.ny; ++j)
                for (int i = 0; i < g.nx; ++i)
                {
                    const auto c = g.idx(i, j, k);
                    o[c] = ax * (q[g.idx(i + 1, j, k)] - 2 * q[c] + q[g.idx(i - 1, j, k)])
                         + ay * (q[g.idx(i, j + 1, k)] - 2 * q[c] + q[g.idx(i, j - 1, k)])
                         + az * (q[g.idx(i, j, k + 1)] - 2 * q[c] + q[g.idx(i, j, k - 1)]);
                }
    };
    lap(U.u, out.u);
    lap(U.v, out.v);
    lap(U.w, out.w);
}

void convection(const Grid& g, const VectorField& U, VectorField& out)
{
    // Skew-symmetric form: divergence form d(ui uj)/dxj (Harlow-Welch, linear
    // interpolation to the flux locations) plus 0.5*ui*div(u) interpolated to
    // the face. The correction vanishes on a discretely solenoidal field and
    // makes the operator energy-conserving when the advecting field carries
    // plain-divergence errors -- which it always does at cut cells (only the
    // APERTURED divergence is projected to zero there) and, on the GPU, at the
    // FP32 projection residual. Without it the divergence form injects energy
    // ~ u^2*div(u) into under-resolved cut-cell shear layers and diverges.
    const auto& u = U.u; const auto& v = U.v; const auto& w = U.w;
    auto cellDiv = [&](int i, int j, int k) {
        return (u[g.idx(i + 1, j, k)] - u[g.idx(i, j, k)]) / g.hx
             + (v[g.idx(i, j + 1, k)] - v[g.idx(i, j, k)]) / g.hy
             + (w[g.idx(i, j, k + 1)] - w[g.idx(i, j, k)]) / g.hz;
    };

    for (int k = 0; k < g.nz; ++k)
        for (int j = 0; j < g.ny; ++j)
            for (int i = 0; i < g.nx; ++i)
            {
                const auto c = g.idx(i, j, k);
                const double divc = cellDiv(i, j, k);

                // --- x-momentum at u-face (i,j,k) ---
                {
                    // d(uu)/dx: uu at cell centres i and i-1
                    const double uc  = avg(u[c], u[g.idx(i + 1, j, k)]);
                    const double ucm = avg(u[g.idx(i - 1, j, k)], u[c]);
                    // d(uv)/dy: fluxes at xy-corners (i, j) and (i, j+1)
                    const double vjp = avg(v[g.idx(i, j + 1, k)], v[g.idx(i - 1, j + 1, k)]);
                    const double vj  = avg(v[c], v[g.idx(i - 1, j, k)]);
                    const double ujp = avg(u[c], u[g.idx(i, j + 1, k)]);
                    const double ujm = avg(u[g.idx(i, j - 1, k)], u[c]);
                    // d(uw)/dz: fluxes at xz-corners (i, k) and (i, k+1)
                    const double wkp = avg(w[g.idx(i, j, k + 1)], w[g.idx(i - 1, j, k + 1)]);
                    const double wk  = avg(w[c], w[g.idx(i - 1, j, k)]);
                    const double ukp = avg(u[c], u[g.idx(i, j, k + 1)]);
                    const double ukm = avg(u[g.idx(i, j, k - 1)], u[c]);

                    out.u[c] = -((uc * uc - ucm * ucm) / g.hx
                               + (vjp * ujp - vj * ujm) / g.hy
                               + (wkp * ukp - wk * ukm) / g.hz)
                             + 0.5 * u[c] * avg(cellDiv(i - 1, j, k), divc);
                }

                // --- y-momentum at v-face (i,j,k) ---
                {
                    const double vc  = avg(v[c], v[g.idx(i, j + 1, k)]);
                    const double vcm = avg(v[g.idx(i, j - 1, k)], v[c]);
                    const double uip = avg(u[g.idx(i + 1, j, k)], u[g.idx(i + 1, j - 1, k)]);
                    const double ui  = avg(u[c], u[g.idx(i, j - 1, k)]);
                    const double vip = avg(v[c], v[g.idx(i + 1, j, k)]);
                    const double vim = avg(v[g.idx(i - 1, j, k)], v[c]);
                    const double wkp = avg(w[g.idx(i, j, k + 1)], w[g.idx(i, j - 1, k + 1)]);
                    const double wk  = avg(w[c], w[g.idx(i, j - 1, k)]);
                    const double vkp = avg(v[c], v[g.idx(i, j, k + 1)]);
                    const double vkm = avg(v[g.idx(i, j, k - 1)], v[c]);

                    out.v[c] = -((uip * vip - ui * vim) / g.hx
                               + (vc * vc - vcm * vcm) / g.hy
                               + (wkp * vkp - wk * vkm) / g.hz)
                             + 0.5 * v[c] * avg(cellDiv(i, j - 1, k), divc);
                }

                // --- z-momentum at w-face (i,j,k) ---
                {
                    const double wc  = avg(w[c], w[g.idx(i, j, k + 1)]);
                    const double wcm = avg(w[g.idx(i, j, k - 1)], w[c]);
                    const double uip = avg(u[g.idx(i + 1, j, k)], u[g.idx(i + 1, j, k - 1)]);
                    const double ui  = avg(u[c], u[g.idx(i, j, k - 1)]);
                    const double wip = avg(w[c], w[g.idx(i + 1, j, k)]);
                    const double wim = avg(w[g.idx(i - 1, j, k)], w[c]);
                    const double vjp = avg(v[g.idx(i, j + 1, k)], v[g.idx(i, j + 1, k - 1)]);
                    const double vj  = avg(v[c], v[g.idx(i, j, k - 1)]);
                    const double wjp = avg(w[c], w[g.idx(i, j + 1, k)]);
                    const double wjm = avg(w[g.idx(i, j - 1, k)], w[c]);

                    out.w[c] = -((uip * wip - ui * wim) / g.hx
                               + (vjp * wjp - vj * wjm) / g.hy
                               + (wc * wc - wcm * wcm) / g.hz)
                             + 0.5 * w[c] * avg(cellDiv(i, j, k - 1), divc);
                }
            }
}

double kineticEnergy(const Grid& g, const VectorField& U)
{
    double s = 0.0;
    for (std::size_t c = 0; c < g.size(); ++c)
        s += U.u[c] * U.u[c] + U.v[c] * U.v[c] + U.w[c] * U.w[c];
    return 0.5 * s * g.cellVolume();
}

double maxCourantPerDt(const Grid& g, const VectorField& U)
{
    // Pointwise sum: SSP-RK3 + 2nd-order central is stable for
    // dt * max(|u|/hx + |v|/hy + |w|/hz) <= sqrt(3).
    double m = 0.0;
    for (std::size_t c = 0; c < g.size(); ++c)
        m = std::max(m, std::abs(U.u[c]) / g.hx
                      + std::abs(U.v[c]) / g.hy
                      + std::abs(U.w[c]) / g.hz);
    return m;
}

Vec3 meanVelocity(const Grid& g, const VectorField& U)
{
    double su = 0.0, sv = 0.0, sw = 0.0;
    for (std::size_t c = 0; c < g.size(); ++c)
    {
        su += U.u[c];
        sv += U.v[c];
        sw += U.w[c];
    }
    const double inv = 1.0 / double(g.size());
    return {su * inv, sv * inv, sw * inv};
}

void subtractMean(const Grid& g, Vec3 m, VectorField& U)
{
    for (std::size_t c = 0; c < g.size(); ++c)
    {
        U.u[c] -= m.x;
        U.v[c] -= m.y;
        U.w[c] -= m.z;
    }
}

double fluctuationEnergy(const Grid& g, const VectorField& U)
{
    const Vec3 m = meanVelocity(g, U);
    double s = 0.0;
    for (std::size_t c = 0; c < g.size(); ++c)
    {
        const double du = U.u[c] - m.x, dv = U.v[c] - m.y, dw = U.w[c] - m.z;
        s += du * du + dv * dv + dw * dw;
    }
    return 0.5 * s / double(g.size());
}

double velocityGradientSq(const Grid& g, const VectorField& U)
{
    // Nine compact 2-point differences at their natural staggered locations;
    // each squared term is volume-averaged where it lives (2nd order).
    const double ix = 1.0 / g.hx, iy = 1.0 / g.hy, iz = 1.0 / g.hz;
    double s = 0.0;
    for (int k = 0; k < g.nz; ++k)
        for (int j = 0; j < g.ny; ++j)
            for (int i = 0; i < g.nx; ++i)
            {
                const auto c = g.idx(i, j, k);
                const double dudx = (U.u[g.idx(i + 1, j, k)] - U.u[c]) * ix;
                const double dudy = (U.u[g.idx(i, j + 1, k)] - U.u[c]) * iy;
                const double dudz = (U.u[g.idx(i, j, k + 1)] - U.u[c]) * iz;
                const double dvdx = (U.v[g.idx(i + 1, j, k)] - U.v[c]) * ix;
                const double dvdy = (U.v[g.idx(i, j + 1, k)] - U.v[c]) * iy;
                const double dvdz = (U.v[g.idx(i, j, k + 1)] - U.v[c]) * iz;
                const double dwdx = (U.w[g.idx(i + 1, j, k)] - U.w[c]) * ix;
                const double dwdy = (U.w[g.idx(i, j + 1, k)] - U.w[c]) * iy;
                const double dwdz = (U.w[g.idx(i, j, k + 1)] - U.w[c]) * iz;
                s += dudx * dudx + dudy * dudy + dudz * dudz
                   + dvdx * dvdx + dvdy * dvdy + dvdz * dvdz
                   + dwdx * dwdx + dwdy * dwdy + dwdz * dwdz;
            }
    return s / double(g.size());
}

double maxAbsDivergence(const Grid& g, const VectorField& U)
{
    double m = 0.0;
    for (int k = 0; k < g.nz; ++k)
        for (int j = 0; j < g.ny; ++j)
            for (int i = 0; i < g.nx; ++i)
            {
                const double d =
                    (U.u[g.idx(i + 1, j, k)] - U.u[g.idx(i, j, k)]) / g.hx
                  + (U.v[g.idx(i, j + 1, k)] - U.v[g.idx(i, j, k)]) / g.hy
                  + (U.w[g.idx(i, j, k + 1)] - U.w[g.idx(i, j, k)]) / g.hz;
                m = std::max(m, std::abs(d));
            }
    return m;
}

} // namespace dns
