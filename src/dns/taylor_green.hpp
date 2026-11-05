// Taylor-Green vortices: 2D analytic decaying solution and the 3D transition
// benchmark initial condition. Domain must be [0,2pi]^d.
#pragma once

#include "dns/field.hpp"
#include "dns/grid.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <vector>

namespace dns {

// 3D TGV IC (Re=1600 benchmark): u = sin x cos y cos z, v = -cos x sin y cos z, w = 0.
inline void taylorGreen3D(const Grid& g, VectorField& U)
{
    for (int k = 0; k < g.nz; ++k)
        for (int j = 0; j < g.ny; ++j)
            for (int i = 0; i < g.nx; ++i)
            {
                const auto c = g.idx(i, j, k);
                const double zc = (k + 0.5) * g.hz;
                const double xu = i * g.hx,         yu = (j + 0.5) * g.hy;
                const double xv = (i + 0.5) * g.hx, yv = j * g.hy;
                U.u[c] = std::sin(xu) * std::cos(yu) * std::cos(zc);
                U.v[c] = -std::cos(xv) * std::sin(yv) * std::cos(zc);
                U.w[c] = 0.0;
            }
}

// u =  sin(x) cos(y) e^(-2 nu t)
// v = -cos(x) sin(y) e^(-2 nu t)
inline void taylorGreen(const Grid& g, double nu, double t, VectorField& U)
{
    const double decay = std::exp(-2.0 * nu * t);
    for (int k = 0; k < g.nz; ++k)
        for (int j = 0; j < g.ny; ++j)
            for (int i = 0; i < g.nx; ++i)
            {
                const auto c = g.idx(i, j, k);
                const double xu = i * g.hx,         yu = (j + 0.5) * g.hy;
                const double xv = (i + 0.5) * g.hx, yv = j * g.hy;
                U.u[c] = std::sin(xu) * std::cos(yu) * decay;
                U.v[c] = -std::cos(xv) * std::sin(yv) * decay;
                U.w[c] = 0.0;
            }
}

// Max-norm error of U against the analytic field at time t.
inline double taylorGreenError(const Grid& g, double nu, double t, const VectorField& U)
{
    VectorField ref = VectorField::zeros(g);
    taylorGreen(g, nu, t, ref);
    double e = 0.0;
    for (std::size_t c = 0; c < g.size(); ++c)
    {
        e = std::max(e, std::abs(U.u[c] - ref.u[c]));
        e = std::max(e, std::abs(U.v[c] - ref.v[c]));
        e = std::max(e, std::abs(U.w[c] - ref.w[c]));
    }
    return e;
}

// The TGV velocity at points C (an FV mesh's centroids): u = sin x cos y cz, v = -cos x
// sin y cz, w = 0, cz = cos z (twoD: 1). Rows u, v, w of 3 n.
inline std::vector<double> taylorGreenAt(const std::array<std::vector<double>, 3>& C, bool twoD)
{
    const std::size_t nC = C[0].size();
    std::vector<double> U0(3 * nC, 0.0);
    for (std::size_t q = 0; q < nC; ++q) {
        const double x = C[0][q], y = C[1][q], z = C[2][q];
        const double cz = twoD ? 1.0 : std::cos(z);
        U0[q] = std::sin(x) * std::cos(y) * cz;
        U0[nC + q] = -std::cos(x) * std::sin(y) * cz;
    }
    return U0;
}

// The 2D TGV error of an FV velocity at time t: max |u - u_exact| over both components and
// the volume-weighted L2 norm. U rows u, v of 3 n; C, V the mesh's centroids and volumes.
struct TaylorGreenFvError
{
    double max = 0, l2 = 0;
};
inline TaylorGreenFvError taylorGreenErrorAt(const std::array<std::vector<double>, 3>& C,
                                             const std::vector<double>& V, const std::vector<double>& U, double nu,
                                             double t)
{
    const std::size_t nC = C[0].size();
    const double decay = std::exp(-2.0 * nu * t);
    double eMax = 0, e2 = 0, vol = 0;
    for (std::size_t q = 0; q < nC; ++q) {
        const double x = C[0][q], y = C[1][q];
        const double du = U[q] - std::sin(x) * std::cos(y) * decay;
        const double dv = U[nC + q] + std::cos(x) * std::sin(y) * decay;
        eMax = std::max({eMax, std::abs(du), std::abs(dv)});
        e2 += V[q] * (du * du + dv * dv);
        vol += V[q];
    }
    return {.max = eMax, .l2 = std::sqrt(e2 / vol)};
}

} // namespace dns
