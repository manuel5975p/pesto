// Taylor-Green vortices: 2D analytic decaying solution and the 3D transition
// benchmark initial condition. Domain must be [0,2pi]^d.
#pragma once

#include "dns/field.hpp"
#include "dns/grid.hpp"

#include <cmath>

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

} // namespace dns
