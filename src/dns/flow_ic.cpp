#include "dns/flow_ic.hpp"

#include <algorithm>
#include <cmath>
#include <numbers>

namespace dns {

using std::numbers::pi;

void incompressibleBodyIC(const Case& c, const CaseGeometry& geo,
                          VectorField& U)
{
    const auto& g = c.grid;
    const double h = c.h;

    // Quiet start (--ramp): rest, exactly consistent with the no-slip body --
    // the impulsive uniform stream is what rings up the trapped stagnation-
    // region waves. The trip stays either way (it seeds the wake asymmetry
    // once flow arrives).
    for (auto& x : U.u) x = c.ramp > 0 ? 0.0 : c.uInf;
    for (int k = 0; k < g.nz; ++k)
        for (int j = 0; j < g.ny; ++j)
            for (int i = 0; i < g.nx; ++i)
            {
                const double x = i * h, y = (j + 0.5) * h;
                if (x > c.placement.cx + geo.D / 2 && x < c.placement.cx + 1.5 * geo.D
                    && std::abs(y - c.placement.cy) < 0.5 * geo.D)
                    U.v[g.idx(i, j, k)] = 0.05 * c.uInf;
            }
}

void compressibleBodyIC(const Case& c, const CaseGeometry& geo,
                        const AxisRemap3& fRemap, const TanhStretchY& yRemap,
                        double rhoInf, double pInf, double gamma,
                        CompState& host, std::vector<float>& mask)
{
    const auto& g = c.grid;
    const double h = c.h;
    const double u0 = c.ramp > 0 ? 0.0 : c.uInf;
    // Deterministic solenoidal trip, quiet start only: a weak Gaussian vortex
    // one D behind the body seeds the wake instability. Divergence-free, so
    // acoustically silent; the quiet start alone is too symmetric to shed.
    const double kA = c.ramp > 0 ? c.kick * c.uInf : 0.0, kw = 0.25 * geo.D;
    const double kx = c.placement.cx + geo.D, ky = c.placement.cy + 0.25 * geo.D;

    // Physical cell-centre positions come from the PointRemap of whichever
    // curvilinear mesh is active (both are the exact identity when off), so
    // the taper and the trip sit where the cells actually are.
    for (int k = 0; k < g.nz; ++k)
        for (int j = 0; j < g.ny; ++j)
            for (int i = 0; i < g.nx; ++i)
            {
                const std::size_t q = g.idx(i, j, k);
                const double x = fRemap.fx.map((i + 0.5) * h),
                             y = fRemap.fy.map(yRemap.mapY((j + 0.5) * g.hy)),
                             z = g.nz > 1 ? fRemap.fz.map((k + 0.5) * g.hz)
                                          : c.placement.cz;
                const double d = geo.sdf(x, y, z);
                const double f = std::clamp(d / (3.0 * h), 0.0, 1.0);
                double uu = u0 * f * f * (3 - 2 * f), vv = 0.0;
                if (kA != 0.0 && d > 0.0)
                {
                    const double dx = x - kx, dy = y - ky;
                    const double e = kA * std::exp(-(dx * dx + dy * dy) / (kw * kw));
                    uu += 2 * dy / kw * e;
                    vv -= 2 * dx / kw * e;
                }
                host.rho[q] = rhoInf;
                host.mx[q] = rhoInf * uu;
                host.my[q] = rhoInf * vv;
                host.E[q] = pInf / (gamma - 1.0)
                          + 0.5 * rhoInf * (uu * uu + vv * vv);
                if (d < 0.0) mask[q] = 0.0f;
            }
}

void channelIC(const Case& c, const ChannelStretchY& remap, double delta,
               double rho0, double gamma, CompState& host,
               std::vector<float>& mask)
{
    const auto& g = c.grid;
    // IC: laminar Poiseuille scaled to the log-law bulk target (so the flux
    // does not have to be shed during transition) plus a deterministic
    // divergence-free trip u' = curl(psi). The envelope
    // gEnv = (y(2d-y)/d^2)^2 has gEnv = gEnv' = 0 at both walls, so the trip
    // respects no-slip; solenoidal, hence acoustically silent. No RNG: fixed
    // modes, fixed phases. nz == 1 keeps psi_x = 0 so w stays exactly 0.
    const double ub = c.ubEst, A = c.kick * ub;
    const double alpha = 2 * pi / c.lxBox;
    const double bz = g.nz > 1 ? 2 * pi / c.lzBox : 0.0;
    for (int k = 0; k < g.nz; ++k)
        for (int j = 0; j < g.ny; ++j)
            for (int i = 0; i < g.nx; ++i)
            {
                const std::size_t q = g.idx(i, j, k);
                const double x = (i + 0.5) * g.hx;
                const double y = remap.mapY((j + 0.5) * g.hy);
                const double z = (k + 0.5) * g.hz;
                double uu = 0, vv = 0, ww = 0;
                if (j >= 1 && j <= g.ny - 2)
                {
                    const double eta = (y - delta) / delta;
                    uu = 1.5 * ub * (1.0 - eta * eta);
                    const double gy = y * (2 * delta - y) / (delta * delta);
                    const double gEnv = gy * gy;
                    const double dEnv = 2.0 * gy * (2 * delta - 2 * y)
                                      / (delta * delta);
                    const double fx = std::sin(alpha * x + 0.1)
                                    + 0.5 * std::sin(2 * alpha * x + 1.3);
                    const double dfx = alpha * std::cos(alpha * x + 0.1)
                                     + alpha * std::cos(2 * alpha * x + 1.3);
                    const double cz = g.nz > 1 ? std::cos(bz * z + 0.5) : 1.0;
                    uu += A * delta * dEnv * fx * cz;   //  d psi_z / dy
                    vv -= A * delta * gEnv * dfx * cz;  // -d psi_z / dx
                    if (g.nz > 1)
                    {   // psi_x = A d gEnv cos(a x + 0.9) sin(b z + 0.2)
                        const double cx = std::cos(alpha * x + 0.9);
                        vv += A * delta * gEnv * cx * bz
                            * std::cos(bz * z + 0.2);   //  d psi_x / dz
                        ww -= A * delta * dEnv * cx
                            * std::sin(bz * z + 0.2);   // -d psi_x / dy
                    }
                }
                else mask[q] = 0.0f; // ghost rows: the wall mirror owns them
                host.rho[q] = rho0;
                host.mx[q] = rho0 * uu;
                host.my[q] = rho0 * vv;
                host.mz[q] = rho0 * ww;
                host.E[q] = c.pRef / (gamma - 1.0)
                          + 0.5 * rho0 * (uu * uu + vv * vv + ww * ww);
            }
}

} // namespace dns
