#include "dns/flow_ic.hpp"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <numbers>

namespace dns {

using std::numbers::pi;

// Area-mean floor temperature. With a hot patch the floor is not isothermal,
// so the linear conduction profile -- which is only a starting guess once the
// patch breaks the 1D problem -- is drawn from the MEAN floor temperature
// instead of the patch's. Analytic fraction (the kernel's discrete count
// differs by at most one cell row, which an initial condition cannot care
// about): F in 2D, F^2 in 3D.
double meanFloorT(const Case& c)
{
    const double f = c.grid.nz == 1 ? c.hotPatch : c.hotPatch * c.hotPatch;
    return f * c.tBot + (1.0 - f) * c.tTop;
}

namespace {

// The lateral trip envelope for `--sides closed` (dns conv, both physics).
//
// A closed side is a wall exactly like the plates, so the stream function must
// vanish there WITH its wall-normal slope: the value is what kills the
// TANGENTIAL velocity, the slope what kills the wall-NORMAL one. q is 1 at
// mid-box and 0 at both walls, so q^2 carries both -- the same shape, and the
// same 2nd-order approach to the wall, as the y envelope the plates use.
//
// X is measured from the wall face (the inner edge of the ghost column) and L
// is the wall-to-wall extent. On an OPEN axis these are not called at all: the
// periodic factor is used verbatim, which is what keeps a run that names no
// --sides bit-for-bit the run it always was.
double sideEnv(double X, double L)
{
    const double q = X * (L - X) / (0.25 * L * L);
    return q * q;
}

double sideEnvSlope(double X, double L)
{
    const double s = 0.25 * L * L;
    return 2.0 * (X * (L - X) / s) * ((L - 2.0 * X) / s);
}

} // namespace

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

void channelRansIC(const Case& c, const std::vector<double>& wallDist,
                   double rho0, double gamma, CompState& host,
                   std::array<std::vector<double>, 2>& turb,
                   std::vector<float>& mask)
{
    const auto& g = c.grid;
    assert(c.rans != RansModel::None && wallDist.size() == g.size());
    const bool sst = ransSst(c.rans);
    for (int i = 0; i < ransScalars(c.rans); ++i) turb[i].assign(g.size(), 0.0);
    for (int k = 0; k < g.nz; ++k)
        for (int j = 0; j < g.ny; ++j)
            for (int i = 0; i < g.nx; ++i)
            {
                const std::size_t q = g.idx(i, j, k);
                const bool fluid = j >= 1 && j <= g.ny - 2;
                const double d = wallDist[q];
                const double uu = fluid ? reichardtUplus(d * c.reTau) : 0.0;
                if (!fluid) mask[q] = 0.0f;
                host.rho[q] = rho0;
                host.mx[q] = rho0 * uu;
                host.my[q] = 0.0;
                host.mz[q] = 0.0;
                host.E[q] = c.pRef / (gamma - 1.0) + 0.5 * rho0 * uu * uu;
                if (!fluid) continue;
                if (!sst) {
                    turb[0][q] = rho0 * saInitialNut(d, 1.0, 1.0);
                    continue;
                }
                const KOmega kw = sstInitialKOmega(d, c.nu / rho0, 1.0);
                turb[0][q] = rho0 * kw.k;
                turb[1][q] = rho0 * kw.omega;
            }
}

void convIC(const Case& c, double gamma, CompState& host,
            std::vector<float>& mask)
{
    const auto& g = c.grid;
    const double Rgas = 1.0, H = c.convH, Tref = 1.0;
    const double delta = 0.5 * H;                 // half-height, as the channel
    const double tFloor = meanFloorT(c);
    const double a = (c.tTop - tFloor) / H;       // conduction gradient, < 0
    const double A = c.kick * c.uFreeFall;
    const double alpha = 2 * pi / c.lxBox;
    const double bz = g.nz > 1 ? 2 * pi / c.lzBox : 0.0;
    // Closed lateral faces (--sides closed) carry the plates' ghost ring, so
    // the trip needs the plates' treatment there too: the wall face sits one
    // spacing in, and every stream-function factor is enveloped so the trip
    // vanishes at it. An open axis keeps the periodic factor verbatim.
    const bool cxS = c.closedSides, czS = c.closedSides && g.nz > 1;
    const double x0 = cxS ? g.hx : 0.0, z0 = czS ? g.hz : 0.0;
    const double Lx = c.lxBox, Lz = c.lzBox;
    // Hydrostatic pressure through a LINEAR temperature profile:
    // dp/dy = -p g / (R T(y)) integrates to p = pRef (T/Tref)^(-g/(R a)),
    // normalised to pRef at mid-layer where T = Tref. a != 0 is guaranteed:
    // validate() requires --dtemp > 0. It is a function of y alone, which is
    // exactly consistent with adiabatic side walls.
    const double pExp = -c.gravity / (Rgas * a);
    for (int k = 0; k < g.nz; ++k)
        for (int j = 0; j < g.ny; ++j)
            for (int i = 0; i < g.nx; ++i)
            {
                const std::size_t q = g.idx(i, j, k);
                // Lateral coordinates measured from the wall face when that
                // side is closed, so the envelope and the phase share an
                // origin; x0 = z0 = 0 leaves the periodic case untouched.
                const double x = (i + 0.5) * g.hx - x0;
                const double z = (k + 0.5) * g.hz - z0;
                // Physical height above the floor: the wall face sits at
                // y = hy, half a cell outside the first fluid row.
                const double y = std::clamp((j - 0.5) * g.hy, 0.0, H);
                const double T = tFloor + a * y;
                const double p = c.pRef * std::pow(T / Tref, pExp);
                const double rho = p / (Rgas * T);

                // Ghost cells belong to the wall mirror -- the two rows always,
                // and the lateral ghost columns once that side is a wall.
                const bool fluid = j >= 1 && j <= g.ny - 2
                    && (!cxS || (i >= 1 && i <= g.nx - 2))
                    && (!czS || (k >= 1 && k <= g.nz - 2));
                double uu = 0, vv = 0, ww = 0;
                if (fluid)
                {
                    const double gy = y * (2 * delta - y) / (delta * delta);
                    const double gEnv = gy * gy;
                    const double dEnv = 2.0 * gy * (2 * delta - 2 * y)
                                      / (delta * delta);
                    const double ex = cxS ? sideEnv(x, Lx) : 1.0;
                    const double ez = czS ? sideEnv(z, Lz) : 1.0;
                    const double fx = (std::sin(alpha * x + 0.1)
                                    + 0.5 * std::sin(2 * alpha * x + 1.3)) * ex;
                    // d(fx * ex)/dx: the product rule term exists only where
                    // the envelope does.
                    double dfx = (alpha * std::cos(alpha * x + 0.1)
                                + alpha * std::cos(2 * alpha * x + 1.3)) * ex;
                    if (cxS)
                        dfx += (std::sin(alpha * x + 0.1)
                             + 0.5 * std::sin(2 * alpha * x + 1.3))
                             * sideEnvSlope(x, Lx);
                    const double cz = g.nz > 1 ? std::cos(bz * z + 0.5) * ez
                                               : 1.0;
                    uu += A * delta * dEnv * fx * cz;   //  d psi_z / dy
                    vv -= A * delta * gEnv * dfx * cz;  // -d psi_z / dx
                    if (g.nz > 1)
                    {   // psi_x = A d gEnv cos(a x + 0.9) sin(b z + 0.2)
                        const double cx = std::cos(alpha * x + 0.9) * ex;
                        vv += A * delta * gEnv * cx * bz
                            * (std::cos(bz * z + 0.2) * ez);  //  d psi_x / dz
                        if (czS)
                            vv += A * delta * gEnv * cx
                                * std::sin(bz * z + 0.2)
                                * sideEnvSlope(z, Lz);
                        ww -= A * delta * dEnv * cx
                            * (std::sin(bz * z + 0.2) * ez); // -d psi_x / dy
                    }
                }
                else mask[q] = 0.0f;
                host.rho[q] = rho;
                host.mx[q] = rho * uu;
                host.my[q] = rho * vv;
                host.mz[q] = rho * ww;
                host.E[q] = p / (gamma - 1.0)
                          + 0.5 * rho * (uu * uu + vv * vv + ww * ww);
            }
}


void rbIC(const Case& c, VectorField& U, Scalar& T, std::vector<float>& mask)
{
    const auto& g = c.grid;
    const double H = c.convH, delta = 0.5 * H;
    const double tFloor = meanFloorT(c);
    const double a = (c.tTop - tFloor) / H;   // conduction gradient, < 0
    const double A = c.kick * c.uFreeFall;
    const double alpha = 2 * pi / c.lxBox;
    const double bz = g.nz > 1 ? 2 * pi / c.lzBox : 0.0;
    // The bottom wall face is the v-face j = 1, so a cell centre sits at
    // (j - 0.5) hy above it and a y-face at (j - 1) hy.
    auto yc = [&](int j) { return (j - 0.5) * g.hy; };
    auto yf = [&](int j) { return (j - 1.0) * g.hy; };
    // Envelope with gEnv = gEnv' = 0 at BOTH plates, so the trip respects
    // no-slip exactly; its curl is the velocity below.
    //
    // The cos(pi y/H) factor is NOT decoration. With plates at +/- dT/2 about
    // Tref = 0 the Boussinesq equations are exactly equivariant under the
    // up-down flip (y -> H-y, T -> -T, v -> -v), so a trip that respects that
    // symmetry leaves the solution locked in a symmetric state for the whole
    // run -- measured: the two wall Nusselt numbers then agree to SIX digits
    // at every instant, by construction rather than because the heat balance
    // closed, and the flow is not generic turbulence. cos(pi y/H) is
    // antisymmetric about mid-height and breaks it. (The compressible branch
    // needs no such term: its hydrostatic stratification breaks the symmetry
    // physically, which is why ITS nuBot - nuTop genuinely fluctuates.)
    auto asym = [&](double y) { return 1.0 + 0.5 * std::cos(pi * y / H); };
    auto dAsym = [&](double y) {
        return -0.5 * (pi / H) * std::sin(pi * y / H);
    };
    auto gEnv = [&](double y) {
        const double q = y * (2 * delta - y) / (delta * delta);
        return q * q * asym(y);
    };
    auto dEnv = [&](double y) {
        const double q = y * (2 * delta - y) / (delta * delta);
        const double dq = 2.0 * q * (2 * delta - 2 * y) / (delta * delta);
        return dq * asym(y) + q * q * dAsym(y);
    };
    // Closed lateral faces: the same story as the plates, one axis over. The
    // wall face is the u-face i = 1 (and the w-face k = 1), so the envelope
    // evaluated at a face coordinate makes the trip's wall-normal component
    // EXACTLY zero there, and its slope makes the tangential ones vanish too.
    // Open axis: the periodic factor verbatim, hence bit-for-bit unchanged.
    const bool cxS = c.closedSides, czS = c.closedSides && g.nz > 1;
    const double x0 = cxS ? g.hx : 0.0, z0 = czS ? g.hz : 0.0;
    const double Lx = c.lxBox, Lz = c.lzBox;
    auto ex = [&](double x) { return cxS ? sideEnv(x, Lx) : 1.0; };
    auto ez = [&](double z) { return czS ? sideEnv(z, Lz) : 1.0; };
    auto fx = [&](double x) {
        return (std::sin(alpha * x + 0.1) + 0.5 * std::sin(2 * alpha * x + 1.3))
             * ex(x);
    };
    auto dfx = [&](double x) {
        double d = (alpha * std::cos(alpha * x + 0.1)
                 + alpha * std::cos(2 * alpha * x + 1.3)) * ex(x);
        if (cxS)
            d += (std::sin(alpha * x + 0.1)
               + 0.5 * std::sin(2 * alpha * x + 1.3)) * sideEnvSlope(x, Lx);
        return d;
    };

    for (int k = 0; k < g.nz; ++k)
        for (int j = 0; j < g.ny; ++j)
            for (int i = 0; i < g.nx; ++i)
            {
                const std::size_t q = g.idx(i, j, k);
                const double z = (k + 0.5) * g.hz - z0;
                T[q] = tFloor + a * std::clamp(yc(j), 0.0, H);
                // Ghost cells belong to the wall mirror -- the two rows always,
                // and the lateral ghost columns once that side is a wall. The
                // same rule, and the same mask contract, as convIC.
                const bool fluid = j >= 1 && j <= g.ny - 2
                    && (!cxS || (i >= 1 && i <= g.nx - 2))
                    && (!czS || (k >= 1 && k <= g.nz - 2));
                if (!fluid)
                {
                    mask[q] = 0.0f;
                    continue;
                }

                const double xu = i * g.hx - x0, xv = (i + 0.5) * g.hx - x0;
                const double czU = g.nz > 1 ? std::cos(bz * z + 0.5) * ez(z)
                                            : 1.0;
                U.u[q] = A * delta * dEnv(yc(j)) * fx(xu) * czU;   //  d psi_z / dy
                U.v[q] = -A * delta * gEnv(yf(j)) * dfx(xv) * czU; // -d psi_z / dx
                if (g.nz > 1)
                {   // psi_x = A d gEnv cos(a x + 0.9) sin(b z + 0.2)
                    const double zf = k * g.hz - z0;
                    U.v[q] += A * delta * gEnv(yf(j))
                            * (std::cos(alpha * xv + 0.9) * ex(xv)) * bz
                            * (std::cos(bz * z + 0.2) * ez(z)); //  d psi_x / dz
                    if (czS)
                        U.v[q] += A * delta * gEnv(yf(j))
                                * (std::cos(alpha * xv + 0.9) * ex(xv))
                                * std::sin(bz * z + 0.2)
                                * sideEnvSlope(z, Lz);
                    U.w[q] = -A * delta * dEnv(yc(j))
                           * (std::cos(alpha * xv + 0.9) * ex(xv))
                           * (std::sin(bz * zf + 0.2) * ez(zf)); // -d psi_x / dy
                }
            }
}

} // namespace dns
