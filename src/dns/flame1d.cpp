#include "dns/flame1d.hpp"

#include "dns/transport.hpp"

#include <cassert>
#include <cmath>
#include <numbers>

namespace dns {

void premixedState(const Grid& g, double gamma, double Tu, double sigma,
                   double m, const std::function<double(double, double)>& t,
                   CompState& q, std::vector<double>& rhoY)
{
    assert(Tu > 0 && sigma > 1 && gamma > 1);
    q = CompState::zeros(g);
    rhoY.assign(g.size(), 0.0);
    const double p0 = Tu; // rho_u = R = 1
    for (int j = 0; j < g.ny; ++j)
        for (int i = 0; i < g.nx; ++i) {
            const double tt = t((i + 0.5) * g.hx, (j + 0.5) * g.hy);
            const double T = Tu * tt;
            const double rho0 = p0 / T;
            const double p = p0 - m * m * (1 / rho0 - 1);
            const double rho = p / T;
            const double u = m / rho;
            const double Y = (sigma - tt) / (sigma - 1);
            for (int k = 0; k < g.nz; ++k) {
                const std::size_t c = g.idx(i, j, k);
                q.rho[c] = rho;
                q.mx[c] = rho * u;
                q.E[c] = p / (gamma - 1) + 0.5 * rho * u * u;
                rhoY[c] = rho * Y;
            }
        }
}

double flameWrinkle(double A, int N, double y, double Ly)
{
    assert(Ly > 0 && N >= 1);
    if (A == 0) return 0.0;
    constexpr double tau = 2 * std::numbers::pi;
    constexpr double phiGold = 0.6180339887498949;
    const double a = A / std::sqrt(double(N));
    double s = 0;
    for (int k = 1; k <= N; ++k) {
        // Quadratic in k: a linear phase k*phiGold is a pure y shift, which
        // sums the modes into one localized bump.
        const double phi = tau * std::fmod(double(k) * k * phiGold, 1.0);
        s += a * std::cos(tau * k * y / Ly + phi);
    }
    return s;
}

void flameIC(const Case& c, CompState& q, std::vector<double>& rhoY)
{
    assert(c.sub == Subcommand::Flame && !c.flameEig.tab.T.empty());
    const FlameEigen& tab = c.flameEig.tab;
    const double xf = c.flameXf0;
    const double A = c.flameWrinkle * c.flameEig.dthOverDD;
    const double Ly = c.grid.ny * c.grid.hy;
    // --tfles F: the oracle thickened by F (the TFLES flame's own profile).
    if (c.flameTfles > 1) {
        const double F = c.flameTfles;
        premixedState(c.grid, kGammaAir, c.flameTu, c.flameModel.sigma,
                      c.flameUIn,
                      [&](double x, double y) {
                          return flameProfile(
                              tab, (x - xf - flameWrinkle(A, c.flameWrinkleModes,
                                                          y, Ly)) / F);
                      },
                      q, rhoY);
        return;
    }
    premixedState(c.grid, kGammaAir, c.flameTu, c.flameModel.sigma, c.flameUIn,
                  [&](double x, double y) {
                      return flameProfile(
                          tab, x - xf - flameWrinkle(A, c.flameWrinkleModes, y, Ly));
                  },
                  q, rhoY);
}

void bunsenIC(const Case& c, CompState& q, std::vector<double>& rhoY)
{
    assert(c.sub == Subcommand::Flame && c.flameBunsen &&
           !c.flameEig.tab.T.empty());
    const Grid& g = c.grid;
    const BunsenGeom& b = c.bunsen;
    const double sig = c.flameModel.sigma, Tu = c.flameTu, p0 = Tu;
    q = CompState::zeros(g);
    rhoY.assign(g.size(), 0.0);
    for (int j = 0; j < g.ny; ++j) {
        const double y = (j + 0.5) * g.hy;
        const BunsenInflowRow in = bunsenInflow(b, y);
        const double m = p0 / (Tu * in.t) * in.u; // the row's inflow rho u
        for (int i = 0; i < g.nx; ++i) {
            const double x = (i + 0.5) * g.hx;
            // --tfles F: the oracle thickened by F across the cone.
            const double dc = c.flameTfles > 1
                                  ? bunsenConeDistance(b, x, y) / c.flameTfles
                                  : bunsenConeDistance(b, x, y);
            const double tf = flameProfile(c.flameEig.tab, dc);
            const double t = tf * in.jet + b.tPilot * in.pilot + b.tCo * in.co;
            const double Y = (sig - tf) / (sig - 1.0) * in.jet;
            const double rho = p0 / (Tu * t);
            const double u = m / rho;
            // 3D: the slot is z-invariant, the IC extruded over the span.
            for (int k = 0; k < g.nz; ++k) {
                const std::size_t cc = g.idx(i, j, k);
                q.rho[cc] = rho;
                q.mx[cc] = rho * u;
                q.E[cc] = p0 / (kGammaAir - 1) + 0.5 * rho * u * u;
                rhoY[cc] = rho * Y;
            }
        }
    }
}

void addBunsenTurbulence(const Case& c,
                         const std::array<std::vector<double>, 3>& box,
                         const std::vector<double>& rhoY, CompState& q)
{
    assert(c.flameTurb > 0 && c.flameBunsen);
    const Grid& g = c.grid;
    const int nbx = c.flameTurbNbx;
    assert(box[0].size() == std::size_t(nbx) * g.ny * g.nz &&
           rhoY.size() == g.size());
    for (int k = 0; k < g.nz; ++k)
        for (int j = 0; j < g.ny; ++j)
            for (int i = 0; i < g.nx; ++i) {
                const std::size_t b =
                    std::size_t(i % nbx) +
                    std::size_t(nbx) * (j + std::size_t(g.ny) * k);
                const std::size_t cc = g.idx(i, j, k);
                const double r = q.rho[cc];
                const double w = rhoY[cc] / r; // the fresh fraction Y
                const double u0 = q.mx[cc] / r, v0 = q.my[cc] / r,
                             w0 = q.mz[cc] / r;
                const double u = u0 + w * box[0][b], v = v0 + w * box[1][b],
                             ww = w0 + w * box[2][b];
                q.E[cc] += 0.5 * r * ((u * u + v * v + ww * ww) -
                                      (u0 * u0 + v0 * v0 + w0 * w0));
                q.mx[cc] = r * u;
                q.my[cc] = r * v;
                q.mz[cc] = r * ww;
            }
}

void diffusionIC(const Case& c, CompState& q, std::vector<double> (&rhoY)[2])
{
    assert(c.sub == Subcommand::Flame && c.flameDiffusion);
    const Grid& g = c.grid;
    const double p0 = c.flameTu;
    q = CompState::zeros(g);
    for (auto& v : rhoY) v.assign(g.size(), 0.0);
    // Candle: the ambient column's pressure on every row (horizontally
    // uniform p, so only buoyancy starts the hot columns); g == 0 takes the
    // diffusion arithmetic verbatim.
    const double gx = c.flameCandle ? c.flameGravity : 0.0;
    const double Lign = c.flameIgnition;
    for (int j = 0; j < g.ny; ++j) {
        const DiffusionInflowRow row =
            c.flameQuiescent ? quiescentInflow(c.diff)
                             : diffusionInflow(c.diff, c.bunsen, (j + 0.5) * g.hy);
        for (int i = 0; i < g.nx; ++i) {
            const double x = (i + 0.5) * g.hx;
            DiffusionInflowRow in = row;
            if (Lign > 0) {
                // The short burning column: Z(y) w(x), its Burke-Schumann
                // state, u from the row's to the coflow's.
                const double w =
                    0.5 * (1.0 - std::tanh((x - Lign) / c.bunsen.edge));
                const SheetState st = burkeSchumann(c.diff, row.Z * w);
                in = {c.bunsen.uCo + (row.u - c.bunsen.uCo) * w, row.Z * w,
                      st.YF, st.YO, st.t, row.jet * w};
            }
            const double rho = 1.0 / in.t; // p0 / (R T0 t), R = 1, p0 = T0
            const std::size_t cc = g.idx(i, j, 0);
            double r = rho, p = p0;
            if (gx > 0) {
                p = candleAmbientP(p0, gx, p0, x); // R T0 = p0
                r = p / (p0 * in.t);
            }
            q.rho[cc] = r;
            q.mx[cc] = r * in.u;
            q.E[cc] = p / (kGammaAir - 1) + 0.5 * r * in.u * in.u;
            rhoY[0][cc] = r * in.YF;
            rhoY[1][cc] = r * in.YO;
        }
    }
}

TurbSpec flameTurbSpec(const Case& c)
{
    assert(c.sub == Subcommand::Flame && c.flameTurb > 0);
    const Grid& g = c.grid;
    TurbSpec s;
    s.urms = c.flameTurb;
    s.lt = c.flameTurbL;
    s.nx = c.flameTurbNbx;
    s.ny = g.ny;
    s.nz = g.nz;
    s.Lx = s.nx * g.hx;
    s.Ly = g.ny * g.hy;
    s.Lz = g.nz * g.hz;
    s.seed = c.seed;
    return s;
}

void addTurbulence(const Case& c, const std::array<std::vector<double>, 3>& box,
                   CompState& q)
{
    assert(c.flameTurb > 0);
    const Grid& g = c.grid;
    const int nbx = c.flameTurbNbx;
    assert(box[0].size() == std::size_t(nbx) * g.ny * g.nz);
    const double xw = c.flameXf0 - 1.5 * c.flameTurbL;
    const double ew = 0.25 * c.flameTurbL;
    for (int k = 0; k < g.nz; ++k)
        for (int j = 0; j < g.ny; ++j)
            for (int i = 0; i < g.nx; ++i) {
                const double x = (i + 0.5) * g.hx;
                const double w = 0.5 * (1.0 - std::tanh((x - xw) / ew));
                const std::size_t b =
                    std::size_t(i % nbx) + std::size_t(nbx) * (j + std::size_t(g.ny) * k);
                const std::size_t cc = g.idx(i, j, k);
                const double r = q.rho[cc];
                const double u0 = q.mx[cc] / r, v0 = q.my[cc] / r,
                             w0 = q.mz[cc] / r;
                const double u = u0 + w * box[0][b], v = v0 + w * box[1][b],
                             ww = w0 + w * box[2][b];
                q.E[cc] += 0.5 * r * ((u * u + v * v + ww * ww) -
                                      (u0 * u0 + v0 * v0 + w0 * w0));
                q.mx[cc] = r * u;
                q.my[cc] = r * v;
                q.mz[cc] = r * ww;
            }
}

} // namespace dns
