#include "dns/flame1d.hpp"

#include "dns/transport.hpp"

#include <algorithm>
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
            const double tf =
                flameProfile(c.flameEig.tab, bunsenConeDistance(b, x, y));
            const double t = tf * in.jet + b.tPilot * in.pilot + b.tCo * in.co;
            const double Y = (sig - tf) / (sig - 1.0) * in.jet;
            const double rho = p0 / (Tu * t);
            const double u = m / rho;
            const std::size_t cc = g.idx(i, j, 0);
            q.rho[cc] = rho;
            q.mx[cc] = rho * u;
            q.E[cc] = p0 / (kGammaAir - 1) + 0.5 * rho * u * u;
            rhoY[cc] = rho * Y;
        }
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

void wickIC(const Case& c, CompState& q, std::vector<double> (&rhoY)[3])
{
    assert(c.sub == Subcommand::Flame && c.flameWick);
    const Grid& g = c.grid;
    const WickSpec& s = c.wick;
    const double p0 = c.flameTu, gx = c.flameGravity;
    q = CompState::zeros(g);
    rhoY[0].assign(g.size(), 0.0);
    rhoY[1].assign(g.size(), 0.0);
    if (c.wickSoot) rhoY[2].assign(g.size(), 0.0);
    else rhoY[2].clear();
    for (int j = 0; j < g.ny; ++j)
        for (int i = 0; i < g.nx; ++i) {
            const double x = (i + 0.5) * g.hx, y = (j + 0.5) * g.hy;
            const std::size_t cc = g.idx(i, j, 0);
            const double p = candleAmbientP(p0, gx, p0, x); // R T0 = p0
            const bool solid = wickSdf(s, x, y) < 0;
            double Z = 0;
            if (!solid && x > s.xTop) {
                // Distance to the wick box alone (the body's top is not a
                // fuel source).
                const double dx =
                    std::max({s.xTop - s.wWick - x, x - s.xTip(), 0.0});
                const double dy =
                    std::max(std::abs(y - s.yc) - 0.5 * s.wWick, 0.0);
                const double w =
                    0.5 * (1.0 - std::tanh((x - (s.xTip() + kWickIcAbove)) /
                                           0.5));
                Z = std::exp(-std::hypot(dx, dy) / kWickIcDecay) * w;
            }
            const SheetState st = burkeSchumann(c.diff, Z);
            const double t = solid ? 1.0 : st.t;
            const double r = p / (p0 * t);
            q.rho[cc] = r;
            q.E[cc] = p / (kGammaAir - 1);
            if (solid) continue;
            rhoY[0][cc] = r * st.YF;
            rhoY[1][cc] = r * st.YO;
        }
}

} // namespace dns
