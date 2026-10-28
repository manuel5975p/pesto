#include "dns/flame1d.hpp"

#include "dns/transport.hpp"

#include <cassert>

namespace dns {

void premixedState(const Grid& g, double gamma, double Tu, double sigma,
                   double m, const std::function<double(double)>& t,
                   CompState& q, std::vector<double>& rhoY)
{
    assert(Tu > 0 && sigma > 1 && gamma > 1);
    q = CompState::zeros(g);
    rhoY.assign(g.size(), 0.0);
    const double p0 = Tu; // rho_u = R = 1
    for (int i = 0; i < g.nx; ++i) {
        const double tt = t((i + 0.5) * g.hx);
        const double T = Tu * tt;
        const double rho0 = p0 / T;
        const double p = p0 - m * m * (1 / rho0 - 1);
        const double rho = p / T;
        const double u = m / rho;
        const double Y = (sigma - tt) / (sigma - 1);
        for (int k = 0; k < g.nz; ++k)
            for (int j = 0; j < g.ny; ++j) {
                const std::size_t c = g.idx(i, j, k);
                q.rho[c] = rho;
                q.mx[c] = rho * u;
                q.E[c] = p / (gamma - 1) + 0.5 * rho * u * u;
                rhoY[c] = rho * Y;
            }
    }
}

void flameIC(const Case& c, CompState& q, std::vector<double>& rhoY)
{
    assert(c.sub == Subcommand::Flame && !c.flameEig.tab.T.empty());
    const FlameEigen& tab = c.flameEig.tab;
    const double xf = c.flameXf0;
    premixedState(c.grid, kGammaAir, c.flameTu, c.flameModel.sigma, c.flameUIn,
                  [&](double x) { return flameProfile(tab, x - xf); }, q, rhoY);
}

} // namespace dns
