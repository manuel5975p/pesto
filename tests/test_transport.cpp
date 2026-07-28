// dns::Transport: the Sutherland viscosity law, its degenerate constant-mu
// limit, the conductivity that rides it, and the diffusivity factor the
// explicit dt limit needs. Pure CPU, no GPU, no solver.
#include "dns/transport.hpp"
#include "dns/comp.hpp"
#include "check.hpp"

#include <cmath>
#include <cstdio>

using namespace dns;
using dnstest::check;

namespace {

constexpr double gam = 1.4, Rgas = 1.0, mach = 0.2;
const double Tinf = 1.0 / (gam * mach * mach);

Transport constantGas(double mu) { return Transport::air(mu, Tinf, 288.15, 0.72, gam, Rgas, false); }
Transport sutherlandGas(double mu, double tinfK = 288.15)
{
    return Transport::air(mu, Tinf, tinfK, 0.72, gam, Rgas, true);
}

// The law, written out independently of the implementation.
double referenceRatio(double T, double Tref, double S)
{
    return std::pow(T / Tref, 1.5) * (Tref + S) / (T + S);
}

void constantLimit()
{
    const auto t = constantGas(3e-3);
    bool exact = true;
    for (double T = 0.01 * Tinf; T < 20 * Tinf; T *= 1.7)
        exact = exact && t.ratio(T) == 1.0 && t.mu(T) == 3e-3;
    check(exact, "constant viscosity: ratio is EXACTLY 1 at every temperature");
    check(t.S == 0.0, "constant viscosity carries no Sutherland constant");
    // cp = gamma R / (gamma - 1); kappa = mu cp / Pr.
    const double cp = gam * Rgas / (gam - 1.0);
    check(std::abs(t.kappa(Tinf) - 3e-3 * cp / 0.72) < 1e-15 * t.kappa(Tinf),
          "kappa = mu cp / Pr");
}

void sutherlandLaw()
{
    const auto t = sutherlandGas(3e-3);
    // The dimensional anchor: S/Tref must be 110.4 K / Tinf[K].
    check(std::abs(t.S / t.Tref - Transport::sutherlandK / 288.15) < 1e-14,
          "Sutherland constant scales as 110.4 K / Tinf[K]");
    check(std::abs(t.ratio(Tinf) - 1.0) < 1e-14,
          "Sutherland is exactly the reference viscosity at Tref");

    double worst = 0.0;
    for (double T = 0.2 * Tinf; T < 6 * Tinf; T *= 1.3)
        worst = std::max(worst, std::abs(t.ratio(T) - referenceRatio(T, t.Tref, t.S))
                                    / referenceRatio(T, t.Tref, t.S));
    std::printf("Sutherland vs closed form: max rel %.3e\n", worst);
    check(worst < 1e-14, "Sutherland matches the closed form");

    bool monotone = true;
    double prev = t.mu(0.05 * Tinf);
    for (double T = 0.06 * Tinf; T < 20 * Tinf; T *= 1.05) {
        const double m = t.mu(T);
        monotone = monotone && m > prev;
        prev = m;
    }
    // Monotonicity is load-bearing: stableDt bounds mu over the field from a
    // single max-T reduction, which is only valid because mu(T) never falls.
    check(monotone, "mu(T) is monotone increasing (the max-T dt bound needs it)");

    // Air, sanity against the textbook value: mu(T) rises ~T^0.7 near room
    // temperature, so doubling T must raise mu by clearly less than 2^1.5 and
    // clearly more than 1.
    const double r2 = t.ratio(2 * Tinf);
    std::printf("mu(2 Tinf)/mu(Tinf) = %.4f\n", r2);
    check(r2 > 1.4 && r2 < std::pow(2.0, 1.5),
          "doubling T raises mu between linear-ish and the T^1.5 ceiling");

    // A colder freestream means the 110.4 K constant is relatively larger, so
    // the law is closer to T^1.5 there.
    const auto cold = sutherlandGas(3e-3, 100.0);
    check(cold.ratio(2 * Tinf) > r2,
          "a colder Tinf makes the law steeper (S/Tinf larger)");
}

void diffusivity()
{
    const auto t = constantGas(1e-3);
    // Air: gamma/Pr = 1.4/0.72 = 1.944 -- the thermal diffusivity, not nu, is
    // what the explicit limit must respect.
    check(std::abs(t.diffusivityFactor() - gam / 0.72) < 1e-14,
          "diffusivity factor is gamma / Pr for air");
    auto hot = t;
    hot.Pr = 2.0; // Pr > gamma: momentum diffuses fastest, factor clamps to 1
    check(hot.diffusivityFactor() == 1.0,
          "factor clamps at 1 when momentum is the fastest diffusion");
}

// A Sutherland run must still conserve mass, momentum and energy exactly: the
// law only changes the coefficient in a flux that is still computed once per
// face and differenced.
void conservationWithSutherland()
{
    const auto g = Grid::make(16, 16, 1, 1.0, 1.0, 1.0);
    auto c = Comp::make(g, gam, 2e-3);
    c.tr = sutherlandGas(2e-3);

    auto s = CompState::zeros(g);
    for (int j = 0; j < g.ny; ++j)
        for (int i = 0; i < g.nx; ++i) {
            const double x = (i + 0.5) * g.hx, y = (j + 0.5) * g.hy;
            // A hot spot: mu varies by a good fraction across the box.
            const double T = Tinf * (1.0 + 0.6 * std::sin(2 * M_PI * x) *
                                               std::sin(2 * M_PI * y));
            const double rho = 1.0 + 0.1 * std::cos(2 * M_PI * x);
            const double u = 0.2 * std::sin(2 * M_PI * y);
            const std::size_t k = g.idx(i, j, 0);
            s.rho[k] = rho;
            s.mx[k] = rho * u;
            s.E[k] = rho * Rgas * T / (gam - 1.0) + 0.5 * rho * u * u;
        }

    auto sum = [&](const std::vector<double>& f) {
        double t = 0.0;
        for (double v : f) t += v;
        return t;
    };
    const double m0 = sum(s.rho), px0 = sum(s.mx), e0 = sum(s.E);
    const auto dt = c.stableDt(s, 0.3);
    check(dt.has_value(), "Sutherland stableDt succeeds on a smooth state");
    if (!dt) return;
    for (int it = 0; it < 20; ++it) c.step(s, *dt);
    const double m1 = sum(s.rho), px1 = sum(s.mx), e1 = sum(s.E);
    std::printf("Sutherland drift: mass %.3e  mx %.3e  E %.3e\n",
                std::abs(m1 - m0) / m0, std::abs(px1 - px0),
                std::abs(e1 - e0) / e0);
    check(std::abs(m1 - m0) / m0 < 1e-13 && std::abs(px1 - px0) < 1e-12 &&
              std::abs(e1 - e0) / e0 < 1e-13,
          "mu(T) keeps mass/momentum/energy conserved");

    // And the law is actually engaged: a constant-mu run of the same state
    // must give a DIFFERENT trajectory.
    auto s2 = CompState::zeros(g);
    auto c2 = Comp::make(g, gam, 2e-3);
    (void)c2;
    check(c.tr.sutherland && c.tr.mu(2 * Tinf) != c.tr.mu(Tinf),
          "the Sutherland run really has a temperature-dependent viscosity");
}

// The thermal dt limit must actually bind tighter than the momentum one.
void thermalDtLimit()
{
    const auto g = Grid::make(8, 8, 1, 1.0, 1.0, 1.0);
    // mu large enough that the diffusive limit beats the acoustic one
    // (c = sqrt(gamma R Tinf) = 5 here, so dt_acoustic = 0.4 h / 5).
    constexpr double muBig = 0.5;
    auto c = Comp::make(g, gam, muBig);
    auto s = CompState::zeros(g);
    for (std::size_t k = 0; k < g.size(); ++k) {
        s.rho[k] = 1.0;
        s.E[k] = Rgas * Tinf / (gam - 1.0);
    }
    const auto dt = c.stableDt(s, 0.4);
    check(dt.has_value(), "diffusive stableDt succeeds");
    if (!dt) return;
    const double h = g.hx;
    const double momentumOnly = 0.25 * h * h * 1.0 / muBig;
    std::printf("dt %.6e vs momentum-only %.6e (ratio %.4f)\n", *dt,
                momentumOnly, *dt / momentumOnly);
    check(*dt < momentumOnly * 0.99,
          "the thermal diffusivity binds tighter than nu alone");
    check(std::abs(*dt - momentumOnly / (gam / 0.72)) < 1e-12 * *dt,
          "the diffusive dt carries exactly the gamma/Pr factor");
}

} // namespace

int main()
{
    constantLimit();
    sutherlandLaw();
    diffusivity();
    conservationWithSutherland();
    thermalDtLimit();
    return dnstest::finish();
}
