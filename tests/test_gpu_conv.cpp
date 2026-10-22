// dns conv, end to end: the buoyancy source and the two-temperature walls
// have to reproduce the ONE thing Rayleigh-Benard convection is known for --
// a layer that does nothing below the critical Rayleigh number and transports
// heat above it -- and the three independently computed Nusselt numbers have
// to agree, which is the internal heat balance closing.
//
// The runner is driven through the built binary because the case is an app
// composition (IC, output sink, runner), exactly as gpu_open_incomp does.
//
// Checks (2D, small grids; a few seconds each on a warm GPU):
//   1. Ra = 1000 < 1708: quiescent. Nu = 1 to 1%, urms/Uff < 1e-2. This is
//      also the gravity/hydrostatic-IC test -- an IC out of hydrostatic
//      balance, or a sign error in the source, cannot sit still.
//   2. Ra = 2e4 and 5e4: convecting, Nu in a generous band around the 2D
//      literature values (Nu ~ 3.0 and ~3.8 for Pr 0.72), and increasing with
//      Ra.
//   3. Wall Nusselt numbers agree top and bottom, and with the interior
//      heat-flux one: the layer is in a statistically steady heat balance.
//   4. --sides closed (a sealed cavity, no-slip adiabatic lateral walls):
//      subcritically STILL at rest -- which is what proves the lateral wall
//      fill forces nothing -- and supercritically in balance and braked, so Nu
//      falls below the periodic layer's. Plus a 3D cavity, the only thing here
//      that exercises the z wall pass and its corner ghosts.
//
// No HIP device -> skip (exit 77).

#include "dns/gpu/device_util.hpp"
#include "gpu_test_util.hpp"
#include "validation_util.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <numeric>
#include <string>
#include <vector>

#ifndef DNS_BIN
#error "DNS_BIN must be defined (path to the dns binary)"
#endif

namespace {

using dnstest::check;


struct Tail
{
    bool ok = false;
    double nuBot = 0, nuTop = 0, nuFlux = 0, urms = 0;
};

// Time-average of the last quarter of the stats rows: the run has to be past
// its transient for any of these numbers to mean anything.
Tail tailMean(const std::string& prefix)
{
    Tail r;
    const auto tbl = valtest::loadCsv(prefix + "_stats.csv");
    if (!tbl) return r;
    const auto& nb = tbl->col("nuBot");
    const auto& nt = tbl->col("nuTop");
    const auto& nf = tbl->col("nuFlux");
    const auto& ur = tbl->col("urms");
    if (nb.size() < 8) return r;
    const std::size_t i0 = nb.size() - nb.size() / 4;
    const double n = double(nb.size() - i0);
    for (std::size_t i = i0; i < nb.size(); ++i)
    {
        r.nuBot += nb[i] / n;
        r.nuTop += nt[i] / n;
        r.nuFlux += nf[i] / n;
        r.urms += ur[i] / n;
    }
    r.ok = std::isfinite(r.nuBot) && std::isfinite(r.nuTop)
        && std::isfinite(r.nuFlux) && std::isfinite(r.urms);
    return r;
}

// The free-fall velocity the case resolves to, so urms can be reported in the
// unit that makes "quiescent" mean something. Mirrors the validator:
// g = M^2 gamma / eps, dTsuper = eps - (g/cp) H, Uff = sqrt(g dTsuper).
double freeFall(double mach, double eps)
{
    const double gamma = 1.4, cp = gamma / (gamma - 1.0);
    const double g = mach * mach * gamma / eps;
    return std::sqrt(g * (eps - g / cp));
}

Tail runCase(const std::string& name, double ra, int nx, int ny, double tt,
             const std::string& extra = "", int nz = 1)
{
    const std::string prefix = "/tmp/dns_test_conv_" + name;
    const std::string cmd = "\"" DNS_BIN "\" conv --nz " + std::to_string(nz)
                          + " --n "
                          + std::to_string(nx) + " --ny " + std::to_string(ny)
                          + " --rayleigh " + std::to_string(ra) + " --tt "
                          + std::to_string(tt) + " " + extra
                          + " --emit stats --quiet --out " + prefix;
    const int rc = valtest::runBinary(cmd);
    check(rc == 0, name + ": run exits 0");
    if (rc != 0) return {};
    return tailMean(prefix);
}

} // namespace

int main()
{
    dns::test::requireDeviceOrSkip();

    const double uff = freeFall(0.05, 0.1);

    // 1. Subcritical: the conduction state is an equilibrium and must stay one.
    {
        const Tail s = runCase("sub", 1000.0, 64, 32, 40.0);
        check(s.ok, "subcritical: stats are finite");
        if (s.ok)
        {
            check(std::abs(s.nuBot - 1.0) < 0.01 && std::abs(s.nuTop - 1.0) < 0.01,
                  "subcritical: Nu = 1 to 1% (pure conduction)");
            check(s.urms / uff < 1e-2,
                  "subcritical: the layer stays at rest (urms < 1% Uff)");
        }
    }

    // 2. Supercritical: convection transports heat, and harder at higher Ra.
    const Tail a = runCase("ra2e4", 2e4, 96, 48, 25.0);
    const Tail b = runCase("ra5e4", 5e4, 96, 48, 25.0);
    check(a.ok && b.ok, "supercritical: stats are finite");
    if (a.ok && b.ok)
    {
        // 2D Rayleigh-Benard at Pr ~ 0.7 runs Nu ~ 3.0 at Ra 2e4 and ~3.8 at
        // 5e4 (the classic Nu ~ Ra^0.28). The bands are deliberately wide:
        // this is a coarse 2D grid, not a converged Nusselt measurement.
        check(a.nuBot > 2.2 && a.nuBot < 3.8, "Ra 2e4: Nu in [2.2, 3.8]");
        check(b.nuBot > 3.0 && b.nuBot < 4.8, "Ra 5e4: Nu in [3.0, 4.8]");
        check(b.nuBot > a.nuBot, "Nu increases with Ra");
        check(a.urms / uff > 0.05, "Ra 2e4: the layer is actually moving");

        // 3. Heat balance: what the floor puts in, the lid takes out, and the
        //    interior carries. Three separate estimators, one number.
        for (const auto& [t, nm] : {std::pair{a, std::string("Ra 2e4")},
                                    std::pair{b, std::string("Ra 5e4")}})
        {
            const double spread =
                std::max({std::abs(t.nuBot - t.nuTop),
                          std::abs(t.nuBot - t.nuFlux),
                          std::abs(t.nuTop - t.nuFlux)});
            check(spread / t.nuBot < 0.05,
                  nm + ": wall and interior Nusselt agree to 5%");
        }
    }

    // 4. --sides closed: the same layer in a SEALED cavity. Two statements,
    //    and the first is the one that proves the lateral wall fill injects
    //    nothing: below onset a closed box must sit just as still as a periodic
    //    one, because the conduction profile with adiabatic sides is still an
    //    exact equilibrium of the discretisation. A sign error in the momentum
    //    antimirror, or a side that conducts, shows up here as motion.
    {
        const Tail s = runCase("closedsub", 1000.0, 64, 32, 40.0,
                               "--sides closed");
        check(s.ok, "closed subcritical: stats are finite");
        if (s.ok)
        {
            check(std::abs(s.nuBot - 1.0) < 0.01 && std::abs(s.nuTop - 1.0) < 0.01,
                  "closed subcritical: Nu = 1 to 1% (pure conduction)");
            check(s.urms / uff < 1e-2,
                  "closed subcritical: the cavity stays at rest");
        }

        const Tail cl = runCase("closed5e4", 5e4, 96, 48, 25.0, "--sides closed");
        check(cl.ok, "closed supercritical: stats are finite");
        if (cl.ok && b.ok)
        {
            check(cl.urms / uff > 0.05, "closed: the cavity is actually moving");
            // The heat balance is the leak test: with INSULATING sides the only
            // way the floor's input can equal the lid's output is if nothing
            // crosses the lateral walls. 7% rather than the periodic 5% because
            // the braked corner regions make the plate fluxes genuinely less
            // symmetric, not because the balance is looser.
            const double spread = std::max({std::abs(cl.nuBot - cl.nuTop),
                                            std::abs(cl.nuBot - cl.nuFlux),
                                            std::abs(cl.nuTop - cl.nuFlux)});
            check(spread / cl.nuBot < 0.07,
                  "closed: wall and interior Nusselt agree to 7%");
            // Sidewall braking. Measured 2026-07-26 at Ra 5e4, aspect 2, 2D:
            // 3.79 periodic -> 3.40 closed. Both runs are deterministic, so
            // this is not a noise comparison -- if it ever flips, the cavity
            // has stopped braking and that is a real regression.
            std::printf("  Nu: periodic %.3f vs closed %.3f\n", b.nuBot,
                        cl.nuBot);
            check(cl.nuBot < b.nuBot,
                  "closed: Nu is BELOW the periodic layer's (no-slip sides "
                  "brake the rolls)");
        }

        // 3D closes z as well, which brings in the second lateral wall kernel
        // pass and the corner ghosts it writes over the x pass. Nothing else in
        // this tree exercises kConvWallSide on axis 2; a wrong index mapping
        // there mirrors the wrong cells, which breaks the balance rather than
        // merely perturbing it. Short and coarse on purpose -- this is a
        // wiring-and-conservation check, not a Nusselt measurement.
        const Tail c3 = runCase("closed3d", 5e4, 24, 16, 8.0, "--sides closed",
                                24);
        check(c3.ok, "3D closed sides: stats are finite");
        if (c3.ok)
        {
            const double spread = std::max({std::abs(c3.nuBot - c3.nuTop),
                                            std::abs(c3.nuBot - c3.nuFlux),
                                            std::abs(c3.nuTop - c3.nuFlux)});
            check(spread / c3.nuBot < 0.07,
                  "3D closed: wall and interior Nusselt agree to 7% (the x and "
                  "z walls leak nothing)");
            check(c3.nuBot > 1.0,
                  "3D closed: the cavity is transporting more than conduction");
        }
    }

    return dnstest::finish();
}
