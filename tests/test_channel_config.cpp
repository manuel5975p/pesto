// dns channel parse + validate: wall-unit derivations, the ghost-row grid,
// the channel-only flag surface, and the rejections.
#include "dns/config.hpp"
#include "dns/remap.hpp"
#include "check.hpp"
#include "cli_test_util.hpp"

#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

namespace {

using dnstest::check;



using dnstest::runCli;
using dnstest::rejected;


} // namespace

int main()
{
    // Defaults resolve to the KMM box with two ghost rows.
    {
        auto c = runCli({"channel"});
        check(c.has_value(), "defaults validate");
        if (c)
        {
            check(c->grid.nx == 192 && c->grid.ny == 130 && c->grid.nz == 160,
                  "grid is 192 x (128+2) x 160");
            check(std::abs(c->lxBox - 4 * 3.14159265358979) < 1e-9,
                  "box length 4 pi");
            check(std::abs(c->reTau - 180.0) < 1e-12
                      && std::abs(c->nu - 1.0 / 180.0) < 1e-15,
                  "nu = 1/reTau");
            check(c->physics == dns::Physics::Compressible
                      && c->far == dns::FarField::Periodic,
                  "compressible periodic");
            check(c->wallIso, "isothermal walls by default");
            check(std::abs(c->tEnd - 30.0) < 1e-12, "tEnd = tt default 30");
            check(c->dtRecompute == 16, "dt-recompute defaults to 16");
            check(c->emit.contains(dns::Emit::Stats), "stats emitted");
            // The grid period must be exactly 2 delta + 2 h1 -- the ghost
            // reflection is only consistent with that box.
            const double h1 = dns::ChannelStretchY::firstCell(130, 2.0, 1.0);
            check(std::abs(c->grid.ny * c->grid.hy - (2 + 2 * h1)) < 1e-12,
                  "y period = 2 delta + 2 h1");
            check(std::abs(c->chanH1 - h1) < 1e-15, "resolved first cell");
            check(c->chanH1 * c->reTau < 1.0,
                  "default first-cell y+ under 1 (wall-resolved)");
            check(c->stretchGrowth > 1.0 && c->stretchGrowth < 1.2,
                  "bounded growth ratio");
            check(c->pRef > 0 && c->ubEst > 10 && c->ubEst < 25,
                  "bulk estimate and pressure sane");
            check(!c->describe().empty(), "describe renders");
        }
    }

    // 2D channel is allowed (native 2D restriction).
    {
        auto c = runCli({"channel", "--nz", "1"});
        check(c.has_value() && c->grid.nz == 1, "nz 1 accepted");
    }
    // Uniform mesh (beta 0) still builds a consistent box.
    {
        auto c = runCli({"channel", "--stretch", "0", "--ny", "16"});
        check(c.has_value(), "beta 0 validates");
        if (c)
            check(std::abs(c->chanH1 - 2.0 / 16.0) < 1e-12,
                  "beta 0 first cell = 2 delta / ny");
    }
    // Adiabatic wall selection.
    {
        auto c = runCli({"channel", "--wall", "adiabatic"});
        check(c.has_value() && !c->wallIso, "--wall adiabatic");
    }

    // Rejections: channel-foreign flags and bad values.
    rejected({"channel", "--re", "100"}, 2, "--re rejected (use --re-tau)");
    rejected({"channel", "--ft", "5"}, 2, "--ft rejected (use --tt)");
    rejected({"channel", "--body", "cylinder"}, 1, "--body is not a channel flag");
    rejected({"channel", "--far", "nscbc"}, 1, "--far is not a channel flag");
    rejected({"channel", "--les", "0.1"}, 1, "--les is not a channel flag");
    rejected({"channel", "--re-tau", "0"}, 2, "re-tau must be positive");
    rejected({"channel", "--re-tau", "nan"}, 2, "re-tau nan caught by validate");
    rejected({"channel", "--mach", "1.5"}, 2, "mach must be under 1");
    rejected({"channel", "--ny", "4"}, 2, "ny under 8 rejected");
    rejected({"channel", "--nz", "2"}, 2, "nz 2 rejected (1 or >= 4)");
    rejected({"channel", "--stretch", "9"}, 2, "stretch above 4 rejected");
    rejected({"channel", "--tt", "0"}, 2, "tt must be positive");
    rejected({"channel", "--profile-dt", "-1"}, 2, "negative profile-dt");
    {
        auto c = runCli({"channel", "--emit", "png"});
        check(c.has_value(), "png allowed in 3D (centre z-plane)");
    }
    rejected({"channel", "--wall", "slippery"}, 1, "unknown wall kind");

    if (dnstest::gFailures) { std::printf("%d failures\n", dnstest::gFailures); return 1; }
    std::printf("all ok\n");
    return 0;
}
