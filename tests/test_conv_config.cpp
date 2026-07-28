// dns conv parse + validate: the Rayleigh-Benard derivation chain (gravity
// from the Mach number, the superadiabatic drop, nu from Ra), the ghost-row
// grid, the hydrostatic initial condition, and the rejections -- above all the
// subadiabatic layer, which is a configuration error and not a result.
#include "dns/cli.hpp"
#include "dns/comp.hpp"
#include "dns/config.hpp"
#include "dns/flow_ic.hpp"
#include "check.hpp"

#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

namespace {

using dnstest::check;

std::expected<dns::Case, dns::ConfigError> run(std::vector<std::string> args)
{
    std::vector<char*> argv;
    argv.push_back(const_cast<char*>("dns"));
    for (auto& a : args) argv.push_back(a.data());
    auto raw = dns::cli::parse(int(argv.size()), argv.data());
    if (!raw)
        return std::unexpected(dns::ConfigError{1, "cli", raw.error().message});
    return dns::validate(*raw);
}

bool rejected(std::vector<std::string> args, int code, const std::string& what)
{
    auto c = run(std::move(args));
    const bool ok = !c && c.error().code == code;
    check(ok, what);
    return ok;
}

bool near(double a, double b, double tol) { return std::abs(a - b) <= tol; }

} // namespace

int main()
{
    const double gamma = 1.4, cp = gamma / (gamma - 1.0);

    // The derivation chain, checked term by term against its own definitions.
    {
        auto c = run({"conv"});
        check(c.has_value(), "defaults validate");
        if (c)
        {
            check(c->grid.nx == 128 && c->grid.ny == 66 && c->grid.nz == 128,
                  "grid is 128 x (64+2) x 128");
            check(near(c->lxBox, 2.0, 1e-12) && near(c->lzBox, 2.0, 1e-12),
                  "aspect 2 box, 2 x 1 x 2");
            // The two ghost rows sit half a cell outside the wall faces, so
            // the mapped y extent is H + 2 hy and the layer is exactly H.
            check(near(double(c->grid.ny) * c->grid.hy, 1.0 + 2.0 / 64, 1e-12),
                  "y extent is H + two ghost rows");
            check(near(c->grid.hy, 1.0 / 64, 1e-15), "hy = H / nyFluid");

            const double eps = 0.1, M = 0.05;
            const double g = M * M * gamma / eps;
            check(near(c->gravity, g, 1e-15), "g = M^2 gamma / dtemp");
            check(near(c->tBot, 1 + eps / 2, 1e-15)
                      && near(c->tTop, 1 - eps / 2, 1e-15),
                  "walls at 1 +/- dtemp/2");
            const double dTs = eps - g / cp;
            check(near(c->dtempSuper, dTs, 1e-15),
                  "superadiabatic drop = dtemp - (g/cp) H");
            check(near(c->uFreeFall, std::sqrt(g * dTs), 1e-15),
                  "free-fall velocity on the superadiabatic drop");
            check(near(c->nu, std::sqrt(g * dTs * 0.72 / 1e6), 1e-18),
                  "nu from Ra on the superadiabatic drop");
            check(near(c->tEnd, 30.0 / c->uFreeFall, 1e-9),
                  "tt counts free-fall times");
            check(c->physics == dns::Physics::Compressible
                      && c->far == dns::FarField::Periodic
                      && c->bodyKind == dns::BodyKind::None,
                  "compressible periodic, no body");
            check(c->dtRecompute == 16, "dt-recompute defaults to 16");
            check(c->warnings.empty(), "the default case raises no advisory");
        }
    }

    // Ra enters only through nu, and as an inverse square root.
    {
        auto a = run({"conv", "--rayleigh", "1e6"});
        auto b = run({"conv", "--rayleigh", "4e6"});
        check(a && b && near(a->nu / b->nu, 2.0, 1e-9),
              "nu scales as 1/sqrt(Ra)");
        check(a && b && near(a->gravity, b->gravity, 1e-15),
              "Ra does not move gravity");
    }

    // The adiabatic fraction is M^2 (gamma-1) / dtemp^2, so it vanishes
    // quadratically with the Mach number.
    {
        auto lo = run({"conv", "--mach", "0.02"});
        auto hi = run({"conv", "--mach", "0.08"});
        check(lo && hi, "both Mach numbers validate");
        if (lo && hi)
        {
            auto frac = [&](const dns::Case& c) {
                return (c.dtemp - c.dtempSuper) / c.dtemp;
            };
            check(near(frac(*hi) / frac(*lo), 16.0, 1e-9),
                  "adiabatic fraction scales as M^2");
            check(frac(*lo) < 0.02, "the Boussinesq limit is nearly adiabatic-free");
        }
    }

    // A subadiabatic layer is stably stratified and cannot convect: rejected
    // by name, not run to a null result.
    rejected({"conv", "--mach", "0.6", "--dtemp", "0.1"}, 2,
             "subadiabatic layer rejected");
    {
        auto c = run({"conv", "--mach", "0.6", "--dtemp", "0.1"});
        check(!c && c.error().field == "--mach"
                  && c.error().message.find("subadiabatic") != std::string::npos,
              "the rejection names --mach and says subadiabatic");
    }

    // Onset and 2D are advisories, not errors: both are legitimate runs.
    {
        auto c = run({"conv", "--rayleigh", "1000"});
        check(c.has_value(), "subcritical Ra validates");
        check(c && c->warnings.size() == 1
                  && c->warnings[0].find("1708") != std::string::npos,
              "subcritical Ra warns about onset");
        auto d = run({"conv", "--nz", "1"});
        check(d && d->warnings.size() == 1
                  && d->warnings[0].find("vortex stretching") != std::string::npos,
              "nz == 1 warns that 2D Nu is qualitative");
    }

    // --sides closed turns the lateral faces into walls, and it does so with
    // the PLATES' idiom: a ghost column per walled axis, so the GRID grows and
    // the fluid extent does not move.
    {
        auto p = run({"conv", "--n", "32", "--ny", "16", "--nz", "8"});
        auto c = run({"conv", "--n", "32", "--ny", "16", "--nz", "8", "--sides",
                      "closed"});
        check(p.has_value() && c.has_value(), "both --sides values validate");
        if (p && c)
        {
            check(!p->closedSides && c->closedSides,
                  "the flag reaches the case");
            check(c->grid.nx == 34 && c->grid.nz == 10,
                  "each closed axis gains two ghost cells");
            check(c->grid.ny == p->grid.ny,
                  "y is untouched: the plates always had theirs");
            check(near(c->grid.hx, p->grid.hx, 1e-15)
                      && near(c->grid.hz, p->grid.hz, 1e-15),
                  "the spacing is unchanged: the ghosts extend the box rather "
                  "than compress the fluid into it");
            check(near(double(c->grid.nx) * c->grid.hx,
                       2.0 + 2.0 * p->grid.hx, 1e-12),
                  "x extent is the fluid width + two ghost columns");
            check(near(c->lxBox, p->lxBox, 1e-15) && c->nxInterior == 32
                      && c->nzInterior == 8,
                  "the fluid extent and counts are what --aspect and --n say");
            check(c->warnings.size() == 1
                      && c->warnings[0].find("cavity") != std::string::npos,
                  "closed sides warn that Nu is a cavity number");
        }
        // 2D closes x only: z at nz == 1 is exact periodic wrap, not a
        // degenerate wall, and a z ghost there would invent a third dimension.
        auto d = run({"conv", "--n", "32", "--ny", "16", "--nz", "1", "--sides",
                      "closed"});
        check(d && d->grid.nx == 34 && d->grid.nz == 1,
              "at nz == 1 only x closes");
        auto e = run({"conv", "--sides", "periodic"});
        auto f = run({"conv"});
        check(e && f && !e->closedSides && e->grid.nx == f->grid.nx,
              "--sides periodic is the default, spelled out");
    }
    rejected({"conv", "--sides", "open"}, 1, "an unknown --sides value rejected");
    rejected({"flow", "--body", "cylinder", "--sides", "closed"}, 1,
             "--sides is not a flow flag (the far field is --far)");

    // The flag surface: conv is configured by Ra, not by a body Reynolds
    // number, and its run length is in free-fall times.
    rejected({"conv", "--re", "100"}, 2, "--re rejected (no body, no stream)");
    rejected({"conv", "--ft", "10"}, 2, "--ft rejected (run length is --tt)");
    rejected({"conv", "--dtemp", "0"}, 2, "--dtemp 0 rejected");
    rejected({"conv", "--dtemp", "2.5"}, 2, "--dtemp >= 2 rejected (lid <= 0 K)");
    rejected({"conv", "--rayleigh", "-1"}, 2, "negative Ra rejected");
    rejected({"conv", "--mach", "1.5"}, 2, "supersonic Mach rejected");
    rejected({"conv", "--ny", "4"}, 2, "too few wall-normal cells rejected");
    rejected({"conv", "--aspect", "0"}, 2, "zero aspect rejected");
    rejected({"conv", "--body", "cylinder"}, 1, "--body is not a conv flag");
    rejected({"conv", "--far", "nscbc"}, 1, "--far is not a conv flag");

    // --temp is a compressible-only frame channel.
    rejected({"flow", "--body", "cylinder", "--temp"}, 2,
             "--temp rejected on the incompressible solver");
    {
        auto c = run({"flow", "--body", "cylinder", "--physics", "compressible",
                      "--temp"});
        check(c && c->tempFrames, "--temp accepted on the compressible solver");
        auto d = run({"flow", "--body", "cylinder", "--physics", "compressible"});
        check(d && !d->tempFrames, "temp frames are off by default");
    }

    // The initial condition is a genuine hydrostatic equilibrium: dp/dy must
    // balance -rho g to the discretisation, not merely be "close to uniform".
    {
        auto c = run({"conv", "--n", "8", "--ny", "32", "--nz", "1",
                      "--kick", "0"});
        check(c.has_value(), "IC case validates");
        if (c)
        {
            const auto& g = c->grid;
            auto st = dns::CompState::zeros(g);
            std::vector<float> mask(g.size(), 1.0f);
            dns::convIC(*c, gamma, st, mask);

            double worst = 0, momentum = 0;
            for (int j = 1; j <= g.ny - 3; ++j)
            {
                const std::size_t a = g.idx(0, j, 0), b = g.idx(0, j + 1, 0);
                auto pressure = [&](std::size_t q) {
                    return (gamma - 1.0) * st.E[q];   // the IC is at rest
                };
                const double dpdy = (pressure(b) - pressure(a)) / g.hy;
                const double rhoFace = 0.5 * (st.rho[a] + st.rho[b]);
                const double resid = dpdy + rhoFace * c->gravity;
                worst = std::max(worst, std::abs(resid));
                momentum = std::max(momentum, std::abs(rhoFace * c->gravity));
            }
            check(worst / momentum < 1e-4,
                  "IC is hydrostatic: dp/dy + rho g = 0 to 1e-4 relative");

            // Ghost rows are the wall mirror's, and the trip respects no-slip.
            check(mask[g.idx(0, 0, 0)] == 0.0f
                      && mask[g.idx(0, g.ny - 1, 0)] == 0.0f,
                  "ghost rows are masked out");
            double maxM = 0;
            for (std::size_t q = 0; q < g.size(); ++q)
                maxM = std::max(maxM, std::abs(double(st.mx[q]))
                                          + std::abs(double(st.my[q])));
            check(maxM == 0.0, "--kick 0 starts exactly at rest");
        }
    }

    // The trip vanishes at both walls. The wall face lies half a cell outside
    // the first fluid row, so the statement that can be checked on the grid is
    // the ORDER at which it vanishes: the envelope is O(y^2) with a zero
    // derivative, so the wall-normal trip velocity in the first row must fall
    // like h^2 under refinement -- a merely small value at one resolution
    // would not distinguish an envelope that vanishes from one that is just
    // weak.
    {
        auto firstRowFraction = [&](int nyFluid) {
            auto c = run({"conv", "--n", "16", "--ny", std::to_string(nyFluid),
                          "--nz", "1", "--kick", "0.05"});
            check(c.has_value(), "kicked case validates (ny "
                                     + std::to_string(nyFluid) + ")");
            if (!c) return -1.0;
            const auto& g = c->grid;
            auto st = dns::CompState::zeros(g);
            std::vector<float> mask(g.size(), 1.0f);
            dns::convIC(*c, gamma, st, mask);
            double wallV = 0, anyV = 0;
            for (int i = 0; i < g.nx; ++i)
            {
                wallV = std::max(wallV, std::abs(double(st.my[g.idx(i, 1, 0)])));
                for (int j = 1; j <= g.ny - 2; ++j)
                    anyV = std::max(anyV,
                                    std::abs(double(st.my[g.idx(i, j, 0)])));
            }
            check(anyV > 0, "the trip is present (ny "
                                + std::to_string(nyFluid) + ")");
            return anyV > 0 ? wallV / anyV : -1.0;
        };
        const double r16 = firstRowFraction(16);
        const double r32 = firstRowFraction(32);
        check(r16 > 0 && r32 > 0 && r32 < 0.005,
              "the trip is negligible in the first fluid row");
        check(r16 > 0 && r32 > 0 && r16 / r32 > 3.0,
              "it vanishes at the wall at 2nd order (h^2 under refinement)");
    }

    // A CLOSED side is a wall, so the same two statements have to hold there:
    // the ghost columns belong to the wall fill, and the trip vanishes at the
    // wall face -- at 2nd order in the LATERAL spacing this time, since the
    // face sits half a cell outside the first fluid column.
    {
        auto firstColFraction = [&](int nxFluid) {
            auto c = run({"conv", "--n", std::to_string(nxFluid), "--ny", "16",
                          "--nz", "1", "--kick", "0.05", "--sides", "closed"});
            check(c.has_value(), "closed kicked case validates (n "
                                     + std::to_string(nxFluid) + ")");
            if (!c) return -1.0;
            const auto& g = c->grid;
            auto st = dns::CompState::zeros(g);
            std::vector<float> mask(g.size(), 1.0f);
            dns::convIC(*c, gamma, st, mask);
            check(mask[g.idx(0, 1, 0)] == 0.0f
                      && mask[g.idx(g.nx - 1, 1, 0)] == 0.0f,
                  "closed: the lateral ghost columns are masked out");
            double wallU = 0, anyU = 0;
            for (int j = 1; j <= g.ny - 2; ++j)
            {
                wallU = std::max(wallU, std::abs(double(st.mx[g.idx(1, j, 0)])));
                for (int i = 1; i <= g.nx - 2; ++i)
                    anyU = std::max(anyU,
                                    std::abs(double(st.mx[g.idx(i, j, 0)])));
            }
            check(anyU > 0, "closed: the trip is present (n "
                                + std::to_string(nxFluid) + ")");
            return anyU > 0 ? wallU / anyU : -1.0;
        };
        const double c16 = firstColFraction(16);
        const double c32 = firstColFraction(32);
        check(c16 > 0 && c32 > 0 && c32 < 0.02,
              "closed: the trip is negligible in the first fluid column");
        check(c16 > 0 && c32 > 0 && c16 / c32 > 3.0,
              "closed: it vanishes at the side wall at 2nd order");
    }

    // The Boussinesq twin, where the statement is exact rather than asymptotic:
    // u lives ON the lateral wall faces (i = 1 and i = nx-1), so the enveloped
    // trip must put a hard zero there, and the conduction profile must not
    // depend on x -- an adiabatic side wall is only consistent with a
    // temperature field that has no lateral gradient to conduct.
    {
        auto c = run({"conv", "--physics", "incompressible", "--n", "32",
                      "--ny", "16", "--nz", "1", "--kick", "0.05", "--sides",
                      "closed"});
        check(c.has_value(), "closed Boussinesq case validates");
        if (c)
        {
            const auto& g = c->grid;
            auto U = dns::VectorField::zeros(g);
            auto T = dns::zeros(g);
            dns::rbIC(*c, U, T);
            double wallLo = 0, wallHi = 0, any = 0, dTdx = 0;
            for (int j = 1; j <= g.ny - 2; ++j)
            {
                wallLo = std::max(wallLo, std::abs(U.u[g.idx(1, j, 0)]));
                wallHi = std::max(wallHi, std::abs(U.u[g.idx(g.nx - 1, j, 0)]));
                for (int i = 1; i <= g.nx - 2; ++i)
                {
                    any = std::max(any, std::abs(U.u[g.idx(i, j, 0)]));
                    dTdx = std::max(dTdx, std::abs(T[g.idx(i, j, 0)]
                                                   - T[g.idx(1, j, 0)]));
                }
            }
            check(any > 0, "closed Boussinesq: the trip is present");
            check(wallLo == 0.0 && wallHi < 1e-12 * std::max(any, 1e-30),
                  "closed Boussinesq: u is zero ON both lateral wall faces");
            check(dTdx == 0.0,
                  "closed Boussinesq: the conduction profile has no lateral "
                  "gradient (consistent with adiabatic sides)");
        }
    }

    return dnstest::finish();
}
