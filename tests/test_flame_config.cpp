// dns flame parse + validate (docs/chem.md): the resolved grid in oracle
// thermal thicknesses, the dry-run text (label, Lambda), every rejection BY
// NAME with code 2, and the CLI-level ones (unknown --mode value, the
// unregistered --lewis) with code 1.
#include "dns/config.hpp"
#include "dns/flame1d.hpp"
#include "check.hpp"
#include "cli_test_util.hpp"

#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

namespace {

using dnstest::check;
using dnstest::runCli;

// Rejected with `code`, and the error names `flag`.
void rejects(std::vector<std::string> args, int code, const std::string& flag,
             const std::string& what)
{
    auto c = runCli(std::move(args));
    const bool ok = !c && c.error().code == code &&
                    (c.error().field == flag ||
                     c.error().message.find(flag) != std::string::npos);
    if (!ok && !c)
        std::printf("    got code %d field '%s': %s\n", c.error().code,
                    c.error().field.c_str(), c.error().message.c_str());
    check(ok, what);
}

} // namespace

int main()
{
    using namespace dns;
    {
        auto c = runCli({"flame"});
        check(c.has_value(), "defaults validate");
        if (c) {
            const double dth = c->flameEig.dthOverDD;
            check(std::abs(c->flameEig.Lambda / 3656401.098 - 1) < 1e-6,
                  "Lambda = 3656401.098 (sigma 5, beta 8)");
            check(c->grid.nx == 22 * 15 + 2 * 16 && c->grid.ny == 4 &&
                      c->grid.nz == 1,
                  "grid 362 x 4 x 1 (22 delta_th at 15 cells + 2 x 16 sink)");
            check(std::abs(c->grid.hx - dth / 15) < 1e-15 &&
                      std::abs(c->grid.hy - c->grid.hx) < 1e-15,
                  "h = delta_th / 15, square cells");
            check(std::abs(c->flameXf0 - (16 * c->grid.hx + 12 * dth)) < 1e-12,
                  "x_f0 = 16 h + 12 delta_th");
            check(std::abs(c->flameTu - 1 / (1.4 * 1e-4)) < 1e-9,
                  "Tu = 1/(gamma M^2) at M 0.01");
            check(c->nscbcSink == 16 && c->nscbcEtaIn == 100,
                  "sink 16, etaIn 100 by default");
            const std::string d = c->describe();
            check(d.find(flameRunLabel(*c)) != std::string::npos &&
                      d.find("MC-limited MUSCL") != std::string::npos &&
                      d.find("(inflated; CH4/air 1.15e-3)") != std::string::npos,
                  "dry-run carries the label (model, inflated M, scheme)");
            check(d.find("3656401.098") != std::string::npos,
                  "dry-run prints Lambda");
            std::printf("%s", d.c_str());
            check(c->warnings.empty(), "no warning at the defaults");

            // The IC: the oracle flame on the grid, fresh Y = 1, burnt p.
            CompState q = CompState::zeros(c->grid);
            std::vector<double> ry;
            flameIC(*c, q, ry);
            const double T0 = c->flameTu, sig = 5;
            const std::size_t i0 = 0, iN = std::size_t(c->grid.nx - 1);
            auto Tat = [&](std::size_t i) {
                const double r = q.rho[i], u = q.mx[i] / r;
                const double p = 0.4 * (q.E[i] - 0.5 * r * u * u);
                return p / r;
            };
            check(std::abs(Tat(i0) / T0 - 1) < 1e-6 &&
                      std::abs(ry[i0] / q.rho[i0] - 1) < 1e-6,
                  "IC fresh end: T = Tu, Y = 1");
            check(std::abs(Tat(iN) / (sig * T0) - 1) < 1e-6 &&
                      std::abs(ry[iN]) < 1e-6,
                  "IC burnt end: T = sigma Tu, Y = 0");
            check(std::abs(q.mx[i0] - 1) < 1e-12 && std::abs(q.mx[iN] - 1) < 1e-12,
                  "IC mass flux = 1 (S_L) everywhere");
        }
    }
    {
        auto c = runCli({"flame", "--mach", "0.03"});
        check(c && c->warnings.size() == 1, "--mach 0.03 warns (above 0.02)");
        auto s = runCli({"flame", "--visc", "sutherland", "--tinf", "300"});
        check(s && s->sutherland && std::abs(s->flameModel.sOverTu - 0.368) < 1e-12,
              "sutherland with --tinf 300: S/Tu = 110.4/300");
        auto b = runCli({"flame", "--cells-per-delta", "30", "--zeldovich", "10"});
        check(b && b->grid.nx == 22 * 30 + 32 &&
                  std::abs(b->flameEig.Lambda / 69343676.77 - 1) < 1e-6,
              "--cells-per-delta 30 --zeldovich 10 resolve");
    }

    rejects({"flame", "--re", "100"}, 2, "--re", "--re rejected by name");
    rejects({"flame", "--ft", "3"}, 2, "--ft", "--ft rejected by name");
    rejects({"flame", "--n", "256"}, 2, "--n", "--n rejected by name");
    rejects({"flame", "--physics", "incompressible"}, 2, "--physics",
            "--physics incompressible rejected (no low-Mach branch)");
    rejects({"flame", "--nz", "4"}, 2, "--nz", "--nz 4 rejected (multi-D is P2)");
    rejects({"flame", "--ny", "3"}, 2, "--ny", "--ny 3 rejected");
    rejects({"flame", "--mach", "0.06"}, 2, "--mach", "--mach 0.06 rejected");
    rejects({"flame", "--mach", "0"}, 2, "--mach", "--mach 0 rejected");
    rejects({"flame", "--expansion", "1"}, 2, "--expansion", "--expansion 1 rejected");
    rejects({"flame", "--expansion", "11"}, 2, "--expansion", "--expansion 11 rejected");
    rejects({"flame", "--zeldovich", "1"}, 2, "--zeldovich", "--zeldovich 1 rejected");
    rejects({"flame", "--cells-per-delta", "3"}, 2, "--cells-per-delta",
            "--cells-per-delta 3 rejected");
    rejects({"flame", "--upstream", "5"}, 2, "--upstream", "--upstream 5 rejected");
    rejects({"flame", "--downstream", "3"}, 2, "--downstream",
            "--downstream 3 rejected");
    rejects({"flame", "--inflow-speed", "0"}, 2, "--inflow-speed",
            "--inflow-speed 0 rejected");
    rejects({"flame", "--visc", "sutherland"}, 2, "--tinf",
            "sutherland without --tinf rejected");
    rejects({"flame", "--tinf", "300"}, 2, "--tinf",
            "--tinf without sutherland rejected");
    rejects({"flame", "--prandtl", "0"}, 2, "--prandtl", "--prandtl 0 rejected");
    rejects({"flame", "--nscbc-sink", "-1"}, 2, "--nscbc-sink",
            "--nscbc-sink -1 rejected");
    rejects({"flame", "--nscbc-sink-amp", "3"}, 2, "--nscbc-sink-amp",
            "--nscbc-sink-amp 3 rejected");
    rejects({"flame", "--nscbc-etain", "0"}, 2, "--nscbc-etain",
            "--nscbc-etain 0 rejected");
    rejects({"flame", "--tt", "0"}, 2, "--tt", "--tt 0 rejected");
    rejects({"flame", "--mode", "diffusion"}, 1, "premixed",
            "--mode diffusion is a CLI error listing premixed");
    rejects({"flame", "--lewis", "0.8"}, 1, "--lewis",
            "--lewis is not registered (no Le != 1 oracle)");
    rejects({"flame", "--body", "cylinder"}, 1, "--body",
            "--body is not a flame flag");
    return dnstest::finish();
}
