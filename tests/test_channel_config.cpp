// dns channel parse + validate: wall-unit derivations, the ghost-row grid,
// the channel-only flag surface, and the rejections.
#include "dns/config.hpp"
#include "dns/output.hpp"
#include "dns/remap.hpp"
#include "check.hpp"
#include "cli_test_util.hpp"

#include <cmath>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iterator>
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
    rejected({"channel", "--re-tau", "nan"}, 1, "re-tau nan rejected by the parser");
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

    // --- RANS (docs/rans.md) ------------------------------------------------
    auto fieldIs = [](std::vector<std::string> a, const char* f) {
        auto c = runCli(std::move(a));
        return !c && c.error().code == 2 && c.error().field == f;
    };
    const std::vector<std::string> base = {"channel", "--n",   "4", "--nz",
                                           "1",       "--ny",  "64"};
    auto with = [&](std::initializer_list<const char*> extra) {
        auto v = base;
        v.insert(v.end(), extra.begin(), extra.end());
        return v;
    };
    {
        auto c = runCli(with({"--rans", "sa-neg", "--steady", "1e-3"}));
        check(c.has_value(), "RANS steady channel validates");
        if (c)
        {
            check(c->rans == dns::RansModel::SaNeg, "--rans sa-neg resolves");
            check(c->kick == 0.0, "RANS kick defaults to 0 (no trip)");
            check(c->steadyTol == 1e-3 && c->steadySteps == 1000000,
                  "--steady TOL, cap 1000000 by default");
            check(c->meta.rans == "SA-neg", "RunMeta carries the RANS label");
            check(c->warnings.empty(), "n 4 nz 1 y+ 0.876: no warnings");
            const std::string d = c->describe();
            check(d.find("  label      RANS SA-neg -- steady (local time "
                         "stepping: pseudo-time, not time-accurate)\n") !=
                      std::string::npos,
                  "describe: steady RANS label");
            check(d.find("  turbulence nu~_wall = 0 (antimirrored ghost), "
                         "mu_t = 0 on the wall face, Pr_t = 0.9, wall-resolved "
                         "(y+_1 = 0.876 est.)\n") != std::string::npos,
                  "describe: turbulence line");
            check(d.find("  run        steady TOL=0.001 cap 1000000 "
                         "iterations\n") != std::string::npos,
                  "describe: steady run line replaces the tt line");
            check(d.find("qualitative only") == std::string::npos,
                  "a RANS 1D mean carries no 2D suffix");
        }
    }
    {
        auto c = runCli(with({"--rans", "sa-neg"}));
        check(c.has_value() && c->describe().find(
                                   "  label      RANS SA-neg -- URANS "
                                   "(time-accurate)\n") != std::string::npos,
              "describe: URANS label");
    }
    {
        auto c = runCli({"channel", "--nz", "1"});
        check(c.has_value() && c->rans == dns::RansModel::None &&
                  c->meta.rans.empty() && c->kick == 0.2 &&
                  c->describe().find("turbulence") == std::string::npos &&
                  c->describe().find("  label      DNS (2D: qualitative only, "
                                     "no vortex stretching)\n") !=
                      std::string::npos,
              "RANS off: DNS label, kick 0.2, no turbulence line");
    }
    check(fieldIs(with({"--rans", "sa-neg", "--kick", "0.1"}), "--kick"),
          "--kick > 0 with --rans rejected");
    {
        auto c = runCli(with({"--rans", "sa-neg", "--kick", "0"}));
        check(c.has_value(), "--kick 0 with --rans accepted");
    }
    check(fieldIs(with({"--steady", "1e-3"}), "--steady"),
          "--steady without --rans rejected (no DNS steady state)");
    check(fieldIs(with({"--rans", "sa-neg", "--steady", "1e-3", "--wall",
                        "adiabatic"}),
                  "--steady"),
          "--steady with adiabatic walls rejected (no heat sink)");
    for (const char* f : {"--tt", "--profile-dt", "--fdt", "--pdt", "--wdt"})
        check(fieldIs(with({"--rans", "sa-neg", "--steady", "1e-3", f, "1"}), f),
              std::string("--steady rejects ") + f);
    for (const char* k : {"png", "web", "bin", "shm", "volume", "slices", "vtk"})
        check(fieldIs(with({"--rans", "sa-neg", "--steady", "1e-3", "--emit", k}),
                      "--emit"),
              std::string("--steady rejects --emit ") + k);
    check(fieldIs(with({"--rans", "sa-neg", "--steady-steps", "10"}),
                  "--steady-steps"),
          "--steady-steps without --steady rejected");
    check(fieldIs(with({"--rans", "sa-neg", "--steady", "0.5", "--steady-steps",
                        "0"}),
                  "--steady-steps"),
          "--steady-steps 0 rejected");
    check(fieldIs(with({"--rans", "sa-neg", "--steady", "-1"}), "--steady"),
          "negative --steady rejected");
    // A relative change of the integrals: [kSteadyTolMin, 1).
    for (const char* tol : {"1", "2", "5e-7"})
        check(fieldIs(with({"--rans", "sa-neg", "--steady", tol}), "--steady"),
              std::string("--steady ") + tol + " rejected (outside [1e-6, 1))");
    check(runCli(with({"--rans", "sa-neg", "--steady", "1e-6"})).has_value(),
          "--steady 1e-6 (kSteadyTolMin) accepted");
    rejected(with({"--rans", "k-epsilon"}), 1, "unknown --rans value");
    for (auto [cli, label] : {std::pair{"sst-2003m", "SST-2003m"},
                              std::pair{"sst-vm", "SST-Vm"}})
    {
        auto c = runCli(with({"--rans", cli, "--steady", "0.5"}));
        check(c.has_value() && c->meta.rans == label &&
                  c->describe().find(std::string("  label      RANS ") + label +
                                     " -- steady (local time stepping: "
                                     "pseudo-time, not time-accurate)\n") !=
                      std::string::npos &&
                  c->describe().find("omega_wall = 60 nu/(beta1 d1^2)") !=
                      std::string::npos,
              std::string("--rans ") + cli + ": label, RunMeta and wall line");
    }
    // --- --steady-scheme (docs/rans.md, "Implicit pseudo-time") -------------
    check(fieldIs(with({"--rans", "sa-neg", "--steady-scheme", "implicit"}),
                  "--steady-scheme"),
          "--steady-scheme implicit without --steady rejected");
    check(fieldIs(with({"--rans", "sa-neg", "--steady-scheme", "explicit"}),
                  "--steady-scheme"),
          "--steady-scheme explicit without --steady rejected (given = typed)");
    check(fieldIs(with({"--rans", "sa-neg", "--steady", "1e-4",
                        "--steady-scheme", "implicit", "--cfl", "0.3"}),
                  "--cfl"),
          "implicit + --cfl rejected");
    check(fieldIs(with({"--rans", "sa-neg", "--steady", "1e-4",
                        "--steady-scheme", "implicit", "--dt-recompute", "8"}),
                  "--dt-recompute"),
          "implicit + --dt-recompute rejected");
    rejected(with({"--rans", "sa-neg", "--steady", "1e-4", "--steady-scheme",
                   "newton"}),
             1, "unknown --steady-scheme value");
    {
        auto c = runCli(with({"--rans", "sa-neg", "--steady", "1e-4",
                              "--steady-scheme", "explicit", "--cfl", "0.3"}));
        check(c.has_value() && c->cfl == 0.3, "explicit + --cfl accepted");
    }
    {
        auto imp = runCli(with({"--rans", "sa-neg", "--steady", "1e-4",
                                "--steady-scheme", "implicit"}));
        auto exp = runCli(with({"--rans", "sa-neg", "--steady", "1e-4"}));
        auto expGiven = runCli(with({"--rans", "sa-neg", "--steady", "1e-4",
                                     "--steady-scheme", "explicit"}));
        check(imp.has_value() && exp.has_value() && expGiven.has_value(),
              "implicit / explicit / explicit-given validate");
        if (imp && exp && expGiven)
        {
            check(imp->steadyScheme == dns::SteadyScheme::Implicit &&
                      imp->steadySteps == 500,
                  "implicit: --steady-steps defaults to 500");
            check(exp->steadyScheme == dns::SteadyScheme::Explicit &&
                      exp->steadySteps == 1000000,
                  "explicit: --steady-steps stays 1000000");
            const std::string d = imp->describe();
            check(d.find("  label      RANS SA-neg -- steady (implicit Newton "
                         "pseudo-time: not time-accurate)\n") != std::string::npos,
                  "describe: implicit label");
            check(d.find("  run        steady TOL=0.0001 cap 500 iterations, "
                         "implicit Newton (CFL 100 x2 -> 1e8, total mass "
                         "pinned)\n") != std::string::npos,
                  "describe: implicit run line with the cap 500");
            check(exp->describe() == expGiven->describe(),
                  "explicit given or defaulted: describe byte-identical");
            check(exp->describe().find("implicit") == std::string::npos,
                  "explicit describe never mentions implicit");
            check(imp->meta.steady == "implicit" && exp->meta.steady.empty() &&
                      expGiven->meta.steady.empty(),
                  "RunMeta.steady is \"implicit\" only for implicit");
        }
    }
    for (const char* given : {"implicit", "explicit"})
    {
        auto c = runCli(with({"--rans", "sst-vm", "--steady", "1e-4",
                              "--steady-scheme", given, "--steady-steps", "77"}));
        check(c.has_value() && c->steadySteps == 77,
              std::string("--steady-steps 77 honoured under ") + given);
    }
    {
        auto c = runCli(with({"--rans", "sst-2003m", "--steady", "1e-4",
                              "--steady-scheme", "implicit"}));
        check(c.has_value() &&
                  c->describe().find("  label      RANS SST-2003m -- steady "
                                     "(implicit Newton pseudo-time: not "
                                     "time-accurate)\n") != std::string::npos,
              "describe: SST implicit label");
    }
    // The json: "steady" appended only on an implicit run.
    {
        namespace fs = std::filesystem;
        const fs::path tmp = fs::temp_directory_path() / "dns_test_channel_config";
        fs::remove_all(tmp);
        fs::create_directories(tmp);
        auto jsonOf = [&](std::initializer_list<const char*> extra) {
            auto c = runCli(with(extra));
            if (!c) return std::string();
            dns::RunOutput::Config cfg;
            cfg.prefix = (tmp / "j").string();
            cfg.kinds = {dns::Emit::Stats};
            {
                auto o = dns::RunOutput::make(cfg, c->meta, c->grid, {});
                if (!o) return std::string();
            }
            std::ifstream f(cfg.prefix + ".json");
            return std::string(std::istreambuf_iterator<char>(f), {});
        };
        const std::string ji = jsonOf({"--rans", "sa-neg", "--steady", "1e-4",
                                       "--steady-scheme", "implicit"});
        const std::string je = jsonOf({"--rans", "sa-neg", "--steady", "1e-4"});
        check(ji.find(",\"rans\":\"SA-neg\",\"steady\":\"implicit\"}") !=
                  std::string::npos,
              "json: implicit run carries \"steady\":\"implicit\"");
        check(!je.empty() && je.find("steady") == std::string::npos &&
                  je.find(",\"rans\":\"SA-neg\"}") != std::string::npos,
              "json: explicit run has no \"steady\" field");
        fs::remove_all(tmp);
    }

    {
        auto c = runCli({"channel", "--rans", "sa-neg", "--ny", "16"});
        check(c.has_value() && c->warnings.size() == 2,
              "RANS on a 3D box with y+ > 1: both warnings");
        if (c && c->warnings.size() == 2)
        {
            check(c->warnings[0] == "the RANS mean is homogeneous in x and z; "
                                    "--n 4 --nz 1 is the same answer",
                  "homogeneity warning text");
            check(c->warnings[1].starts_with("wall-resolved RANS with "
                                             "first-cell y+ "),
                  "y+ warning text");
        }
    }

    if (dnstest::gFailures) { std::printf("%d failures\n", dnstest::gFailures); return 1; }
    std::printf("all ok\n");
    return 0;
}
