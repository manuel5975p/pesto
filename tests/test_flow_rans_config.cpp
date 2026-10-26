// dns flow --rans / compressible --steady (masterplan_flowrans section 3.6):
// every accept, reject and warn by flag name and exit 2, the describe lines,
// RANS-off describes byte-identical to the strings HEAD (2eb06ba) printed,
// explicit-given == defaulted, and the RunMeta keys only on flow RANS/steady.
// Host-only, milliseconds.
#include "dns/config.hpp"
#include "dns/ogrid.hpp"
#include "dns/pseudo_time.hpp"
#include "dns/rans.hpp"
#include "check.hpp"
#include "cli_test_util.hpp"

#include <cmath>
#include <cstdio>
#include <numbers>
#include <string>
#include <vector>

namespace {

using dnstest::check;
using dnstest::rejected;
using dnstest::runCli;
using Args = std::vector<std::string>;

// The O-grid cylinder every case starts from (validates at HEAD).
Args og(Args extra = {})
{
    Args a = {"flow",    "--physics", "compressible", "--body", "cylinder",
              "--ogrid", "--n",       "256",          "--ny",   "96",
              "--re",    "40",        "--out",        "o"};
    a.insert(a.end(), extra.begin(), extra.end());
    return a;
}

// Rejected with exit 2 AND naming `flag` as the field.
void rejectedAt(Args a, const std::string& flag, const std::string& what)
{
    auto c = runCli(a);
    const bool ok = !c && c.error().code == 2 && c.error().field == flag;
    if (!c && !ok)
        std::printf("    got field '%s': %s\n", c.error().field.c_str(),
                    c.error().message.c_str());
    check(ok, what);
}

bool hasWarning(const dns::Case& c, const std::string& part)
{
    for (const auto& w : c.warnings)
        if (w.find(part) != std::string::npos) return true;
    return false;
}

// RANS off: the describe HEAD printed, byte for byte.
void headDescribes()
{
    const char* ogHead =
        "case:\n"
        "  physics    compressible\n"
        "  body       cylinder\n"
        "  far field  o-grid outer row (characteristic LODI, gpu/ogrid_bc)\n"
        "  grid       256 x 96 x 1  h=0.162928  L=31.4159  [native 2D]\n"
        "  placement  cx=0 cy=0 cz=0 D=0.785398 (4.8 cells)\n"
        "  Re=40  nu=0.019635  Mach=0.2  cfl=0.4  Cs=0 (DNS)\n"
        "  ft=20  tEnd=628.319  fdt=0.15\n"
        "  mode       time-march to tEnd\n"
        "  out        o\n"
        "  o-grid     BODY-FITTED: 256 angular x 96 radial (2 ghost rows) x 1 "
        "spanwise\n"
        "             rFar=15.708 (20 D)  beta=3  span=1  h1\xe2\x89\x88"
        "5.004e-03 (nominal circle, beta=3)\n"
        "             the wall is a MESH FACE -- no cut cells, no immersed "
        "body,\n"
        "             forces are an exact surface integral\n";
    auto a = runCli(og());
    check(a && a->describe() == ogHead, "RANS off: O-grid describe == HEAD");
    if (a && a->describe() != ogHead) std::printf("%s", a->describe().c_str());

    const char* incHead =
        "case:\n"
        "  physics    incompressible\n"
        "  body       cylinder\n"
        "  far field  tunnel\n"
        "  grid       130 x 130 x 1  h=0.0490874  L=6.28319  [native 2D]\n"
        "  placement  cx=2.1587 cy=3.19902 cz=0.0245437 D=0.785398 (16.0 "
        "cells)\n"
        "  Re=40  nu=0.019635  Mach=0.2  cfl=0.8  Cs=0 (DNS)\n"
        "  ft=20  tEnd=125.664  fdt=0.15\n"
        "  mode       stationary (stop at steady residual, tEnd = give-up cap)\n"
        "  out        i\n";
    auto b = runCli({"flow", "--body", "cylinder", "--n", "128", "--re", "40",
                     "--steady", "1e-3", "--out", "i"});
    check(b && b->describe() == incHead,
          "incompressible --steady describe == HEAD (its meaning unchanged)");

    const char* sphHead =
        "case:\n"
        "  physics    compressible\n"
        "  body       sphere\n"
        "  far field  nscbc\n"
        "  grid       96 x 96 x 96  h=0.0654498  L=6.28319\n"
        "  placement  cx=2.1107 cy=3.15619 cz=3.14159 D=0.785398 (12.0 "
        "cells)\n"
        "  Re=300  nu=0.00261799  Mach=0.2  cfl=0.4  Cs=0 (DNS)\n"
        "  ft=20  tEnd=125.664  fdt=0.15\n"
        "  mode       time-march to tEnd\n"
        "  out        s\n"
        "  nscbc      scale-selective sink: 16-cell band, sin^4 cutoff (~86% "
        "at 4dx, ~1% at 16dx), sigma_max = 1 c/h\n";
    auto s = runCli({"flow", "--physics", "compressible", "--body", "sphere",
                     "--n", "96", "--nz", "96", "--re", "300", "--out", "s"});
    check(s && s->describe() == sphHead,
          "RANS off: immersed compressible describe == HEAD");
    for (auto* c : {&a, &b, &s})
        check(*c && (*c)->meta.ransMode.empty() &&
                  (*c)->meta.ransInflow.empty() && (*c)->meta.rans.empty(),
              "RANS off: no flow-RANS RunMeta keys");
    auto ch = runCli({"channel", "--rans", "sa-neg", "--steady", "1e-3"});
    check(ch && ch->meta.ransMode.empty() && ch->meta.ransInflow.empty() &&
              ch->meta.rans == "SA-neg",
          "channel RunMeta unchanged (no flow-RANS keys)");
}

void accepts()
{
    for (const char* m : {"sa-neg", "sst-2003m", "sst-vm"}) {
        auto c = runCli(og({"--rans", m}));
        check(c.has_value(), std::string("accepted: --ogrid --rans ") + m);
        if (!c) continue;
        check(c->meta.ransMode == "urans" && !c->meta.ransInflow.empty(),
              std::string(m) + ": RunMeta ransMode urans + ransInflow");
        const std::string d = c->describe();
        check(d.find("  label      RANS " + std::string(c->meta.rans) +
                     " -- URANS (time-accurate)\n") != std::string::npos,
              std::string(m) + ": describe label line");
        check(d.find("  turbulence ") != std::string::npos &&
                  d.find("  wall       ") != std::string::npos &&
                  d.find("  y+         first cell ~") != std::string::npos &&
                  d.find("  run        URANS") != std::string::npos,
              std::string(m) + ": turbulence, wall, y+ and run lines");
        check(d.find("Cs=0 (RANS)") != std::string::npos &&
                  d.find("qualitative") == std::string::npos,
              std::string(m) + ": labelled RANS, no 2D qualitative suffix");
    }
    // Steady: the explicit LTS march, defaults resolved.
    {
        auto c = runCli(og({"--rans", "sa-neg", "--steady", "1e-4"}));
        check(c.has_value(), "accepted: --rans sa-neg --steady 1e-4");
        if (c) {
            check(c->compSteady && c->steadySteps == 1000000 &&
                      c->dtRecompute == 16 && c->ramp == 0 && c->kick == 0 &&
                      c->emit.size() == 1 && c->emit.contains(dns::Emit::Stats),
                  "steady: cap 1e6, localDt every 16, no ramp/kick, stats only");
            check(c->meta.ransMode == "steady-explicit" && c->meta.steady.empty(),
                  "steady: RunMeta ransMode steady-explicit");
            const std::string d = c->describe();
            check(d.find("  label      RANS SA-neg -- steady (local time "
                         "stepping: pseudo-time, not time-accurate)\n") !=
                      std::string::npos,
                  "steady: label line");
            check(d.find("  run        steady TOL=0.0001 cap 1000000 "
                         "iterations") != std::string::npos,
                  "steady: run line");
            check(d.find("  turbulence nu~inf/nu = 3 (TMR), Pr_t = 0.9\n") !=
                      std::string::npos,
                  "SA turbulence line");
            check(std::abs(c->ransPhiInf[0] - 3.0 * c->nu) < 1e-15,
                  "SA nu~inf = 3 nu");
            // explicit given == defaulted.
            auto g = runCli(og({"--rans", "sa-neg", "--steady", "1e-4",
                                "--steady-scheme", "explicit",
                                "--rans-nutinf", "3", "--steady-steps",
                                "1000000"}));
            check(g && g->describe() == d,
                  "explicit-given == defaulted: describe byte-identical");
        }
    }
    // Laminar compressible steady on the O-grid.
    {
        auto c = runCli(og({"--steady", "1e-5"}));
        check(c && c->compSteady && c->rans == dns::RansModel::None &&
                  c->meta.ransMode == "steady-explicit" && c->meta.rans.empty(),
              "accepted: laminar --ogrid --steady (RunMeta steady-explicit)");
        check(c && c->describe().find("  label      DNS -- steady (local time "
                                      "stepping: pseudo-time, not "
                                      "time-accurate)\n") != std::string::npos,
              "laminar steady label");
    }
    // SST freestream: omega = k/(R nu); the TMR M 0.2 value.
    {
        auto c = runCli(og({"--rans", "sst-2003m", "--re", "10000"}));
        check(c.has_value(), "accepted: SST Re 1e4");
        if (c) {
            const double k = 2.25e-7, w = k / (0.009 * c->nu);
            check(std::abs(c->ransPhiInf[0] - k) < 1e-20 &&
                      std::abs(c->ransPhiInf[1] - w) < 1e-12 * w,
                  "SST k_inf = K U^2, omega_inf = k/(R nu)");
            check(std::abs(w * c->placement.D - 0.25) < 1e-12,
                  "omega_inf = 0.25 U/D at Re 1e4 (decision 4)");
            check(c->describe().find("  turbulence k = 2.25e-07 U^2 (Tu "
                                     "0.0387%), mu_t/mu = 0.009 -> omega = "
                                     "0.25 U/D, Pr_t = 0.9\n") !=
                      std::string::npos,
                  "SST turbulence line");
            check(c->meta.ransInflow == "k = 2.25e-07 U^2 (Tu 0.0387%), "
                                        "mu_t/mu = 0.009 -> omega = 0.25 U/D",
                  "RunMeta ransInflow is the describe string");
        }
    }
    {
        auto c = runCli(og({"--rans", "sa-neg", "--visc", "sutherland",
                            "--wall", "isothermal", "--twall", "1.2",
                            "--temp", "--emit", "surface", "--emit", "stats"}));
        check(c.has_value(), "accepted: RANS with the thermal knobs, --temp, "
                             "--emit surface");
    }
    {
        auto c = runCli(og({"--rans", "sa-neg", "--ramp", "1", "--kick",
                            "0.1"}));
        check(c.has_value(), "accepted: --ramp/--kick with URANS");
    }
}

void rejects()
{
    rejectedAt({"flow", "--body", "cylinder", "--rans", "sa-neg", "--out", "x"},
               "--rans", "incompressible + --rans");
    rejectedAt(og({"--rans", "sa-neg", "--les", "0.1"}), "--rans",
               "--rans + --les");
    rejectedAt(og({"--rans", "sa-neg", "--sgs", "sigma"}), "--sgs",
               "--rans + --sgs alone (no --les): the --sgs rule first");
    rejectedAt(og({"--rans", "sa-neg", "--rotor-mach", "0.2", "--rotor-frame",
                   "mrf", "--rotor-axis", "z"}),
               "--rans", "--rans + rotor (MRF)");
    rejectedAt({"flow", "--physics", "compressible", "--body", "cylinder",
                "--n", "256", "--rans", "sa-neg", "--noslip", "--out", "x"},
               "--rans", "--rans + --noslip");
    rejectedAt({"flow", "--physics", "compressible", "--body", "none", "--n",
                "256", "--rans", "sa-neg", "--out", "x"},
               "--rans", "--rans + --body none");
    rejectedAt({"flow", "--physics", "compressible", "--body", "cylinder",
                "--n", "256", "--rans", "sa-neg", "--penalize", "0.01", "--out",
                "x"},
               "--rans", "--rans + --penalize");
    rejectedAt({"flow", "--physics", "compressible", "--body", "cylinder",
                "--n", "256", "--rans", "sa-neg", "--pot", "--out", "x"},
               "--rans", "--rans + --pot");
    rejectedAt(og({"--rans-nutinf", "4"}), "--rans-nutinf",
               "--rans-nutinf without --rans");
    rejectedAt(og({"--rans", "sst-vm", "--rans-nutinf", "4"}), "--rans-nutinf",
               "--rans-nutinf with SST");
    rejectedAt(og({"--rans", "sa-neg", "--rans-kinf", "1e-6"}), "--rans-kinf",
               "--rans-kinf with SA");
    rejectedAt(og({"--rans", "sa-neg", "--rans-mutinf", "0.01"}),
               "--rans-mutinf", "--rans-mutinf with SA");
    rejectedAt(og({"--rans", "sa-neg", "--rans-nutinf", "0"}), "--rans-nutinf",
               "--rans-nutinf 0");
    rejectedAt(og({"--rans", "sst-2003m", "--rans-kinf", "-1e-7"}),
               "--rans-kinf", "--rans-kinf < 0");
    rejected(og({"--rans", "sst-2003m", "--rans-kinf", "nan"}), 1,
             "--rans-kinf nan: not a number (CLI, exit 1)");
    rejectedAt(og({"--rans", "sst-2003m", "--rans-mutinf", "-1"}),
               "--rans-mutinf", "--rans-mutinf -1");
    // Compressible steady.
    rejectedAt({"flow", "--physics", "compressible", "--body", "cylinder",
                "--n", "256", "--ny", "96", "--steady", "1e-4", "--out", "x"},
               "--steady", "compressible --steady without --ogrid or --rans");
    rejectedAt(og({"--steady", "1e-7"}), "--steady", "--steady below 1e-6");
    rejectedAt(og({"--steady", "1"}), "--steady", "--steady >= 1");
    for (const char* f : {"--ramp", "--kick", "--ft"})
        rejectedAt(og({"--rans", "sa-neg", "--steady", "1e-4", f, "1"}), f,
                   std::string("compressible --steady + ") + f);
    rejectedAt(og({"--steady", "1e-4", "--emit", "png"}), "--emit",
               "compressible --steady + time-keyed --emit");
    rejectedAt(og({"--steady", "1e-4", "--steady-steps", "0"}),
               "--steady-steps", "--steady-steps 0");
    rejectedAt(og({"--steady-steps", "100"}), "--steady-steps",
               "--steady-steps without --steady");
    rejectedAt(og({"--steady-scheme", "explicit"}), "--steady-scheme",
               "--steady-scheme without --steady");
    rejectedAt({"flow", "--body", "cylinder", "--steady", "1e-3",
                "--steady-steps", "10", "--out", "x"},
               "--steady-steps", "--steady-steps on incompressible flow");
    rejectedAt({"flow", "--physics", "compressible", "--body", "cylinder",
                "--n", "256", "--rans", "sa-neg", "--steady", "1e-4",
                "--steady-scheme", "implicit", "--out", "x"},
               "--steady-scheme", "implicit on an immersed body (no lines)");
    rejectedAt(og({"--steady", "1e-4", "--steady-scheme", "implicit", "--nz",
                   "4"}),
               "--steady-scheme", "implicit on an extruded 3D O-grid (P4)");
    rejectedAt(og({"--steady", "1e-4", "--steady-scheme", "implicit", "--cfl",
                   "0.3"}),
               "--cfl", "implicit + --cfl");
    rejectedAt({"flow", "--physics", "compressible", "--body", "cylinder",
                "--n", "256", "--emit", "surface", "--out", "x"},
               "--emit", "--emit surface without --ogrid");
    rejected({"channel", "--emit", "surface"}, 2, "--emit surface on channel");
}

// The flow Newton-Krylov march (P3) and immersed RANS (P5): accepted where
// built, with their labels, run lines and RunMeta.
void builtLater()
{
    auto nk = runCli(og({"--steady", "1e-4", "--steady-scheme", "implicit"}));
    check(nk && nk->steadySteps == dns::kImplicitSteadySteps &&
              nk->meta.ransMode == "steady-implicit-nk" &&
              dns::flowRunLabel(*nk) ==
                  "DNS -- steady (implicit Newton-Krylov pseudo-time: not "
                  "time-accurate)" &&
              nk->describe().find("implicit Newton-Krylov (CFL 10 x2 -> 1e8, "
                                  "GMRES(40)") != std::string::npos,
          "--steady-scheme implicit on the O-grid: cap 500, label, run line, "
          "RunMeta steady-implicit-nk");
    auto sa = runCli(og({"--rans", "sa-neg", "--steady", "1e-4",
                         "--steady-scheme", "implicit"}));
    check(sa && dns::flowRunLabel(*sa) ==
                    "RANS SA-neg -- steady (implicit Newton-Krylov "
                    "pseudo-time: not time-accurate)",
          "SA + implicit: the RANS label");
    const Args imm = {"flow", "--physics", "compressible", "--body", "cylinder",
                      "--nz", "1", "--n", "384", "--re", "1000", "--out", "x"};
    Args a = imm;
    a.insert(a.end(), {"--rans", "sst-2003m"});
    auto ic = runCli(a);
    check(ic && dns::flowRunLabel(*ic) ==
                    "RANS SST-2003m -- URANS (time-accurate) [immersed ghost "
                    "wall: first-order wall position, NOT body-fitted]" &&
              ic->describe().find("immersed ghost wall, the velocity's linear "
                                  "image") != std::string::npos &&
              hasWarning(*ic, "refine --n or cluster with --focal"),
          "immersed RANS: label suffix, wall line, y+ warning from h_fine");
    Args b = imm;
    b.insert(b.end(), {"--rans", "sa-neg", "--steady", "1e-4"});
    auto is = runCli(b);
    check(is && is->compSteady && is->meta.ransMode == "steady-explicit",
          "immersed RANS --steady: the explicit pseudo-time march");
    Args l = imm;
    l.insert(l.end(), {"--body", "lens", "--rans", "sa-neg"});
    auto il = runCli(l);
    check(il && hasWarning(*il, "not a true distance"),
          "immersed RANS on a lens warns: d = |sdf| approximate");
}

void warnings()
{
    auto c = runCli(og({"--rans", "sa-neg", "--re", "1e5"}));
    check(c && hasWarning(*c, "first-cell y+"), "warns: y+ estimate > 1");
    auto d = runCli(og({"--rans", "sa-neg", "--rans-nutinf", "10"}));
    check(d && hasWarning(*d, "outside TMR's [3, 5]"), "warns: nu~/nu > 5");
    auto e = runCli(og({"--rans", "sst-vm", "--rans-mutinf", "0.1"}));
    check(e && hasWarning(*e, "Spalart-Rumsey"), "warns: mu_t/mu > 1e-2");
    auto f = runCli(og({"--rans", "sa-neg", "--nz", "8"}));
    check(f && hasWarning(*f, "--nz 1 is the same answer"),
          "warns: --nz > 1 on a z-extruded O-grid");
    auto g = runCli(og({"--rans", "sa-neg"}));
    check(g && !hasWarning(*g, "y+") && !hasWarning(*g, "TMR"),
          "no warning on a resolved default case");
}

// The revolved sphere O-grid (P2a): --nz 1 is the axisymmetric wedge.
void revolved()
{
    const Args sph = {"flow", "--physics", "compressible", "--body", "sphere",
                      "--ogrid", "--n", "64", "--ny", "48", "--out", "s"};
    auto with = [&](Args extra) {
        Args a = sph;
        a.insert(a.end(), extra.begin(), extra.end());
        return a;
    };
    auto c = runCli(with({"--nz", "1"}));
    check(c && c->ogridRevolved && c->grid.nx == 66 && c->grid.nz == 3 &&
              c->ogridWedgeDeg == 5 && c->meta.wedgeDeg == 5,
          "sphere --ogrid --nz 1: the 5-degree wedge, grid 66 x 48 x 3");
    if (c) {
        const std::string d = c->describe();
        check(d.find("REVOLVED about x") != std::string::npos &&
                  d.find("BODY-FITTED") == std::string::npos,
              "wedge describe: the revolved block, not the extruded one");
        check(d.find("  label      DNS [axisymmetric wedge 5 deg: no swirl, "
                     "no 3D wake modes]; axisymmetric unsteadiness only -- NOT "
                     "the sphere's physical shedding\n") != std::string::npos,
              "wedge DNS time march: labelled axisymmetric");
    }
    auto r = runCli(with({"--nz", "1", "--rans", "sst-2003m", "--steady",
                          "1e-5", "--ogrid-wedge", "2.5"}));
    check(r && r->ogridWedgeDeg == 2.5 &&
              r->describe().find("  label      RANS SST-2003m -- steady (local "
                                 "time stepping: pseudo-time, not "
                                 "time-accurate) [axisymmetric wedge 2.5 deg: "
                                 "ensemble mean, no swirl, no 3D wake "
                                 "modes]\n") != std::string::npos,
          "wedge RANS steady label with its angle");
    check(r && !hasWarning(*r, "--nz 1 is the same answer"),
          "no z-invariance warning on the wedge (nz 3 is internal)");
    // The full revolution (P4): --nz N > 1, N even and >= 4.
    auto f = runCli(with({"--nz", "32"}));
    check(f && f->ogridRevolved && f->ogridNphi == 32 && f->grid.nx == 66 &&
              f->grid.ny == 48 && f->grid.nz == 32 && f->nzInterior == 32 &&
              f->ogridWedgeDeg == 0 && f->meta.wedgeDeg == 0 &&
              std::abs(f->grid.hz - 2 * std::numbers::pi / 32) < 1e-15,
          "sphere --ogrid --nz 32: the full revolution, grid 66 x 48 x 32");
    if (f) {
        const std::string d = f->describe();
        check(d.find("FULL revolution 32 azimuthal cells") != std::string::npos &&
                  d.find("axisymmetric WEDGE") == std::string::npos &&
                  d.find("  label      DNS [full revolution 32: odd-mode trip "
                         "0.05 U]\n") != std::string::npos,
              "full revolution describe: its block, no wedge, the DNS time "
              "march labelled with its resolved trip");
        check(f->ogridTrip == dns::kOgridTripDefault && dns::kOgridTripDefault == 0.05 &&
                  d.find("odd-mode trip 0.05 U (solenoidal, 1 D behind the "
                         "body; the default)") != std::string::npos &&
                  d.find("initial state rest (quiet-start ramp)") !=
                      std::string::npos,
              "the trip is ON by default (0.05 U), printed as the default; the "
              "initial state line");
        check(!hasWarning(*f, "polygon") && !hasWarning(*f, "pi about the axis"),
              "--nz 32 at Re 100: no facet warning (0.64% < 1%), no "
              "symmetry warning (tripped)");
    }
    auto fd = runCli(sph);
    check(fd && fd->ogridNphi == 64 && fd->grid.nz == 64,
          "sphere --ogrid without --nz: the full revolution, N = --n");
    auto fr = runCli(with({"--nz", "16", "--rans", "sa-neg", "--steady",
                           "1e-4", "--re", "1e4"}));
    check(fr && fr->describe().find("  label      RANS SA-neg -- steady "
                                    "(local time stepping: pseudo-time, not "
                                    "time-accurate) [full revolution 16: "
                                    "odd-mode trip 0.05 U]\n") !=
                    std::string::npos,
          "full revolution RANS steady label: the full-revolution suffix, no "
          "wedge suffix");
    check(fr && hasWarning(*fr, "polygon about the axis, its frontal area "
                                "2.55% below"),
          "--nz 16 warns: a 16-gon, frontal area 2.55% below the circle's");
    check(fr && !hasWarning(*fr, "rotation by pi"),
          "full revolution RANS with the default trip: no symmetry warning");
    auto f0 = runCli(with({"--nz", "16", "--rans", "sa-neg", "--steady",
                           "1e-4", "--re", "1e4", "--ogrid-trip", "0"}));
    check(f0 && f0->ogridTrip == 0 &&
              hasWarning(*f0, "--ogrid-trip 0: the full-revolution O-grid") &&
              f0->describe().find("[full revolution 16: untripped, exactly "
                                  "pi-symmetric]") != std::string::npos,
          "an explicit --ogrid-trip 0 under RANS: warned, labelled untripped");
    auto f0l = runCli(with({"--nz", "16", "--ogrid-trip", "0"}));
    check(f0l && !hasWarning(*f0l, "--ogrid-trip 0"),
          "an explicit --ogrid-trip 0 at laminar Re 100: no warning (stable)");
    auto f0h = runCli(with({"--nz", "16", "--ogrid-trip", "0", "--re", "200"}));
    check(f0h && hasWarning(*f0h, "--ogrid-trip 0"),
          "an explicit --ogrid-trip 0 at Re 200: warned");
    auto ft = runCli(with({"--nz", "16", "--re", "300", "--ogrid-trip",
                           "0.02"}));
    check(ft && ft->ogridTrip == 0.02 &&
              !hasWarning(*ft, "rotation by pi") &&
              ft->describe().find("odd-mode trip 0.02 U (solenoidal, 1 D "
                                  "behind the body)\n") != std::string::npos,
          "--ogrid-trip 0.02: resolved, described (not the default), no "
          "symmetry warning");
    auto fu = runCli(with({"--nz", "16", "--rans", "sst-2003m", "--re", "1e4"}));
    check(fu && fu->describe().find("  label      RANS SST-2003m -- URANS "
                                    "(time-accurate) [full revolution 16: "
                                    "odd-mode trip 0.05 U]\n") !=
                    std::string::npos,
          "full revolution URANS label: no axisymmetric-unsteadiness tail");
    // --init-from: the wedge start (the file is main's, phase 3).
    auto fi = runCli(with({"--nz", "16", "--steady", "1e-4", "--init-from",
                           "runs/w5"}));
    check(fi && fi->ogridInitFrom == "runs/w5" && fi->ramp == 0 &&
              fi->describe().find("initial state the converged wedge "
                                  "runs/w5_final.bin revolved over the 16 "
                                  "planes") != std::string::npos &&
              fi->describe().find("started from the wedge runs/w5]") !=
                  std::string::npos,
          "--init-from: resolved, ramp 0, described and labelled");
    auto fit = runCli(with({"--nz", "16", "--init-from", "runs/w5"}));
    check(fit && fit->ramp == 0 && fit->compSteady == false,
          "--init-from on a time march: impulsive from the wedge's state");
    rejectedAt(with({"--nz", "16", "--init-from", "w", "--ramp", "1"}),
               "--ramp", "--ramp with --init-from");
    rejectedAt(with({"--nz", "1", "--init-from", "w"}), "--init-from",
               "--init-from on the wedge");
    rejectedAt(og({"--init-from", "w"}), "--init-from",
               "--init-from on an extruded O-grid");
    rejectedAt({"flow", "--physics", "compressible", "--body", "sphere",
                "--n", "64", "--init-from", "w", "--out", "x"},
               "--init-from", "--init-from without --ogrid");
    rejectedAt(with({"--nz", "16", "--init-from", "a\"b"}), "--init-from",
               "--init-from with a quote");
    rejectedAt(with({"--nz", "7"}), "--nz", "an odd azimuthal count");
    rejectedAt(with({"--nz", "2"}), "--nz", "--nz 2 (below 4)");
    rejectedAt(with({"--nz", "8", "--ogrid-wedge", "5"}), "--ogrid-wedge",
               "--ogrid-wedge on the full revolution");
    rejectedAt(with({"--nz", "8", "--les", "0.1"}), "--les",
               "--les on the full revolution");
    rejectedAt(with({"--nz", "8", "--steady", "1e-4", "--steady-scheme",
                     "implicit"}),
               "--steady-scheme", "Newton-Krylov on the full revolution");
    rejectedAt(with({"--nz", "8", "--ogrid-trip", "0.6"}), "--ogrid-trip",
               "--ogrid-trip > 0.5");
    rejectedAt(with({"--nz", "1", "--ogrid-trip", "0.1"}), "--ogrid-trip",
               "--ogrid-trip on the wedge");
    rejectedAt(og({"--ogrid-trip", "0.1"}), "--ogrid-trip",
               "--ogrid-trip on an extruded O-grid");
    rejectedAt({"flow", "--physics", "compressible", "--body", "sphere",
                "--n", "64", "--ogrid-trip", "0.1", "--out", "x"},
               "--ogrid-trip", "--ogrid-trip without --ogrid");
    rejectedAt(with({"--nz", "1", "--ogrid-wedge", "12"}), "--ogrid-wedge",
               "--ogrid-wedge > 10");
    rejectedAt(with({"--nz", "1", "--ogrid-wedge", "0"}), "--ogrid-wedge",
               "--ogrid-wedge 0");
    rejectedAt(og({"--ogrid-wedge", "5"}), "--ogrid-wedge",
               "--ogrid-wedge on an extruded O-grid");
    rejectedAt(with({"--nz", "1", "--les", "0.1"}), "--les",
               "--les on the axisymmetric wedge");
    rejectedAt({"flow", "--physics", "compressible", "--body", "sphere",
                "--n", "64", "--nz", "1", "--out", "x"},
               "--nz", "a Cartesian sphere still needs nz > 1");
}

} // namespace

int main()
{
    revolved();
    headDescribes();
    accepts();
    rejects();
    builtLater();
    warnings();
    return dnstest::finish();
}
