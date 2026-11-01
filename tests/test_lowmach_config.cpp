// dns flame --physics lowmach, config layer (masterplan real_bunsen S4;
// docs/chem.md "Low-Mach"): the resolved case (box units Tu = P0 = 1, no
// Mach, no sink, the FFT-size rule), the label and dry-run text, every
// rejection BY NAME (code 2), --physics lowmach outside dns flame, the
// --chem-dt default = kLmChemDtDefault, and the IC (lowmachIC: rho T = 1, the
// Shvab-Zeldovich species, the 1D momentum balance, Z = 1 for the C2 jet).
#include "dns/chem.hpp"
#include "dns/config.hpp"
#include "dns/lowmach.hpp"
#include "check.hpp"
#include "cli_test_util.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <format>
#include <string>
#include <vector>

namespace {

using dnstest::check;
using dnstest::runCli;

void rejects(std::vector<std::string> args, const std::string& flag,
             const std::string& what, const std::string& text = {})
{
    auto c = runCli(std::move(args));
    const bool ok = !c && c.error().code == 2 &&
                    (c.error().field == flag ||
                     c.error().message.find(flag) != std::string::npos) &&
                    (text.empty() || c.error().message.find(text) != std::string::npos);
    if (!ok && !c)
        std::printf("    got code %d field '%s': %s\n", c.error().code,
                    c.error().field.c_str(), c.error().message.c_str());
    check(ok, what);
}

} // namespace

int main()
{
    using namespace dns;
    // --- the FFT-size helpers ---
    check(lmFftSmooth(1) && lmFftSmooth(2) && lmFftSmooth(168) &&
              lmFftSmooth(2 * 3 * 5 * 7 * 8) && !lmFftSmooth(11) &&
              !lmFftSmooth(165) && !lmFftSmooth(176),
          "lmFftSmooth: prime factors <= 7");
    check(lmNextSmooth(165) == 168 && lmPrevSmooth(165) == 162 &&
              lmNextSmooth(11) == 12 && lmPrevSmooth(11) == 10 &&
              lmNextSmooth(176) == 180 && lmNextSmooth(64) == 64,
          "lmNextSmooth / lmPrevSmooth");

    // --- the default low-Mach case ---
    {
        auto c = runCli({"flame", "--physics", "lowmach"});
        check(c.has_value(), "flame --physics lowmach validates");
        if (c) {
            const double dth = c->flameEig.dthOverDD;
            check(c->physics == Physics::LowMach && c->flameTu == 1.0 &&
                      c->mach == 0 && c->nscbcSink == 0,
                  "box units Tu = P0 = 1, no Mach, no sink");
            check(c->grid.nx == 336 && c->lmNxPad == 6 && c->grid.nx % 2 == 0 &&
                      lmFftSmooth(c->grid.nx / 2),
                  "nx 330 -> 336 (nx/2 = 168 7-smooth), 6 cells downstream");
            check(std::abs(c->flameXf0 - 12 * dth) < 1e-12,
                  "x_f0 = 12 delta_th (no sink band)");
            check(c->cfl == kLmCflDefault && c->lmChemDt == kLmChemDtDefault,
                  "cfl and --chem-dt defaults (kLmCflDefault, kLmChemDtDefault)");
            const std::string l = flameRunLabel(*c);
            check(l.find("LOW-MACH M->0 limit") != std::string::npos &&
                      l.find("M_flame") == std::string::npos &&
                      l.find("PROJECTED mass flux") != std::string::npos,
                  "label names the low-Mach model, no flame Mach");
            const std::string d = c->describe();
            check(d.find("--physics lowmach") != std::string::npos &&
                      d.find("LOW-MACH (M -> 0)") != std::string::npos &&
                      d.find("nx rounded up by 6 cells") != std::string::npos &&
                      d.find("diffusive") != std::string::npos &&
                      d.find("--chem-dt 1") != std::string::npos &&
                      d.find(l) != std::string::npos,
                  "dry-run: solver line, FFT rounding, dt bounds, label");
            check(c->meta.physics == std::string("lowmach"),
                  "json physics = lowmach");
        }
        // The compressible default is untouched by the new code.
        auto comp = runCli({"flame"});
        check(comp && comp->physics == Physics::Compressible &&
                  comp->grid.nx == 362 &&
                  flameRunLabel(*comp).find("M_flame=0.01 (inflated; CH4/air "
                                            "1.15e-3)") != std::string::npos,
              "flame default stays compressible, its label unchanged");
    }
    // 3D and 2D sizes; a non-FFT-friendly ny / nz is an error naming the
    // nearest sizes.
    {
        auto c3 = runCli({"flame", "--physics", "lowmach", "--ny", "16", "--nz", "16",
                          "--cells-per-delta", "8"});
        check(c3 && c3->grid.nz == 16 && c3->grid.nx == 180 && c3->lmNxPad == 4,
              "3D: nz 16, nx 176 -> 180");
        rejects({"flame", "--physics", "lowmach", "--ny", "11"}, "--ny",
                "--ny 11 rejected (FFT size)", "nearest 10 or 12");
        rejects({"flame", "--physics", "lowmach", "--ny", "16", "--nz", "22"},
                "--nz", "--nz 22 rejected (FFT size)", "nearest 21 or 24");
    }
    // --chem-dt.
    {
        auto k = runCli({"flame", "--physics", "lowmach", "--chem-dt", "2"});
        check(k && k->lmChemDt == 2, "--chem-dt 2 accepted");
        rejects({"flame", "--physics", "lowmach", "--chem-dt", "0"}, "--chem-dt",
                "--chem-dt 0 rejected");
        rejects({"flame", "--physics", "lowmach", "--chem-dt", "3"}, "--chem-dt",
                "--chem-dt 3 rejected (above 2.5)");
        rejects({"flame", "--chem-dt", "1"}, "--chem-dt",
                "--chem-dt without --physics lowmach rejected");
    }
    // Everything the low-Mach solver does not have, by name.
    for (const char* f : {"--mach", "--nscbc-etain", "--nscbc-sink",
                          "--nscbc-sink-amp", "--les", "--tfles", "--turb"}) {
        std::vector<std::string> a = {"flame", "--physics", "lowmach", f,
                                      std::string(f) == "--mach" ? "0.01"
                                      : std::string(f) == "--nscbc-sink-amp"
                                          ? "1"
                                      : std::string(f) == "--tfles" ? "2"
                                      : std::string(f) == "--les" ? "0.2"
                                      : std::string(f) == "--turb" ? "2"
                                                                   : "16"};
        rejects(a, f, std::format("{} rejected with --physics lowmach", f));
    }
    rejects({"flame", "--physics", "lowmach", "--nscbc-tinflow"}, "--nscbc-tinflow",
            "--nscbc-tinflow rejected with --physics lowmach", "face conditions");
    for (const char* m : {"diffusion", "candle", "chamber"})
        rejects({"flame", "--physics", "lowmach", "--mode", m}, "--mode",
                std::format("--mode {} rejected with --physics lowmach", m));
    rejects({"flame", "--physics", "incompressible"}, "--physics",
            "flame --physics incompressible rejected");
    // --physics lowmach outside dns flame.
    for (std::vector<std::string> a :
         {std::vector<std::string>{"flow", "--body", "cylinder", "--physics", "lowmach"},
          {"conv", "--physics", "lowmach"},
          {"bench", "--physics", "lowmach"}})
        rejects(a, "--physics", a[0] + " --physics lowmach rejected",
                "dns flame only");

    // --- the IC ---
    {
        auto c = runCli({"flame", "--physics", "lowmach", "--inflow-speed", "0.98",
                         "--cells-per-delta", "8"});
        check(c.has_value(), "IC case validates");
        if (c) {
            const LowMachState s = lowmachIC(*c);
            const Grid& g = c->grid;
            const double sig = c->flameModel.sigma;
            bool eos = true, sz = true;
            for (std::size_t q = 0; q < g.size(); ++q) {
                const double T = 1 / s.rho[q], Y = s.rhoY[0][q] / s.rho[q];
                sz = sz && std::abs(T + (sig - 1) * Y - sig) < 1e-12;
                eos = eos && T >= 1 - 1e-12 && T <= sig + 1e-12;
            }
            check(eos && sz, "IC: T = 1/rho in [1, sigma], H = T + (sigma-1) Y = sigma");
            check(s.mx.size() == std::size_t(g.nx + 1) * g.ny * g.nz &&
                      s.mx[0] == 0.98 && s.mx[std::size_t(g.nx)] == 0.98,
                  "IC: nx + 1 x faces per row, m = u_in");
            const double p0 = s.p[0], pEnd = s.p[std::size_t(g.nx - 1)];
            check(pEnd == 0 && std::abs(p0 - 0.98 * 0.98 * (sig - 1)) < 1e-6,
                  std::format("IC: p = m^2 (T_last - T): p_end 0, p_0 {:.8g}", p0));
        }
        auto r = runCli({"flame", "--physics", "lowmach", "--equivalence", "1.3",
                         "--expansion", "6.88", "--cells-per-delta", "8"});
        check(r.has_value(), "--equivalence with --physics lowmach validates");
        if (r) {
            const LowMachState s = lowmachIC(*r);
            double zmax = 0;
            for (std::size_t q = 0; q < r->grid.size(); ++q) {
                const double z = r->premix.mixtureFraction(s.rhoY[0][q] / s.rho[q],
                                                           s.rhoY[1][q] / s.rho[q]);
                zmax = std::max(zmax, std::abs(z - 1));
            }
            check(s.nY == 2 && zmax < 1e-12,
                  std::format("C2 IC: two reactants, Z = 1 (max dev {:.2e})", zmax));
            check(flameRunLabel(*r).find("LOW-MACH") != std::string::npos &&
                      flameRunLabel(*r).find("S_L(phi) law C2b") != std::string::npos,
                  "C2 label: low-Mach and the C2b law");
        }
    }

    // --- S5: --mode bunsen --physics lowmach ---
    {
        auto c = runCli({"flame", "--mode", "bunsen", "--physics", "lowmach"});
        check(c.has_value(), "bunsen --physics lowmach validates");
        if (c) {
            check(c->flameBunsen && c->grid.nx == 360 && c->grid.ny == 288 &&
                      c->lmNxPad == 0 && c->lmNyPad == 0 && c->nscbcSink == 0 &&
                      c->dtRecompute == 1 && c->bunsen.tCo == c->flameModel.sigma,
                  "G2a burner: 360 x 288 (no bands), burnt coflow, dt every step");
            const std::string d = c->describe();
            check(d.find("2D slot Bunsen (dns flame --mode bunsen --physics "
                         "lowmach)") != std::string::npos &&
                      d.find("x-lo inflow FACE per row") != std::string::npos &&
                      d.find("frames     temp, fuel, omega, speed") !=
                          std::string::npos &&
                      d.find(flameRunLabel(*c)) != std::string::npos,
                  "bunsen dry-run: mode, face inflow, frames, label");
            const LowMachState s = lowmachIC(*c);
            const Grid& g = c->grid;
            const double sig = c->flameModel.sigma;
            bool sz = true, rows = true;
            for (int j = 0; j < g.ny; ++j) {
                const LmInflowRow in = lmInflowRow(*c, (j + 0.5) * g.hy, 0.5);
                for (int i = 0; i <= g.nx; ++i)
                    rows = rows && s.mx[std::size_t(i) + std::size_t(g.nx + 1) * j] ==
                                       in.m;
                rows = rows && std::abs(in.m * in.t - in.jet * 5 - in.co * 1) < 1e-12;
            }
            for (std::size_t q = 0; q < g.size(); ++q) {
                const double T = 1 / s.rho[q], Y = s.rhoY[0][q] / s.rho[q];
                sz = sz && std::abs(T + (sig - 1) * Y - sig) < 1e-9;
            }
            check(rows, "bunsen IC: every x face carries its row's inflow m = u/t");
            check(sz, "bunsen IC (burnt coflow): H = T + (sigma-1) Y = sigma");
        }
        // The --equivalence burner: air coflow, rich-product pilots, the FFT
        // rounding of ny.
        auto r = runCli({"flame", "--mode", "bunsen", "--physics", "lowmach",
                         "--equivalence", "1.3", "--expansion", "6.88", "--visc",
                         "sutherland", "--tinf", "298", "--slot-width", "4",
                         "--jet-speed", "3", "--pilot-width", "1.5",
                         "--pilot-speed", "3", "--coflow-width", "6"});
        check(r.has_value(), "partially premixed burner validates");
        if (r) {
            const PartialPremixModel& pm = r->premix;
            check(r->bunsen.tCo == 1.0 && r->bunsen.tPilot == pm.sigma &&
                      lmFftSmooth(r->grid.ny) && lmFftSmooth(r->grid.nx / 2),
                  std::format("air coflow T 1, pilots at sigma, ny {} nx {} "
                              "7-smooth (ny +{})",
                              r->grid.ny, r->grid.nx, r->lmNyPad));
            const std::string l = flameRunLabel(*r);
            check(l.find("AIR coflow") != std::string::npos &&
                      l.find("rich-product pilot strips") != std::string::npos &&
                      l.find("flame index (Takeno)") != std::string::npos,
                  "partially premixed label: air, pilots, flame index");
            check(r->describe().find("findex (Takeno)") != std::string::npos,
                  "partially premixed dry-run lists the findex frame");
            // Rows and IC: Z = J + P (1 jet and products, 0 air), never a
            // products/air reactive overlap (Burke-Schumann laid).
            const LowMachState s = lowmachIC(*r);
            const Grid& g = r->grid;
            double zdev = 0, overlap = 0;
            for (int j = 0; j < g.ny; ++j) {
                const LmInflowRow in = lmInflowRow(*r, (j + 0.5) * g.hy, 0.5);
                zdev = std::max(zdev, std::abs(pm.mixtureFraction(in.yF, in.yO) -
                                               (in.jet + in.pilot)));
                if (in.jet < 1e-12) overlap = std::max(overlap, in.yF * in.yO);
            }
            double zmin = 1, zmax = 0;
            for (std::size_t q = 0; q < g.size(); ++q) {
                const double z = pm.mixtureFraction(s.rhoY[0][q] / s.rho[q],
                                                    s.rhoY[1][q] / s.rho[q]);
                zmin = std::min(zmin, z);
                zmax = std::max(zmax, z);
            }
            check(zdev < 1e-12, "rows: Z = J + P");
            check(overlap < 1e-12, "rows: no fuel/oxidizer overlap outside the "
                                   "jet (the pilot/air edge enters burnt)");
            check(zmin > -1e-12 && zmax < 1 + 1e-12, "IC: Z in [0, 1]");
        }
        // The round nozzle's partially premixed IC: reactants in [0, 1], Z
        // in [0, 1], T in [1, sigma_st] (3D, the profile at r(y, z)).
        auto rd = runCli({"flame", "--mode", "bunsen", "--physics", "lowmach",
                          "--nozzle", "round", "--nz", "64", "--equivalence", "1.3",
                          "--expansion", "6.88", "--visc", "sutherland", "--tinf",
                          "298", "--cells-per-delta", "4", "--slot-width", "8",
                          "--pilot-width", "2", "--pilot-speed", "5",
                          "--coflow-width", "2"});
        check(rd.has_value(), "round partially premixed burner validates");
        if (rd) {
            const LowMachState s = lowmachIC(*rd);
            double yfMin = 1, yoMin = 1, tMin = 10, tMax = 0;
            for (std::size_t q = 0; q < rd->grid.size(); ++q) {
                yfMin = std::min(yfMin, s.rhoY[0][q] / s.rho[q]);
                yoMin = std::min(yoMin, s.rhoY[1][q] / s.rho[q]);
                tMin = std::min(tMin, 1 / s.rho[q]);
                tMax = std::max(tMax, 1 / s.rho[q]);
            }
            std::printf("  round IC: Y_F >= %.3g, Y_O >= %.3g, T in [%.6g, %.6g]\n",
                        yfMin, yoMin, tMin, tMax);
            check(yfMin >= 0 && yoMin >= 0 && tMin >= 1 - 1e-12 &&
                      tMax <= rd->premix.sigmaSt() + 1e-9,
                  "round IC: reactants >= 0, T in [1, sigma_st]");
        }
        // Rejections.
        rejects({"flame", "--mode", "bunsen", "--equivalence", "1.3"},
                "--equivalence", "--equivalence bunsen compressible rejected",
                "--physics lowmach");
        rejects({"flame", "--mode", "bunsen", "--physics", "lowmach", "--pulse",
                 "0.1", "--pulse-freq", "1"},
                "--pulse", "--pulse rejected with the low-Mach burner");
        rejects({"flame", "--mode", "bunsen", "--physics", "lowmach", "--equivalence",
                 "1.3", "--mix-phi", "0.8", "--expansion", "6.88"},
                "--mix-phi", "--mix-phi rejected on the burner");
        rejects({"flame", "--mode", "bunsen", "--physics", "lowmach", "--nozzle",
                 "round", "--nz", "60"},
                "--nozzle", "round needs nz = ny");
        auto w = runCli({"flame", "--mode", "bunsen", "--physics", "lowmach",
                         "--equivalence", "1.3", "--expansion", "6.88"});
        check(w && std::ranges::any_of(w->warnings,
                                       [](const std::string& s) {
                                           return s.find("without --pilot-width") !=
                                                  std::string::npos;
                                       }),
              "--equivalence burner without pilots warns");
    }

    // --- S6: --laterals open, --reaction off, the burner's --profile-dt ---
    {
        const std::vector<std::string> pp = {"flame", "--mode", "bunsen", "--physics",
                                              "lowmach", "--equivalence", "1.3",
                                              "--expansion", "6.88", "--pilot-width", "1.5"};
        auto per = runCli(pp);
        auto withLat = [&](std::vector<std::string> a, std::vector<std::string> more) {
            a.insert(a.end(), more.begin(), more.end());
            return a;
        };
        auto op = runCli(withLat(pp, {"--laterals", "open"}));
        check(per && op && !per->lmOpenY && !per->lmOpenZ && op->lmOpenY &&
                  !op->lmOpenZ && !op->lmInert,
              "--laterals open: the slot opens y only (z its periodic span)");
        if (per && op) {
            check(per->grid.nx == op->grid.nx && per->grid.ny == op->grid.ny &&
                      per->bunsen.yc == op->bunsen.yc,
                  "--laterals open: the same box (--coflow-width = the clearance)");
            const std::string lp = flameRunLabel(*per), lo = flameRunLabel(*op);
            check(lp.find("OPEN") == std::string::npos &&
                      lo.find("OPEN laterals (y)") != std::string::npos &&
                      lo.find("air") != std::string::npos &&
                      lo.starts_with(lp),
                  "label: periodic unchanged, open appends 'OPEN laterals (y) ... "
                  "air'");
            const std::string dp = per->describe(), dop = op->describe();
            check(dp.find("y/z periodic, one direct solve") != std::string::npos &&
                      dp.find("no diffusion; y periodic") != std::string::npos &&
                      dop.find("y DD (open), z periodic") != std::string::npos &&
                      dop.find("y OPEN (--laterals open)") != std::string::npos,
                  "dry-run: periodic text unchanged, open names the DD solve and "
                  "the face rules");
            check(op->meta.farField == std::string("lowmach-open") &&
                      per->meta.farField == std::string("lowmach-faces"),
                  "json far field: lowmach-open | lowmach-faces");
        }
        auto rd = runCli({"flame", "--mode", "bunsen", "--physics", "lowmach",
                          "--nozzle", "round", "--nz", "40", "--slot-width", "4",
                          "--coflow-width", "3", "--cells-per-delta", "4", "--laterals",
                          "open"});
        check(rd && rd->lmOpenY && rd->lmOpenZ &&
                  rd->describe().find("y/z DD (open)") != std::string::npos &&
                  flameRunLabel(*rd).find("OPEN laterals (y, z)") != std::string::npos,
              "--laterals open --nozzle round: y and z open");
        auto in = runCli({"flame", "--mode", "bunsen", "--physics", "lowmach",
                          "--reaction", "off", "--coflow-temp", "1", "--laterals",
                          "open", "--profile-dt", "10"});
        check(in && in->lmInert && in->profileDt == 10 &&
                  flameRunLabel(*in).find("INERT jet") != std::string::npos &&
                  flameRunLabel(*in).find("inert Y=0") != std::string::npos,
              "--reaction off on the low-Mach burner: inert, labelled; --profile-dt "
              "accepted (the jet profiles)");
        if (in) {
            // The inert IC: every cell its row's inflow state (no cone).
            const LowMachState s = lowmachIC(*in);
            const Grid& g = in->grid;
            bool rows = true;
            for (int j = 0; j < g.ny; ++j) {
                const LmInflowRow r = lmInflowRow(*in, (j + 0.5) * g.hy, 0.5 * g.hz);
                for (int i = 0; i < g.nx; ++i) {
                    const std::size_t c = g.idx(i, j, 0);
                    rows = rows && s.rho[c] == 1.0 / r.t &&
                           s.rhoY[0][c] == (1.0 / r.t) * r.yF;
                }
            }
            check(rows && s.myHi.empty() && s.mzHi.empty(),
                  "inert IC: the rows extruded, no cone; hi faces empty (0)");
        }
        rejects({"flame", "--mode", "bunsen", "--laterals", "open"}, "--laterals",
                "--laterals rejected on the compressible burner (D6)",
                "--physics lowmach");
        rejects({"flame", "--physics", "lowmach", "--laterals", "open"}, "--laterals",
                "--laterals rejected on the low-Mach planar flame", "--mode bunsen");
        rejects({"flame", "--mode", "bunsen", "--laterals", "periodic"}, "--laterals",
                "--laterals periodic named on the compressible burner rejected too");
        rejects({"flame", "--mode", "bunsen", "--reaction", "off"}, "--reaction",
                "--reaction rejected on the compressible burner",
                "--physics lowmach");
        rejects({"flame", "--physics", "lowmach", "--reaction", "off"}, "--reaction",
                "--reaction rejected on the low-Mach planar flame");
        rejects({"flame", "--mode", "bunsen", "--profile-dt", "1"}, "--profile-dt",
                "--profile-dt rejected on the compressible burner");
        auto bad = runCli({"flame", "--mode", "bunsen", "--physics", "lowmach",
                           "--laterals", "sideways"});
        check(!bad && bad.error().code == 1 &&
                  bad.error().message.find("periodic, open") != std::string::npos,
              "--laterals sideways: a CLI error naming periodic, open (exit 1)");
    }

    // --- S7: --gravity on the low-Mach burner ---
    {
        const std::vector<std::string> pp = {"flame",        "--mode",     "bunsen",
                                              "--physics",    "lowmach",    "--equivalence",
                                              "1.3",          "--expansion", "6.88",
                                              "--pilot-width", "1.5",       "--laterals",
                                              "open"};
        auto with = [&](std::vector<std::string> more) {
            auto a = pp;
            a.insert(a.end(), more.begin(), more.end());
            return a;
        };
        auto off = runCli(pp), z = runCli(with({"--gravity", "0"})),
             on = runCli(with({"--gravity", "0.0241"}));
        check(off && z && on && off->lmGravity == 0 && z->lmGravity == 0 &&
                  on->lmGravity == 0.0241,
              "--gravity: absent 0, explicit 0 is 0, the value carried");
        if (off && z && on) {
            const std::string lo = flameRunLabel(*off), lz = flameRunLabel(*z),
                              lg = flameRunLabel(*on);
            check(lo == lz && lo.find("GRAVITY") == std::string::npos &&
                      lg.starts_with(lo) &&
                      lg.find("GRAVITY g=0.0241 S_L^2/delta_D along -x") !=
                          std::string::npos,
                  "label: g 0 unchanged, gravity appends 'GRAVITY g=... along -x'");
            const std::string d0 = off->describe(), dg = on->describe();
            check(d0.find("gravity") == std::string::npos &&
                      dg.find("Fr = U^2/(g D) = ") != std::string::npos &&
                      dg.find("Ri = g D (1 - 1/sigma)/U^2 = ") != std::string::npos &&
                      dg.find("free-fall u_ff") != std::string::npos,
                  "dry-run: no gravity line at g 0; Fr, Ri, u_ff with gravity");
            check(on->warnings.empty() || std::ranges::none_of(on->warnings, [](auto& w) {
                      return w.find("lattice of plumes") != std::string::npos;
                  }),
                  "--gravity with open laterals: no lattice warning");
        }
        // Real g at the M1 mapping (GRI-3.0 phi 1.3, the model's 4.70964).
        check(std::abs(lmRealGravity(4.70964) - 0.024096) < 1e-5,
              std::format("lmRealGravity(4.70964) = {:.6f} (0.0241)",
                          lmRealGravity(4.70964)));
        auto per = runCli({"flame", "--mode", "bunsen", "--physics", "lowmach",
                           "--gravity", "0.1"});
        check(per && std::ranges::any_of(per->warnings,
                                         [](auto& w) {
                                             return w.find("lattice of plumes") !=
                                                    std::string::npos;
                                         }),
              "--gravity with periodic laterals warns (a lattice of plumes)");
        rejects({"flame", "--mode", "bunsen", "--gravity", "0.1"}, "--gravity",
                "--gravity rejected on the compressible burner (D6)",
                "--physics lowmach");
        rejects({"flame", "--physics", "lowmach", "--gravity", "0.1"}, "--gravity",
                "--gravity rejected on the low-Mach planar flame", "--mode bunsen");
        rejects(with({"--gravity", "-0.1"}), "--gravity", "--gravity < 0 rejected",
                "[0, 100]");
        rejects(with({"--gravity", "101"}), "--gravity", "--gravity > 100 rejected",
                "[0, 100]");
        rejects({"flame", "--mode", "bunsen", "--physics", "lowmach", "--quiescent"},
                "--quiescent", "--quiescent stays the candle's");
    }

    // --- S8: --rim, the immersed isothermal tube ---
    {
        const std::vector<std::string> rr = {
            "flame",       "--mode",      "bunsen", "--physics",  "lowmach",
            "--equivalence", "1.3",       "--expansion", "6.88",  "--slot-width",
            "14",          "--jet-speed", "6",      "--laterals", "open",
            "--coflow-width", "6",        "--cells-per-delta", "4"};
        auto with = [&](std::vector<std::string> more) {
            auto a = rr;
            a.insert(a.end(), more.begin(), more.end());
            return a;
        };
        auto c = runCli(with({"--rim", "1.4"}));
        check(c.has_value(), "--rim 1.4 validates on the low-Mach burner");
        if (c) {
            const BunsenGeom& b = c->bunsen;
            const double dth = c->flameEig.dthOverDD, h = c->h;
            check(c->lmRim && std::abs(b.rimLip - 1.4 * dth) < 1e-12 &&
                      b.xRim == 30 * h && b.tRim == 1 && b.uCo == kLmRoomDraft &&
                      b.pilot == 0 && b.tCo == 1,
                  "--rim: lip, the rim top snapped to 30 cells (7.5 delta_th), "
                  "T_w 1, room draft 0.4, no pilot, cold air");
            check(c->grid.nx == 270 &&
                      std::abs(c->grid.nx * h - (7.5 + 60) * dth) < 1e-9,
                  std::format("nx = tube 7.5 + --length 60 delta_th (nx {})",
                              c->grid.nx));
            // The tube: an exact distance in the (x, r) half plane.
            const double ri = 0.5 * b.W, ro = ri + b.rimLip, yc = b.yc;
            check(std::abs(lmTubeSdf(b, 0.5 * b.xRim, yc + 0.5 * (ri + ro), 0) +
                           0.5 * b.rimLip) < 1e-12 &&
                      std::abs(lmTubeSdf(b, b.xRim + 3, yc + ri - 4, 0) - 5) < 1e-12 &&
                      std::abs(lmTubeSdf(b, 1, yc - ri + 2, 0) - 2) < 1e-12 &&
                      std::abs(lmTubeSdf(b, b.xRim - 1, yc - ro - 0.5, 0) - 0.5) <
                          1e-12,
                  "lmTubeSdf: inside -lip/2, corner 5 (3-4-5), radial 2, outside 0.5");
            const auto solid = lmTubeSolid(b, c->grid);
            long ns = 0;
            for (auto v : solid) ns += v;
            check(ns == 2 * 6 * 30, std::format("staircase: 2 lips x 6 cells x 30 "
                                                "rows = 360 solid ({})",
                                                ns));
            // Poiseuille: mean 1 over the jet cells; the antimirror ghost
            // makes the 3-point Laplacian constant up to the wall cells.
            const auto f = lmPoiseuille(b, c->grid);
            const Grid& g = c->grid;
            double mean = 0, lapMin = HUGE_VAL, lapMax = -HUGE_VAL;
            int n = 0;
            for (int j = 0; j < g.ny; ++j) {
                if (f[std::size_t(j)] == 0) continue;
                mean += f[std::size_t(j)];
                ++n;
                auto at = [&](int q) {
                    const double v = f[std::size_t(q)];
                    return v != 0 ? v : -f[std::size_t(j)]; // the antimirror
                };
                const double lap = at(j - 1) - 2 * f[std::size_t(j)] + at(j + 1);
                lapMin = std::min(lapMin, lap);
                lapMax = std::max(lapMax, lap);
            }
            mean /= n;
            check(n == 56 && std::abs(mean - 1) < 1e-14 &&
                      std::abs(lapMax - lapMin) < 1e-11 * std::abs(lapMin),
                  std::format("lmPoiseuille: 56 jet cells, mean {:.16g}, the "
                              "discrete Laplacian constant to {:.2e} (wall rows "
                              "included)",
                              mean, (lapMax - lapMin) / std::abs(lapMin)));
            // -lap/h^2 = 12/W_e^2 to O(h^2/W^2): the developed dp/dx.
            const double we = 56 * h;
            check(std::abs(-lapMin / (h * h) / (12 / (we * we)) - 1) < 1e-3,
                  std::format("Poiseuille curvature = 12 U/W^2 ({:.6f})",
                              -lapMin / (h * h) / (12 / (we * we))));
            // The rows: Poiseuille in the tube, closed wall, room draft.
            const LmInflowRow a = lmInflowRow(*c, yc + 0.5 * h, 0.5);
            const LmInflowRow w = lmInflowRow(*c, yc + ri + 0.5 * h, 0.5);
            const LmInflowRow d = lmInflowRow(*c, yc + ro + 2 * h, 0.5);
            check(std::abs(a.m - 6 * f[std::size_t(g.ny / 2)]) < 1e-12 && a.t == 1 &&
                      w.m == 0 && w.t == 1 && w.yF == 0 && w.yO == 0 &&
                      d.m == kLmRoomDraft && d.yF == 0 && d.yO == c->premix.yO2,
                  "rows: Poiseuille jet, closed wall (m 0, Y 0), draft of air");
            const std::string l = flameRunLabel(*c);
            check(l.find("RIM: immersed isothermal tube T_w=1Tu") != std::string::npos &&
                      l.find("STAIRCASE") != std::string::npos &&
                      l.find("apertured FP32 PCG") != std::string::npos,
                  "label names the rim, the staircase and the PCG");
            const std::string ds = c->describe();
            check(ds.find("TUBE (--rim)") != std::string::npos &&
                      ds.find("360 solid cells") != std::string::npos,
                  "dry-run prints the tube");
            // The IC: the tube solid at 1/T_w, rho Y 0; faces touching it 0.
            const LowMachState ic = lowmachIC(*c);
            bool ok = true;
            for (int i = 0; i < g.nx; ++i)
                for (int j = 0; j < g.ny; ++j) {
                    const std::size_t q = g.idx(i, j, 0);
                    if (solid[q])
                        ok = ok && ic.rho[q] == 1.0 && ic.rhoY[0][q] == 0 &&
                             ic.rhoY[1][q] == 0;
                    const bool touch = solid[q] || (i > 0 && solid[g.idx(i - 1, j, 0)]);
                    if (touch) ok = ok && ic.mx[std::size_t(i) + (g.nx + 1) * std::size_t(j)] == 0;
                }
            check(ok, "IC: the tube rho 1/T_w, rho Y 0, every x face touching it 0");
        }
        auto off = runCli(rr);
        check(off && !off->lmRim && off->bunsen.rimLip == 0,
              "no --rim: no tube (S7 bit for bit)");
        auto hot = runCli(with({"--rim", "1.4", "--rim-temp", "1.5", "--tube-length",
                                "4", "--coflow-speed", "1"}));
        check(hot && hot->bunsen.tRim == 1.5 && hot->bunsen.uCo == 1 &&
                  hot->bunsen.xRim == 16 * hot->h,
              "--rim-temp, --tube-length, --coflow-speed carried");
        rejects({"flame", "--mode", "bunsen", "--rim", "1.4"}, "--rim",
                "--rim rejected on the compressible burner (D6)", "--physics lowmach");
        rejects({"flame", "--physics", "lowmach", "--rim", "1.4"}, "--rim",
                "--rim rejected on the low-Mach planar flame", "--mode bunsen");
        rejects(with({"--tube-length", "4"}), "--tube-length",
                "--tube-length needs --rim", "needs --rim");
        rejects(with({"--rim-temp", "1.5"}), "--rim-temp", "--rim-temp needs --rim",
                "needs --rim");
        for (const char* f : {"--pilot-width", "--pilot-temp", "--pilot-speed",
                              "--coflow-temp", "--slot-edge"})
            rejects(with({"--rim", "1.4", f, "1"}), f,
                    std::format("{} rejected with --rim", f), "the rim anchors");
        rejects(with({"--rim", "0.9"}), "--rim", "--rim below 4 cells rejected",
                ">= 4 cells");
        rejects(with({"--rim", "1.4", "--tube-length", "1.5"}), "--tube-length",
                "--tube-length below 8 cells rejected", ">= 8 cells");
        rejects(with({"--rim", "1.4", "--rim-temp", "3.5"}), "--rim-temp",
                "--rim-temp above 3 rejected", "[1, 3]");
        // The wall heat: a linear profile against a slab, exact.
        {
            const Grid g = Grid::make(8, 4, 1, 8.0, 4.0, 1.0);
            std::vector<std::uint8_t> s(g.size(), 0);
            std::vector<float> T(g.size(), 1.0f);
            for (int j = 0; j < 4; ++j) {
                for (int i = 4; i < 8; ++i) s[g.idx(i, j, 0)] = 1;
                for (int i = 0; i < 4; ++i) T[g.idx(i, j, 0)] = float(2.0 - (i + 0.5) / 4);
            }
            const double q = lmWallHeat(g, s, T, 3.5, 1.0, true, false);
            check(std::abs(q - 3.5 * (1.0 / 4) * 4) < 1e-6,
                  std::format("lmWallHeat: lambda dT/L x area = 3.5 ({:.9g})", q));
            const double qp = lmWallHeat(g, s, T, 3.5, 1.0, false, false);
            check(qp == q, "lmWallHeat: a slab spanning y has no lateral wall faces");
        }
    }

    // --- S9: --dt-rule (the burner's per-cell positivity bound), the SI echo ---
    {
        const std::vector<std::string> bb = {"flame",       "--mode",   "bunsen",
                                             "--physics",   "lowmach",  "--equivalence",
                                             "1.3",         "--expansion", "6.88",
                                             "--slot-width", "14",      "--jet-speed", "6",
                                             "--rim",       "1.4",      "--laterals", "open",
                                             "--coflow-width", "6",     "--cells-per-delta",
                                             "4"};
        auto with = [&](std::vector<std::string> more) {
            auto a = bb;
            a.insert(a.end(), more.begin(), more.end());
            return a;
        };
        auto b = runCli(bb);
        auto p = runCli({"flame", "--physics", "lowmach"});
        check(b && b->lmDtCell && p && !p->lmDtCell,
              "--dt-rule default: cell on the low-Mach burner, bounds on the planar "
              "flame");
        auto bb2 = runCli(with({"--dt-rule", "bounds"}));
        auto pc = runCli({"flame", "--physics", "lowmach", "--dt-rule", "cell"});
        check(bb2 && !bb2->lmDtCell && pc && pc->lmDtCell,
              "--dt-rule bounds / cell given explicitly");
        if (b && p) {
            const std::string db = b->describe(), dp = p->describe();
            check(db.find("--dt-rule cell: and the per-cell positivity bound") !=
                          std::string::npos &&
                      dp.find("--dt-rule") == std::string::npos,
                  "dry-run names the cell rule; the planar default's text unchanged");
        }
        rejects({"flame", "--dt-rule", "cell"}, "--dt-rule",
                "--dt-rule rejected on the compressible flame", "--physics lowmach");
        check(!runCli({"flame", "--physics", "lowmach", "--dt-rule", "fast"}),
              "--dt-rule fast is a CLI error");
        // The SI echo: label, dry-run, no physics.
        auto si = runCli(with({"--si-delta-d", "0.1312", "--si-sl", "0.23124"}));
        check(si && std::abs(si->siDeltaD - 0.1312e-3) < 1e-18 && si->siSL == 0.23124 &&
                  b &&
                  si->grid.nx == b->grid.nx && si->h == b->h &&
                  si->bunsen.uJet == b->bunsen.uJet,
              "--si-delta-d / --si-sl carried (m, m/s), the case unchanged");
        if (si) {
            const std::string d = si->describe(), l = flameRunLabel(*si);
            check(d.find("SI         delta_D = 0.1312 mm, S_L = 0.23124 m/s") !=
                          std::string::npos &&
                      d.find("Re_D = U D/nu_u") != std::string::npos &&
                      l.find("| SI (echo, no physics): delta_D=0.1312 mm") !=
                          std::string::npos,
                  "the SI echo in the dry-run and the label");
            check(b && flameRunLabel(*b).find("SI (echo") == std::string::npos &&
                      b->describe().find("  SI  ") == std::string::npos,
                  "no SI text without the flags");
        }
        rejects(with({"--si-delta-d", "0.1312"}), "--si-sl",
                "--si-delta-d without --si-sl rejected", "come together");
        rejects(with({"--si-sl", "0.23"}), "--si-delta-d",
                "--si-sl without --si-delta-d rejected", "come together");
        rejects(with({"--si-delta-d", "0", "--si-sl", "0.23"}), "--si-delta-d",
                "--si-delta-d 0 rejected", "(0, 10) mm");
        rejects({"flame", "--physics", "lowmach", "--si-delta-d", "0.13", "--si-sl",
                 "0.23"},
                "--si-delta-d", "the SI echo rejected on the planar flame",
                "low-Mach burner");
        rejects({"flame", "--mode", "bunsen", "--si-delta-d", "0.13", "--si-sl", "0.23"},
                "--si-delta-d", "the SI echo rejected on the compressible burner",
                "low-Mach burner");
    }
    return dnstest::finish();
}
