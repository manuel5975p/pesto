// dns flame --mode diffusion, host side (docs/chem.md, dns/diffusion.hpp):
// the model (Z_st, the Burke-Schumann sheet, Shvab-Zeldovich, the stream
// values), erfcInv, the slot-jet inflow and IC, measureJet on drawn fields,
// the parse + validate of the mode (box, defaults, dry-run, label) and every
// rejection BY NAME.
#include "dns/config.hpp"
#include "dns/diffusion.hpp"
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

std::vector<std::string> diff(std::vector<std::string> extra)
{
    std::vector<std::string> a{"flame", "--mode", "diffusion"};
    a.insert(a.end(), extra.begin(), extra.end());
    return a;
}

} // namespace

int main()
{
    using namespace dns;
    // --- model ---------------------------------------------------------------
    {
        DiffusionModel m;
        m.s = 3;
        m.yF1 = 0.5;
        m.yO2 = 0.8;
        m.sigma = 4;
        m.beta = 6;
        m.Da = 300;
        const double zs = 0.8 / (3 * 0.5 + 0.8);
        check(std::abs(m.zst() - zs) < 1e-15, "Z_st = Y_O2 / (s Y_F1 + Y_O2)");
        check(std::abs(mixtureFraction(m, 0.5, 0) - 1) < 1e-15 &&
                  std::abs(mixtureFraction(m, 0, 0.8)) < 1e-15,
              "Z = 1 in the fuel stream, 0 in the oxidizer");
        const SheetState st = burkeSchumann(m, zs);
        check(st.YF == 0 && std::abs(st.YO) < 1e-15 &&
                  std::abs(st.t - 4) < 1e-13,
              "Burke-Schumann at Z_st: no reactant, T = sigma T0");
        double worstZ = 0, worstH = 0;
        bool bounded = true, linear = true;
        for (int k = 0; k <= 200; ++k) {
            const double Z = k / 200.0;
            const SheetState b = burkeSchumann(m, Z);
            worstZ = std::max(worstZ, std::abs(mixtureFraction(m, b.YF, b.YO) - Z));
            // Shvab-Zeldovich: t - 1 + q (Y_F - Y_F1 Z) = 0.
            worstH = std::max(worstH, std::abs(b.t - 1 + m.qOverCpT0() *
                                                         (b.YF - m.yF1 * Z)));
            bounded = bounded && b.YF >= 0 && b.YO >= 0 && b.YF * b.YO == 0 &&
                      b.t >= 1 - 1e-15 && b.t <= 4 + 1e-12;
            if (Z < zs)
                linear = linear && std::abs(b.t - (1 + 3 * Z / zs)) < 1e-12;
        }
        check(worstZ < 1e-15, "the sheet state carries its own Z back");
        check(worstH < 1e-13, "the sheet state is Shvab-Zeldovich (cp T + q Y_F)");
        check(bounded, "the sheet: one reactant zero, T in [T0, sigma T0]");
        check(linear, "lean side T linear in Z up to sigma at Z_st");
        check(std::abs(m.theta() - 6 * 16 / 3.0) < 1e-12 &&
                  std::abs(m.lnA() - (std::log(300.0) + 6 * 16 / 3.0 / 4)) <
                      1e-12,
              "Ta/T0 = beta sigma^2/(sigma-1), ln A = ln Da + Ta/T_ad");
        m.sigma = 1;
        check(m.theta() == 0 && m.qOverCpT0() == 0 &&
                  m.lnA() == std::log(300.0),
              "sigma 1: no heat release, Ta 0, ln A = ln Da");
    }
    // --- erfcInv, the constant-density tip -------------------------------------
    {
        double worst = 0;
        for (double y : {1e-6, 0.01, 0.3, 0.5, 1.0, 1.5, 1.99})
            worst = std::max(worst, std::abs(std::erfc(erfcInv(y)) / y - 1));
        check(worst < 1e-14, "erfc(erfcInv(y)) = y to 1e-14");
        const double W = 16, zs = 0.25, x = sheetTipConstantDensity(W, zs);
        check(std::abs(std::erf(W / (4 * std::sqrt(x))) - zs) < 1e-13,
              "the tip: erf(W / (4 sqrt x)) = Z_st there");
    }
    // --- inflow, IC, defaults --------------------------------------------------
    {
        auto c = runCli(diff({}));
        check(c.has_value(), "diffusion defaults validate");
        if (c) {
            check(c->flameDiffusion && !c->flameBunsen, "mode flags");
            check(std::abs(c->mach - 0.05) < 1e-15 && c->diff.beta == 5 &&
                      c->bunsen.edge == 3 && c->bunsen.W == 16 &&
                      c->diff.Da == 1000 && c->diff.s == 1,
                  "defaults: M 0.05, beta 5, edge 3 l, W 16 l, Da 1000, s 1");
            check(c->grid.ny == 2 * 4 * (8 + 16) && c->grid.hx == 0.25 &&
                      c->grid.nx == 4 * 96 + 32,
                  "box: 4 cells/l, ny 2 (W/2 + coflow) 4, nx 96 l + 2 bands");
            check(std::abs(c->flameTu - 1 / (1.4 * 0.0025)) < 1e-9,
                  "T0 = 1/(gamma M^2)");
            const DiffusionModel& m = c->diff;
            const BunsenGeom& b = c->bunsen;
            double zmax = 0;
            bool sheet = true;
            for (int j = 0; j < c->grid.ny; ++j) {
                const auto in = diffusionInflow(m, b, (j + 0.5) * c->grid.hy);
                zmax = std::max(zmax, in.Z);
                const SheetState s = burkeSchumann(m, in.Z);
                sheet = sheet && s.YF == in.YF && s.YO == in.YO && s.t == in.t;
            }
            check(sheet, "the inflow row is the Burke-Schumann state of Z(y)");
            check(diffusionInflow(m, b, b.yc).Z >= zmax && zmax > 0.99,
                  "Z peaks on the slot axis (> 0.99 at edge 3, W 16)");
            CompState q = CompState::zeros(c->grid);
            std::vector<double> ry[2];
            diffusionIC(*c, q, ry);
            const std::size_t cc = c->grid.idx(17, 40, 0);
            const auto in = diffusionInflow(m, b, 40.5 * c->grid.hy);
            check(std::abs(q.rho[cc] * in.t - 1) < 1e-14 &&
                      std::abs(ry[0][cc] / q.rho[cc] - in.YF) < 1e-14 &&
                      std::abs(ry[1][cc] / q.rho[cc] - in.YO) < 1e-14 &&
                      std::abs(q.mx[cc] / q.rho[cc] - in.u) < 1e-14,
                  "IC: the inflow row extruded (rho = 1/t, the row's Y, u)");
            const std::string d = c->describe();
            check(d.find(flameRunLabel(*c)) != std::string::npos &&
                      d.find("Z_st = 0.5") != std::string::npos &&
                      flameRunLabel(*c).find("ONE limiter shared") !=
                          std::string::npos,
                  "dry-run: the label, Z_st, the shared limiter named");
        }
        auto p = runCli(diff({"--pilot-width", "2", "--pilot-speed", "0.5",
                              "--stoich", "3", "--fuel-y", "0.5",
                              "--slot-edge", "0.25"}));
        check(p.has_value(), "pilot + stoich + diluted fuel validate");
        if (p) {
            const auto in = diffusionInflow(p->diff, p->bunsen,
                                            p->bunsen.yc + 0.5 * 16 + 1.0);
            check(std::abs(in.Z - p->diff.zst()) < 0.02 && in.t > 3.9,
                  "a pilot strip is burnt gas at Z_st (T near T_ad)");
        }
    }
    // --- measureJet on a drawn field -------------------------------------------
    {
        DiffusionModel m;
        m.s = 1;
        const int nx = 80, ny = 40;
        const double h = 0.5, yc = 10;
        std::vector<float> yf(nx * ny), yo(nx * ny), w(nx * ny, 0.0f);
        // A wedge of fuel (Z = 1) narrowing to x = 30: half-width 5 (1 - x/30).
        // The axis is read as the mean of the two columns h/2 either side of
        // yc, so its Z_st crossing sits where the half-width is h/2: x 28.5.
        for (int j = 0; j < ny; ++j)
            for (int i = 0; i < nx; ++i) {
                const double x = (i + 0.5) * h, y = (j + 0.5) * h;
                const double hw = 5 * (1 - x / 30);
                const double Z = 0.5 * (1 - std::tanh((std::abs(y - yc) - hw) / 0.5));
                const SheetState s = burkeSchumann(m, Z);
                yf[j * nx + i] = float(s.YF);
                yo[j * nx + i] = float(s.YO);
                if (i >= 4) w[j * nx + i] = float(std::exp(-std::pow(Z - 0.5, 2) / 0.01));
            }
        const JetMeasure jm = measureJet(m, yf, yo, w, nx, ny, h, yc);
        std::printf("  drawn jet: tip %.4f (28.5), base %.3f (2), half %.4f "
                    "(2.5 at x 15)\n",
                    jm.xTip, jm.xBase, jm.half);
        check(std::abs(jm.xTip - 28.5) < 0.05,
              "tip of the drawn wedge within 0.1 cell");
        check(jm.xBase == 2.0, "base: the first burning row's lower face");
        check(std::abs(jm.half - 5 * (1 - 0.5 * jm.xTip / 30)) < 0.1,
              "half-width at tip/2 within 0.2 cells");
        check(jm.hits == 0 && jm.zMin >= 0 && jm.zMax <= 1,
              "bounds of the drawn field");
    }
    // --- rejections ----------------------------------------------------------------
    for (const char* f : {"--upstream", "--downstream", "--inflow-speed",
                          "--wrinkle", "--ny", "--profile-dt",
                          "--cells-per-delta", "--jet-speed", "--coflow-temp",
                          "--pilot-temp"})
        rejects(diff({f, "8"}), 2, f,
                std::string(f) + " rejected by name with --mode diffusion");
    rejects(diff({"--wrinkle-modes", "4"}), 2, "--wrinkle-modes",
            "--wrinkle-modes rejected by name with --mode diffusion");
    for (const char* f : {"--stoich", "--fuel-y", "--ox-y", "--damkohler",
                          "--cells-per-unit"}) {
        rejects({"flame", f, "2"}, 2, f,
                std::string(f) + " rejected by name with --mode premixed");
        rejects({"flame", "--mode", "bunsen", f, "2"}, 2, f,
                std::string(f) + " rejected by name with --mode bunsen");
    }
    rejects(diff({"--re", "100"}), 2, "--re", "--re rejected");
    rejects(diff({"--ft", "3"}), 2, "--ft", "--ft rejected");
    rejects(diff({"--n", "64"}), 2, "--n", "--n rejected");
    rejects(diff({"--nz", "4"}), 2, "--nz", "--nz 4 rejected");
    rejects(diff({"--physics", "incompressible"}), 2, "--physics",
            "--physics incompressible rejected");
    rejects(diff({"--stoich", "0"}), 2, "--stoich", "--stoich 0 rejected");
    rejects(diff({"--fuel-y", "1.5"}), 2, "--fuel-y", "--fuel-y 1.5 rejected");
    rejects(diff({"--ox-y", "0"}), 2, "--ox-y", "--ox-y 0 rejected");
    rejects(diff({"--damkohler", "-1"}), 2, "--damkohler", "--damkohler -1 rejected");
    rejects(diff({"--expansion", "1"}), 2, "--expansion", "--expansion 1 rejected");
    rejects(diff({"--zeldovich", "1"}), 2, "--zeldovich", "--zeldovich 1 rejected");
    rejects(diff({"--mach", "0.25"}), 2, "--mach", "--mach 0.25 rejected");
    rejects(diff({"--mach", "0.2", "--coflow-speed", "2"}), 2, "--mach",
            "the fastest stream's Mach above 0.3 rejected");
    rejects(diff({"--cells-per-unit", "1"}), 2, "--cells-per-unit",
            "--cells-per-unit 1 rejected");
    rejects(diff({"--slot-width", "3"}), 2, "--slot-width", "--slot-width 3 rejected");
    rejects(diff({"--coflow-speed", "3"}), 2, "--coflow-speed",
            "--coflow-speed 3 rejected");
    rejects(diff({"--pilot-speed", "1"}), 2, "--pilot-speed",
            "--pilot-speed without --pilot-width rejected");
    rejects(diff({"--slot-edge", "0.1"}), 2, "--slot-edge",
            "--slot-edge below one cell rejected");
    rejects(diff({"--length", "8"}), 2, "--length", "--length < W rejected");
    rejects(diff({"--pulse-freq", "0.1"}), 2, "--pulse-freq",
            "--pulse-freq without --pulse rejected");
    rejects(diff({"--tinf", "300"}), 2, "--tinf", "--tinf without sutherland rejected");
    rejects({"flame", "--mode", "nonpremixed"}, 1, "--mode",
            "an unknown --mode value is a CLI error");
    return dnstest::finish();
}
