// dns flame --mode candle, host side (docs/chem.md, candle section): the
// ambient column, the candle's dimensionless groups, the hydrostatic IC
// (horizontally uniform p, exp(-g x/(R T0)) up the column), the quiescent IC,
// --ignition's closed sheet, the g = 0 IC == the diffusion IC bit for bit,
// the defaults, the dry-run and label, and every rejection BY NAME.
#include "dns/config.hpp"
#include "dns/diffusion.hpp"
#include "dns/flame1d.hpp"
#include "check.hpp"
#include "cli_test_util.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
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
    if (c) std::printf("    accepted\n");
    check(ok, what);
}

std::vector<std::string> candle(std::vector<std::string> extra)
{
    std::vector<std::string> a{"flame", "--mode", "candle"};
    a.insert(a.end(), extra.begin(), extra.end());
    return a;
}

double pressure(const dns::CompState& q, std::size_t c)
{
    const double u = q.mx[c] / q.rho[c];
    return 0.4 * (q.E[c] - 0.5 * q.rho[c] * u * u);
}

bool sameBits(const std::vector<double>& a, const std::vector<double>& b)
{
    return a.size() == b.size() &&
           std::memcmp(a.data(), b.data(), a.size() * sizeof(double)) == 0;
}

} // namespace

int main()
{
    using namespace dns;
    // --- the ambient column and the groups -------------------------------------
    {
        check(candleAmbientP(5.0, 0.0, 5.0, 123.0) == 5.0,
              "g = 0: p_a = p0 exactly");
        const double r = candleAmbientP(2e4, 14, 2e4, 1.125) /
                         candleAmbientP(2e4, 14, 2e4, 1.0);
        check(std::abs(r - std::exp(-14 * 0.125 / 2e4)) < 1e-15,
              "p_a(x + h)/p_a(x) = exp(-g h/(R T0))");
        DiffusionModel m; // sigma 5
        const CandleScales s = candleScales(m, 14, 1, 36, 0.005);
        std::printf("  defaults: Fr %.5g, u_ff %.4g, Mb %.4g, adiabatic %.4g, "
                    "drop %.4g, flicker %.4g\n",
                    s.froude, s.uFreeFall, s.machBuoyant, s.adiabatic,
                    s.hydroDrop, s.flicker);
        check(std::abs(s.froude - 1.0 / 14) < 1e-15, "Fr = 1/(g W)");
        check(std::abs(s.uFreeFall - std::sqrt(2 * 14 * 36 * 0.8)) < 1e-12,
              "free-fall speed sqrt(2 g L (sigma-1)/sigma)");
        check(std::abs(s.adiabatic - 14 * 36 * 0.4 * 0.005 * 0.005 / 4) < 1e-15,
              "adiabatic fraction g L (gamma-1) M^2/(sigma-1)");
        check(std::abs(s.hydroDrop -
                       (1 - std::exp(-1.4 * 0.005 * 0.005 * 14 * 36))) < 1e-15,
              "hydrostatic drop 1 - exp(-gamma M^2 g L)");
        check(std::abs(s.flicker - 1.5 / std::sqrt(9.81) * std::sqrt(14.0)) <
                  1e-12,
              "flicker scale 0.479 sqrt(g/W)");
        const DiffusionInflowRow q = quiescentInflow(m);
        check(q.u == 0 && q.Z == 0 && q.YF == 0 && q.YO == m.yO2 && q.t == 1,
              "quiescent row: oxidizer at rest at T0");
    }
    // --- defaults, dry-run, label ------------------------------------------------
    {
        auto c = runCli(candle({}));
        check(c.has_value(), "the candle defaults validate");
        if (c) {
            const Grid& g = c->grid;
            std::printf("  defaults: %d x %d, h %.4g, g %g, W %g, Z_st %.5f, "
                        "Da %g, M %g, ignition %g\n",
                        g.nx, g.ny, g.hx, c->flameGravity, c->bunsen.W,
                        c->diff.zst(), c->diff.Da, c->mach, c->flameIgnition);
            check(c->flameCandle && c->flameDiffusion && !c->flameQuiescent,
                  "candle = the diffusion model + gravity");
            check(c->flameGravity == 14 && c->bunsen.W == 1 &&
                      c->flameCells == 8 && c->diff.Da == 5000 &&
                      c->mach == 0.005 && c->flameIgnition == 4 &&
                      c->bunsen.edge == 0.5 && c->bunsen.uCo == 1 &&
                      c->bunsen.wCo == 12,
                  "candle defaults (g 14, W 1, 8 cells/l, Da 5000, M 0.005, "
                  "ignition 4, edge 0.5, coflow 1 x 12)");
            check(std::abs(c->diff.zst() - 0.233 / (3.4 + 0.233)) < 1e-15,
                  "paraffin/air: s 3.4, Y_O2 0.233");
            check(g.nx == 32 * 8 + 32 && g.ny == 200, "box 288 x 200");
            const std::string d = c->describe();
            check(d.find("Fr = U^2/(g W) = 0.07143") != std::string::npos &&
                      d.find("adiabatic fraction") != std::string::npos &&
                      d.find("p_a(x_out)") != std::string::npos &&
                      d.find(flameRunLabel(*c)) != std::string::npos,
                  "dry-run: Fr, adiabatic fraction, the hydrostatic top, the "
                  "label");
            check(flameRunLabel(*c).find("candle: gravity g=14") !=
                      std::string::npos,
                  "label names the gravity");
            check(c->warnings.empty(), "the defaults raise no warning");

            // The hydrostatic IC: p = p_a(x) on every cell (horizontally
            // uniform), whatever the column's T.
            CompState q = CompState::zeros(g);
            std::vector<double> ry[2];
            diffusionIC(*c, q, ry);
            const double T0 = c->flameTu;
            double worst = 0, tMax = 0;
            for (int i = 0; i < g.nx; i += 7)
                for (int j = 0; j < g.ny; ++j) {
                    const std::size_t cc = g.idx(i, j, 0);
                    const double pa =
                        candleAmbientP(T0, 14, T0, (i + 0.5) * g.hx);
                    worst = std::max(worst, std::abs(pressure(q, cc) / pa - 1));
                    tMax = std::max(tMax, pressure(q, cc) / q.rho[cc] / T0);
                }
            std::printf("  IC: max |p/p_a - 1| %.3g, max T/T0 %.4f\n", worst,
                        tMax);
            check(worst < 1e-13, "IC pressure = the ambient column p_a(x)");
            check(tMax > 4.5, "IC: the ignition column burns (T near T_ad)");
            // Past the ignition column: pure coflow air at T0 and u_co.
            const std::size_t far = g.idx(g.nx - 20, 3, 0);
            check(ry[0][far] == 0 &&
                      std::abs(ry[1][far] / q.rho[far] - 0.233) < 1e-15 &&
                      std::abs(pressure(q, far) / q.rho[far] / T0 - 1) <
                          1e-13 &&
                      std::abs(q.mx[far] / q.rho[far] - 1) < 1e-15,
                  "IC past the ignition column: coflow air at T0");
        }
    }
    // --- quiescent ---------------------------------------------------------------
    {
        auto c = runCli(candle({"--quiescent"}));
        check(c.has_value() && c->flameQuiescent, "--quiescent validates");
        if (c) {
            CompState q = CompState::zeros(c->grid);
            std::vector<double> ry[2];
            diffusionIC(*c, q, ry);
            bool rest = true;
            for (std::size_t k = 0; k < q.rho.size(); ++k)
                rest = rest && q.mx[k] == 0 && ry[0][k] == 0;
            check(rest, "quiescent IC: at rest, no fuel");
            check(c->describe().find("NO fuel") != std::string::npos &&
                      flameRunLabel(*c).find("QUIESCENT") != std::string::npos,
                  "quiescent named in the dry-run and the label");
        }
    }
    // --- g = 0: the diffusion IC bit for bit --------------------------------------
    {
        const std::vector<std::string> shared{
            "--slot-width", "4",     "--cells-per-unit", "4",
            "--slot-edge",  "1",     "--length",         "16",
            "--coflow-width", "4",   "--ignition",       "4",
            "--stoich",     "3.4",   "--ox-y",           "0.233",
            "--damkohler",  "5000",  "--mach",           "0.05"};
        std::vector<std::string> a = candle({"--gravity", "0"});
        a.insert(a.end(), shared.begin(), shared.end());
        std::vector<std::string> b{"flame", "--mode", "diffusion"};
        b.insert(b.end(), shared.begin(), shared.end());
        auto ca = runCli(a), cb = runCli(b);
        check(ca.has_value() && cb.has_value(),
              "candle --gravity 0 and diffusion validate");
        if (ca && cb) {
            check(ca->grid.nx == cb->grid.nx && ca->grid.ny == cb->grid.ny,
                  "g = 0: the same box");
            CompState qa = CompState::zeros(ca->grid), qb = qa;
            std::vector<double> ra[2], rb[2];
            diffusionIC(*ca, qa, ra);
            diffusionIC(*cb, qb, rb);
            check(sameBits(qa.rho, qb.rho) && sameBits(qa.mx, qb.mx) &&
                      sameBits(qa.E, qb.E) && sameBits(ra[0], rb[0]) &&
                      sameBits(ra[1], rb[1]),
                  "g = 0: the candle IC is the diffusion IC bit for bit");
            check(ca->describe().find("g = 0: the diffusion jet bit for "
                                      "bit") != std::string::npos,
                  "g = 0 named in the dry-run");
        }
        // Without --ignition the diffusion IC is the extruded row (default).
        auto cd = runCli({"flame", "--mode", "diffusion"});
        check(cd.has_value() && cd->flameIgnition == 0,
              "diffusion default: no ignition column (the extruded row)");
    }
    // --- warnings ----------------------------------------------------------------
    {
        auto c = runCli(candle({"--gravity", "1"}));
        check(c.has_value() &&
                  std::any_of(c->warnings.begin(), c->warnings.end(),
                              [](const std::string& w) {
                                  return w.find("Froude") != std::string::npos;
                              }),
              "Fr = 1 warns (jet momentum not negligible)");
        auto m = runCli(candle({"--mach", "0.01"}));
        check(m.has_value() &&
                  std::any_of(m->warnings.begin(), m->warnings.end(),
                              [](const std::string& w) {
                                  return w.find("buoyant Mach") !=
                                         std::string::npos;
                              }),
              "buoyant Mach 0.28 warns");
    }
    // --- rejections ----------------------------------------------------------------
    for (const char* f : {"--gravity", "--ignition"}) {
        rejects({"flame", f, "2"}, 2, f,
                std::string(f) + " rejected by name with --mode premixed");
        rejects({"flame", "--mode", "bunsen", f, "2"}, 2, f,
                std::string(f) + " rejected by name with --mode bunsen");
    }
    rejects({"flame", "--mode", "diffusion", "--gravity", "2"}, 2, "--gravity",
            "--gravity rejected by name with --mode diffusion");
    rejects({"flame", "--quiescent"}, 2, "--quiescent",
            "--quiescent rejected by name with --mode premixed");
    rejects({"flame", "--mode", "diffusion", "--quiescent"}, 2, "--quiescent",
            "--quiescent rejected by name with --mode diffusion");
    for (const char* f : {"--pilot-width", "--pilot-speed"})
        rejects(candle({f, "1"}), 2, f,
                std::string(f) + " rejected by name with --mode candle");
    for (const char* f : {"--slot-width", "--slot-edge", "--coflow-speed",
                          "--pulse", "--ignition"})
        rejects(candle({"--quiescent", f, "1"}), 2, f,
                std::string(f) + " rejected by name with --quiescent");
    rejects(candle({"--quiescent", "--pulse-freq", "1"}), 2, "--pulse-freq",
            "--pulse-freq rejected by name with --quiescent");
    rejects(candle({"--gravity", "-1"}), 2, "--gravity", "--gravity -1 rejected");
    rejects(candle({"--gravity", "nan"}), 1, "--gravity",
            "--gravity nan rejected (a CLI error: not a number)");
    rejects(candle({"--gravity", "2000"}), 2, "--gravity",
            "--gravity 2000 rejected");
    rejects(candle({"--ignition", "0.5"}), 2, "--ignition",
            "--ignition below 1 l rejected");
    rejects(candle({"--ignition", "40"}), 2, "--ignition",
            "--ignition past --length rejected");
    rejects(candle({"--slot-width", "0.4"}), 2, "--slot-width",
            "--slot-width 0.4 rejected");
    rejects(candle({"--mach", "0.05"}), 2, "--mach",
            "buoyant Mach above 0.3 rejected");
    return dnstest::finish();
}
