// dns flame --mode chamber, host side (docs/chem.md, "Chamber"): the polygon
// SDF against hand distances, the chamber's regions (closed cavity, exit
// column, probes, injector), the IC, the isentropic nozzle inverse, the
// resolved case and dry-run, and every rejection BY NAME (code 2), in both
// directions: chamber-only flags without --mode chamber, premixed-only ones
// with it.
#include "dns/chamber.hpp"
#include "dns/config.hpp"
#include "check.hpp"
#include "cli_test_util.hpp"

#include <algorithm>
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

std::vector<std::string> ch(std::vector<std::string> more)
{
    std::vector<std::string> a{"flame", "--mode", "chamber"};
    a.insert(a.end(), more.begin(), more.end());
    return a;
}

} // namespace

int main()
{
    using namespace dns;

    // --- polygon SDF: unit square, both orientations ----------------------
    {
        const std::vector<Point2> sq{{0, 0}, {1, 0}, {1, 1}, {0, 1}};
        std::vector<Point2> cw(sq.rbegin(), sq.rend());
        check(std::abs(polygonSdf(sq, 0.5, 0.5) + 0.5) < 1e-15 &&
                  std::abs(polygonSdf(cw, 0.5, 0.5) + 0.5) < 1e-15,
              "square centre: -0.5 (either orientation)");
        check(std::abs(polygonSdf(sq, 2, 0.5) - 1) < 1e-15,
              "outside, beside an edge: +1");
        check(std::abs(polygonSdf(sq, 2, 2) - std::sqrt(2.0)) < 1e-15,
              "outside, beyond a corner: sqrt(2)");
        check(std::abs(polygonSdf(sq, 0.9, 0.2) + 0.1) < 1e-15,
              "inside near an edge: -0.1");
    }

    // --- the default case --------------------------------------------------
    auto c = runCli(ch({}));
    check(c.has_value(), "--mode chamber validates at the defaults");
    if (!c) return dnstest::finish();
    const double dth = c->flameEig.dthOverDD;
    const ChamberSpec& s = c->chamber;
    const double h = c->grid.hx;
    check(c->chamberMode && std::abs(h - dth / 4) < 1e-15,
          "h = delta_th / 4 (chamber default)");
    check(std::abs(c->mach - 0.02) < 1e-15 && std::abs(c->tEnd - 60) < 1e-15,
          "chamber defaults: M 0.02, tt 60");
    check(std::abs(s.Hc - 15 * dth) < 1e-12 && std::abs(s.Lc - 30 * dth) < 1e-12 &&
              std::abs(s.He - 5 * dth) < 1e-12 && std::abs(s.tw - 1.5 * dth) < 1e-12,
          "Hc 15, Lc 30, He 5, wall 1.5 delta_th");
    check(std::abs(c->chamberMdot - 3 * s.Hc) < 1e-12,
          "mdot = rho_u 3 S_L Hc");
    check(std::abs(s.x0 - (16 * h + 4 * dth)) < 1e-12,
          "back wall 4 delta_th past the 16-cell sink band");
    check(c->grid.nx * h - s.xe() >= 40 * dth + 16 * h - 1e-9,
          "40 delta_th of jet + the band past the exit");
    check(c->grid.nz == 1 && std::abs(c->grid.hy - h) < 1e-15,
          "2D, square cells");
    check(s.tw / h >= 4 && s.He / h >= 8, "wall >= 4 cells, exit >= 8 cells");
    const std::string d = c->describe();
    std::printf("%s", d.c_str());
    check(d.find(chamberRunLabel(*c)) != std::string::npos &&
              d.find("(inflated; CH4/air 1.15e-3)") != std::string::npos &&
              d.find("NOT body-fitted") != std::string::npos,
          "dry-run carries the label (inflated M, immersed wall)");
    check(c->warnings.empty(), "no warning at the defaults");

    // --- regions -----------------------------------------------------------
    const Grid& g = c->grid;
    const auto reg = chamberRegions(g, s);
    auto R = [&](int i, int j) { return ChamberRegion(reg[g.idx(i, j, 0)]); };
    int counts[kChamberRegions] = {};
    for (auto r : reg) ++counts[r];
    for (int k = 0; k < kChamberRegions; ++k) std::printf("    region %d: %d cells\n", k, counts[k]);
    for (int k = 0; k < kChamberRegions; ++k)
        check(counts[k] > 0, "region " + std::to_string(k) + " is populated");
    // The cavity is closed except through the exit: no cavity cell is
    // face-adjacent to an ambient cell, except across the exit column's +x
    // face.
    bool closed = true;
    const int ie = int(std::ceil(s.xe() / h - 0.5)) - 1;
    for (int j = 0; j < g.ny; ++j)
        for (int i = 0; i < g.nx; ++i) {
            if (!inCavity(std::uint8_t(R(i, j)))) continue;
            const int ni[4] = {i + 1, i - 1, i, i};
            const int nj[4] = {j, j, j + 1, j - 1};
            for (int f = 0; f < 4; ++f) {
                const auto q = R(Grid::wrap(ni[f], g.nx), Grid::wrap(nj[f], g.ny));
                const bool amb = q == ChamberRegion::Ambient ||
                                 q == ChamberRegion::AmbientProbe;
                if (amb && !(f == 0 && i == ie)) closed = false;
            }
        }
    check(closed, "the cavity touches the ambient only through the exit plane");
    // Exit column: its cavity cells span He (to a cell).
    check(std::abs(counts[int(ChamberRegion::Exit)] * h - s.He) <= 2 * h,
          "exit column spans He to within a cell per side");
    // Solid = sdf < 0 at the centres, exactly.
    bool solidOk = true;
    for (int j = 0; j < g.ny; ++j)
        for (int i = 0; i < g.nx; ++i) {
            const bool sol = chamberSdf(s, (i + 0.5) * h, (j + 0.5) * h) < 0;
            if (sol != (R(i, j) == ChamberRegion::Solid)) solidOk = false;
        }
    check(solidOk, "Solid region == sdf < 0 at the cell centres");
    // No solid cell in the sink bands or within 4 delta_th of x-lo.
    bool bandsFree = true;
    for (int j = 0; j < g.ny; ++j)
        for (int i = 0; i < g.nx; ++i)
            if (R(i, j) == ChamberRegion::Solid &&
                (i < 16 + int(4 * dth / h) - 1 || i >= g.nx - 16))
                bandsFree = false;
    check(bandsFree, "no solid cell inside the sink bands");

    // --- IC ------------------------------------------------------------------
    {
        CompState q;
        std::vector<double> ry;
        chamberIC(g, reg, 1.4, c->flameTu, 5.0, 2.0,
                  [&](double x) { return flameProfile(c->flameEig.tab, x); },
                  s.xBase(), q, ry);
        const int jc = g.ny / 2;
        const int iInj = int((s.xb() + 0.5 * s.wInj) / h);
        const int iBurnt = int((s.xn() - 1) / h);
        const std::size_t ci = g.idx(iInj, jc, 0), cb = g.idx(iBurnt, jc, 0);
        // 5.6 delta_D ahead of the oracle's mid temperature: the preheat tail
        // is down to ~1e-3 there.
        check(std::abs(ry[ci] / q.rho[ci] - 1) < 2e-3 &&
                  std::abs(q.rho[ci] - 1) < 2e-3 && q.mx[ci] == 0,
              "injector band fresh: Y = 1, rho_u = 1, at rest (preheat tail "
              "< 2e-3)");
        check(ry[cb] / q.rho[cb] < 1e-6 && std::abs(q.rho[cb] - 0.2) < 1e-6,
              "nozzle entry burnt: Y ~ 0, rho = 1/sigma");
        bool solid0 = true, amb = true;
        for (std::size_t k = 0; k < g.size(); ++k) {
            if (reg[k] == std::uint8_t(ChamberRegion::Solid) && ry[k] != 0)
                solid0 = false;
            if (reg[k] == std::uint8_t(ChamberRegion::Ambient) &&
                (ry[k] != 0 || std::abs(q.mx[k] - 2.0) > 1e-12))
                amb = false;
        }
        check(solid0, "solid cells carry rhoY = 0");
        check(amb, "ambient inert (Y = 0) at the coflow");
        const double p = 0.4 * (q.E[cb] - 0.5 * q.mx[cb] * q.mx[cb] / q.rho[cb]);
        check(std::abs(p / c->flameTu - 1) < 1e-12, "IC at p0 everywhere");
    }

    // --- isentropic nozzle ---------------------------------------------------
    {
        const double Tu = 1785.714285714286, T0 = 5 * Tu, A = 8, pa = Tu;
        auto e = isentropicNozzle(72, T0, A, pa, 1.4, 1.0);
        check(e.has_value(), "isentropic estimate exists below sonic");
        if (e) {
            const double Te = T0 / (1 + 0.2 * e->Me * e->Me);
            const double mdot = pa / Te * e->Me * std::sqrt(1.4 * Te) * A;
            check(std::abs(mdot / 72 - 1) < 1e-12, "inverse reproduces mdot");
            check(std::abs(e->p0 / (pa * std::pow(1 + 0.2 * e->Me * e->Me, 3.5)) - 1) <
                      1e-14,
                  "p0 = pa (1 + 0.2 Me^2)^3.5");
            std::printf("    default: Me %.6f, p0/pa %.6f\n", e->Me, e->p0 / pa);
        }
        check(!isentropicNozzle(1e6, T0, A, pa, 1.4, 1.0).has_value(),
              "a super-sonic flux is reported as choked");
    }

    // --- rejections ------------------------------------------------------------
    rejects(ch({"--ny", "8"}), 2, "--ny", "chamber rejects --ny");
    rejects(ch({"--upstream", "8"}), 2, "--upstream", "chamber rejects --upstream");
    rejects(ch({"--inflow-speed", "2"}), 2, "--inflow-speed",
            "chamber rejects --inflow-speed");
    rejects(ch({"--profile-dt", "1"}), 2, "--profile-dt",
            "chamber rejects --profile-dt");
    for (const char* f : {"--inject-speed", "--coflow", "--chamber-height",
                          "--chamber-length", "--nozzle-ratio", "--holder-height",
                          "--clearance", "--inject-ramp"})
        rejects({"flame", f, "3"}, 2, f, std::string("premixed rejects ") + f);
    rejects({"flame", "--reaction", "off"}, 2, "--reaction",
            "premixed rejects --reaction");
    rejects(ch({"--inject-speed", "0"}), 2, "--inject-speed", "inject speed > 0");
    rejects(ch({"--coflow", "0"}), 2, "--coflow", "coflow > 0");
    rejects(ch({"--chamber-height", "7"}), 2, "--chamber-height", "height >= 8");
    rejects(ch({"--chamber-length", "9"}), 2, "--chamber-length", "length >= 10");
    rejects(ch({"--nozzle-ratio", "1.2"}), 2, "--nozzle-ratio", "ratio >= 1.5");
    rejects(ch({"--nozzle-ratio", "7"}), 2, "--nozzle-ratio", "ratio <= 6");
    rejects(ch({"--holder-height", "12"}), 2, "--holder-height",
            "holder <= height - 4");
    rejects(ch({"--clearance", "3"}), 2, "--clearance", "clearance >= 4");
    rejects(ch({"--inject-ramp", "-1"}), 2, "--inject-ramp", "ramp >= 0");
    rejects(ch({"--downstream", "5"}), 2, "--downstream", "jet region >= 10");
    rejects(ch({"--inject-speed", "15"}), 2, "--inject-speed",
            "a choking injection is rejected");
    rejects(ch({"--nz", "4"}), 2, "--nz", "chamber is 2D");
    rejects(ch({"--physics", "incompressible"}), 2, "--physics",
            "chamber is compressible");
    rejects(ch({"--chamber-height", "8", "--nozzle-ratio", "6"}), 2,
            "--nozzle-ratio", "an exit under 8 cells is rejected");
    {
        auto off = runCli(ch({"--reaction", "off"}));
        check(off.has_value() && !off->chamberReaction &&
                  chamberRunLabel(*off).find("inert") != std::string::npos,
              "--reaction off validates and says inert in the label");
    }
    return dnstest::finish();
}
