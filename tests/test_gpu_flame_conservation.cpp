// Conservation of the reacting stage (docs/chem.md), device API level: a
// periodic x box holding two back-to-back oracle flames (burnt gas between
// them), M 0.02. Mass and total energy E + q rho Y have no source; the fuel
// lost equals the FP64 tally of the rate the stage used (trapezoid over steps
// of sum omega dV dt -- the RK3 stage weights' second-order quadrature, ~1e-8
// relative here); no cell leaves [0, 1].
//
// Measured 2026-10-28 (gfx1100): over t = 2 (5696 steps) mass drifts 5.1e-5
// and E + q rho Y 5.3e-5 relative -- the FP32 STATE rounding of the stage
// write, which the NON-REACTING twin (same IC, A = 0: rho never sees the
// species code) shows at the same size. The fuel inventory carries the same
// rounding (~1e-3 absolute of ~16), i.e. 4.6e-4 of the 2.17 burnt. So the
// 1e-6 bands are checked over the first 50 steps, where that rounding is
// ~1e-7, and the t = 2 drifts are pinned against the twin's own.
#include "dns/chem.hpp"
#include "dns/comp.hpp"
#include "dns/flame1d.hpp"
#include "dns/gpu/chem.hpp"
#include "dns/gpu/comp.hpp"
#include "dns/transport.hpp"
#include "gpu_test_util.hpp"
#include "check.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <vector>

using namespace dns;

namespace {

using dnstest::check;

struct Tot
{
    double mass = 0, energy = 0, fuel = 0, hits = 0;
};

struct Drift
{
    double dm50 = 0, de50 = 0, df50 = 0; // after 50 steps
    double dm = 0, de = 0, lost = 0, burnt = 0, fuel0 = 0, hits = 0;
    long steps = 0;
};

Drift run(bool reacting, double tEnd)
{
    const PremixedModel md{};
    auto eig = premixedEigen(md);
    if (!eig) std::exit(EXIT_FAILURE);
    const double dth = eig->dthOverDD, M = 0.02, gam = 1.4;
    const double Tu = 1 / (gam * M * M), sig = md.sigma;
    const int cells = 15, nx = 40 * cells, ny = 4;
    const double h = dth / cells;
    const Grid g = Grid::make(nx, ny, 1, nx * h, ny * h, 1.0);
    const double xa = 15 * dth, xb = 25 * dth;
    CompState ic;
    std::vector<double> ry;
    premixedState(g, gam, Tu, sig, 0.0,
                  [&](double x, double) {
                      return std::min(flameProfile(eig->tab, x - xa),
                                      flameProfile(eig->tab, xb - x));
                  },
                  ic, ry);

    const auto tr = Transport::air(md.Pr, Tu, 288.15, md.Pr, gam, 1.0, false);
    auto dev = gpu::DeviceComp::make(g, tr);
    auto red = gpu::ReduceScratch::make();
    if (!dev || !red) std::exit(EXIT_FAILURE);
    gpu::SpeciesSetup sp;
    sp.lnA = reacting ? std::log(eig->Lambda)
                      : -std::numeric_limits<double>::infinity();
    sp.Ta = md.theta() * Tu;
    sp.q = tr.cp() * (sig - 1) * Tu;
    sp.invPr = 1 / md.Pr;
    if (!dev->enableSpecies(sp)) std::exit(EXIT_FAILURE);
    auto ds = gpu::DeviceCompState::make(g.size(), 0, false, 1);
    if (!ds || ds->upload(ic) != hipSuccess ||
        ds->uploadSpecies(0, ry) != hipSuccess)
        std::exit(EXIT_FAILURE);

    gpu::DeviceBuffer<double> scr;
    std::vector<double> ps;
    const double dV = g.cellVolume();
    auto totals = [&] {
        dev->updateChem(*ds);
        Tot t;
        if (gpu::flameColumnStats(*dev, *ds, scr, ps) != hipSuccess)
            std::exit(EXIT_FAILURE);
        for (int i = 0; i < nx; ++i) {
            const double* c = &ps[std::size_t(i) * gpu::kFlameSums];
            t.mass += c[0] * dV;
            t.energy += c[5] * dV;
            t.fuel += c[3] * dV;
            t.hits += c[8];
        }
        return t;
    };
    const Tot t0 = totals();
    Drift d;
    d.fuel0 = t0.fuel;
    double W = gpu::sumScalar(dev->omega(), *red) * dV, t = 0, dt = 0;
    while (t < tEnd) {
        if (d.steps % 16 == 0)
            dt = 0.95 * std::min(dev->stableDt(*ds, *red, 0.4),
                                 gpu::chemStableDt(*dev, *ds, *red));
        const double s = std::min(dt, tEnd - t);
        dev->step(*ds, gpu::Real(s));
        dev->updateChem(*ds);
        const double W1 = gpu::sumScalar(dev->omega(), *red) * dV;
        d.burnt += 0.5 * (W + W1) * s;
        W = W1;
        t += s;
        ++d.steps;
        if (d.steps == 50) {
            const Tot a = totals();
            d.dm50 = std::abs(a.mass / t0.mass - 1);
            d.de50 = std::abs(a.energy / t0.energy - 1);
            d.df50 = std::abs((t0.fuel - a.fuel) - d.burnt) / t0.fuel;
            d.hits += a.hits;
        }
        if (d.steps % 500 == 0) d.hits += totals().hits;
    }
    const Tot t1 = totals();
    d.hits += t1.hits;
    d.dm = t1.mass / t0.mass - 1;
    d.de = t1.energy / t0.energy - 1;
    d.lost = t0.fuel - t1.fuel;
    return d;
}

} // namespace

int main()
{
    test::requireDeviceOrSkip();
    const Drift r = run(true, 2.0), z = run(false, 2.0);
    std::printf("  reacting: %ld steps; after 50: mass %.3e, E+q rhoY %.3e, "
                "(lost - tally)/inventory %.3e\n",
                r.steps, r.dm50, r.de50, r.df50);
    std::printf("  reacting t=2: mass %+.3e, E+q rhoY %+.3e; fuel lost %.8g, "
                "tally %.8g (rel %.3e of burnt, %.3e of inventory %.6g); "
                "hits %g\n",
                r.dm, r.de, r.lost, r.burnt, r.lost / r.burnt - 1,
                (r.lost - r.burnt) / r.fuel0, r.fuel0, r.hits);
    std::printf("  non-reacting twin t=2: mass %+.3e, E+q rhoY %+.3e, fuel "
                "lost %.3e (no reaction: tally %.3g)\n",
                z.dm, z.de, z.lost, z.burnt);
    check(r.burnt > 1.0, "the flames burn (tally > 1 fuel unit)");
    check(z.burnt == 0.0, "the twin burns nothing (A = 0 is omega == 0)");
    check(r.dm50 < 1e-6, "50 steps: |dmass|/mass < 1e-6");
    check(r.de50 < 1e-6, "50 steps: |d(E + q rhoY)|/(E + q rhoY) < 1e-6");
    check(r.df50 < 1e-6,
          "50 steps: |fuel lost - FP64 rate tally| < 1e-6 of the inventory");
    // t = 2: the drifts are the FP32 state rounding, the twin's size.
    const double floor = std::max({std::abs(z.dm), std::abs(z.de),
                                   std::abs(z.lost) / z.fuel0});
    check(std::abs(r.dm) < 3 * floor && std::abs(r.de) < 3 * floor &&
              std::abs(r.lost - r.burnt) / r.fuel0 < 3 * floor,
          "t = 2: mass, E + q rhoY and fuel balance drift within 3x the "
          "non-reacting twin's FP32 rounding drift");
    check(r.hits == 0 && z.hits == 0, "no cell with Y outside [-1e-6, 1 + 1e-6]");
    return dnstest::finish();
}
