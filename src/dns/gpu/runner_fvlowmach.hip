#include "dns/gpu/runner_fvlowmach.hpp"

#include "dns/gpu/fv_incomp.hpp"
#include "dns/gpu/hip_runtime.hpp"
#include "dns/gpu/internal/frame_scratch.hpp"
#include "dns/gpu/internal/run_loop.hpp"
#include "dns/interrupt.hpp"
#include "dns/la/graph.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <format>
#include <string>
#include <vector>

namespace dns::gpu {

using internal::fail;
using internal::finishTiming;

std::expected<RunStats, RunError> runFvLowMach(const FaceMesh& fm, const FvLowMachIC& ic, FvLowMachRun r,
                                               const FvRaster& raster, RunOutput& out)
{
    if (auto b = bindFvIncompBcs(r.bcs.flow, fm); !b) return fail(3, b.error());
    auto setup = buildFvIncompSetup(fm, r.bcs.flow);
    if (!setup) return fail(3, setup.error());
    const FvIncompSetup& s = *setup;
    const FvLowMachTables tab = buildFvLowMachTables(fm, r.bcs, s, r.model.prGas);
    const auto graph = la::graphOf(fm);

    const std::string cols[] = {"t",      "dt",      "dtChem",   "dtDiff", "SL",     "SLdisp", "xf",    "dth",
                                "Tmax",   "mass",    "massTally", "omegaEdge", "yHits", "uFresh", "pFresh", "TFresh",
                                "yMin",   "yMax",    "pJump",    "szDefect", "divErr", "eosErr", "pcgIt",
                                "burn",   "fuel",    "eDefect"};
    if (auto e = out.declareSeries("stats", cols, r.label); !e) return fail(4, e.error());
    const std::string pcols[] = {"t", "x", "T", "rho", "u", "p", "Y", "omega"};
    if (auto e = out.declareSeries("profiles", pcols); !e) return fail(4, e.error());

    auto made = DeviceFvLowMach::make(s, tab, graph, r.model, r.ctl, ic);
    if (!made) return fail(6, made.error());
    DeviceFvLowMach& dev = *made;
    if (!r.quiet) {
        std::printf("# %s\n# FV path: %d cells, %d internal faces, %d boundary faces, max non-orthogonality %.2f deg\n",
                    r.label.c_str(), s.nC, s.nI, s.nB, s.nonOrthMaxDeg);
        std::fflush(stdout);
    }

    const FvFlameProbe& probe = *r.flame;
    const FvFlameSpec& fl = probe.spec();
    const double Hin = fl.Tu + r.model.qcp;
    const Grid& rg = raster.grid();
    std::vector<float> chan0(rg.size()), chan1(rg.size()), chan2(rg.size());

    RunStats st;
    double t = 0, dtStab = 0, dtFirst = 0, xfPrev = NAN, tPrev = 0, lastHits = 0, lastMin = 0, lastMax = 1;
    FvLowMachDt bound;
    int sinceDt = 1 << 30;
    long step = 0, unconverged = 0, solves = 0;
    int lastIters = 0;
    bool bindChem = false;
    internal::StatsCadence cad{r.statsEvery};
    internal::ProfileSchedule prof(r.profileDt);
    const auto wall0 = std::chrono::steady_clock::now();

    auto measured = [&](FvLowMachCells& c, std::vector<double>& ry) {
        ry = c.ry[0];
        return probe.measureLowMach({.rho = &c.rho, .ry = &ry, .ru = &c.ru, .T = &c.T, .p = &c.p, .omega = &c.omega});
    };
    auto writeFrame = [&]() -> std::expected<void, RunError> {
        auto c = dev.cells();
        if (!c) return fail(6, c.error());
        std::vector<double> dT(c->T), Y(c->T.size());
        std::ranges::for_each(dT, [&](double& v) { v -= fl.Tu; });
        for (std::size_t q = 0; q < Y.size(); ++q) Y[q] = c->ry[0][q] / c->rho[q];
        raster.sample(dT, chan0);
        raster.sample(Y, chan1);
        raster.sample(c->omega, chan2);
        const FrameField ff[3] = {{"temp", chan0.data(), 0, float((fl.sigma - 1.0) * fl.Tu), false},
                                  {"fuel", chan1.data(), 0, 1.0f, false},
                                  {"omega", chan2.data(), 0, float(1.0 / fl.dth), false}};
        if (auto e = out.frame(t, ff); !e) return fail(4, e.error());
        return {};
    };
    auto row = [&](double dt, bool stats, bool profile) -> std::expected<void, RunError> {
        auto c = dev.cells();
        if (!c) return fail(6, c.error());
        std::vector<double> ry;
        const FvFlameMeasure m = measured(*c, ry);
        if (!std::isfinite(m.mass) || !std::isfinite(m.Tmax) || !std::isfinite(m.xf))
            return fail(7, "flame state is not finite or has no mid-temperature crossing (diverged)");
        lastHits = m.hits;
        lastMin = m.yMin;
        lastMax = m.yMax;
        if (stats) {
            auto to = dev.totals();
            if (!to) return fail(6, to.error());
            double sz = 0;
            for (std::size_t q = 0; q < c->T.size(); ++q)
                sz = std::max(sz, std::abs(c->T[q] + r.model.qcp * c->ry[0][q] / c->rho[q] - Hin));
            const double dxf = std::isfinite(xfPrev) && t > tPrev ? (m.xf - xfPrev) / (t - tPrev) : 0.0;
            const double v[] = {t,        dt,          bound.chem,      bound.diff, m.SL,     m.uRef - dxf, m.xf,
                                m.dth,    m.Tmax,      m.mass,          to->massTally, m.omegaEdge, m.hits, m.uRef,
                                m.pRef,   m.TRef,      m.yMin,          m.yMax,     m.pRef - m.slabs.back().p, sz,
                                to->divErr, to->eosErr, double(lastIters), m.burn,  m.fuel,   to->enthalpy - to->enthalpyTally};
            out.row("stats", v);
            xfPrev = m.xf;
            tPrev = t;
            cad.logged(step);
        }
        if (profile) {
            for (const FvFlameSlab& k : m.slabs) {
                const double v[] = {t, k.s, k.T, k.rho, k.u, k.p, k.ry / k.rho, k.omega};
                out.row("profiles", v);
            }
            prof.done(t);
        }
        if (!r.quiet && cad.printDue(step, t, r.tEnd)) {
            std::printf("t=%.3f/%.3f SL=%.6f xf=%.4f dth=%.4f Tmax/Tb=%.5f yHits=%g dt=%.2e pcg=%d%s\n", t, r.tEnd, m.SL,
                        m.xf, m.dth, m.Tmax / (fl.sigma * fl.Tu), m.hits, dt, lastIters, bindChem ? " (chem dt)" : "");
            std::fflush(stdout);
            cad.printed(step);
        }
        const double guard = 2.0 * fl.dth;
        if (m.xf < guard || m.xf > probe.length() - guard)
            return fail(8, std::format("flame not stabilized: drifted to the {} end at t={:.4g} (x_f={:.4g}; adjust "
                                       "--inflow-speed)",
                                       m.xf < guard ? "inflow" : "outflow", t, m.xf));
        return {};
    };

    while (t < r.tEnd) {
        if (stopRequested()) break;
        std::string diverged;
        if (auto e = internal::recomputeDt(sinceDt, dtStab, r.dtRecompute, r.dtMargin, 1e-4 * dtFirst, [&] {
                auto d = dev.stableDt(r.cfl, r.chemK);
                if (!d) {
                    diverged = d.error();
                    return -1.0;
                }
                bound = *d;
                const double a = std::min(d->conv, d->diff);
                bindChem = d->chem < a;
                return std::min(a, d->chem);
            });
            !e)
            return fail(7, diverged.empty() ? e.error() : diverged);
        if (dtFirst == 0.0) dtFirst = dtStab;
        // Equal steps to tEnd, never a sliver: the projection's memory of c.
        const double rem = r.tEnd - t;
        if (rem <= 1e-9 * dtStab) break;
        const double dt = rem / std::ceil(rem / dtStab);
        auto lg = dev.step(dt);
        if (!lg) return fail(7, std::format("t = {:g}: {}", t, lg.error()));
        lastIters = lg->iterations;
        unconverged += lg->unconverged;
        solves += lg->solves;
        t += dt;
        ++step;
        const bool stats = cad.due(step, t, r.tEnd), profile = prof.due(t, r.tEnd);
        if (stats || profile)
            if (auto e = row(dt, stats, profile); !e) return std::unexpected(e.error());
        if (out.wantsAnyFrame(t))
            if (auto e = writeFrame(); !e) return std::unexpected(e.error());
    }
    if (unconverged > 0)
        std::fprintf(stderr, "dns: warning: %ld of %ld pressure solves stopped short of --tol\n", unconverged, solves);
    if (lastHits > 0)
        std::fprintf(stderr,
                     "dns: warning: %g cells with Y outside [-1e-6, 1+1e-6] at the last stats row (counted, never "
                     "clamped): Y in [%.3g, 1%+.3g]\n",
                     lastHits, lastMin, lastMax - 1);
    std::printf("label: %s\n", r.label.c_str());
    st.steps = step;
    st.tStop = t;
    finishTiming(st, wall0, out);
    return st;
}

} // namespace dns::gpu
