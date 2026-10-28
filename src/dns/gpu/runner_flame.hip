#include "dns/gpu/runner_flame.hpp"

#include "dns/interrupt.hpp"

#include "dns/gpu/chem.hpp"
#include "dns/gpu/comp_nscbc.hpp"
#include "dns/gpu/comp_sink.hpp"
#include "dns/gpu/device_util.hpp"
#include "dns/gpu/internal/frame_scratch.hpp"
#include "dns/gpu/internal/run_loop.hpp"
#include "dns/gpu/reductions.hpp"

#include <hip/hip_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <format>
#include <optional>
#include <vector>

namespace dns::gpu {

using internal::fail;
using internal::finishTiming;

std::expected<RunStats, RunError> runFlame(const Grid& g, const CompState& ic,
                                           const std::vector<double>& rhoY,
                                           const FlameRun& r, RunOutput& out)
{
    auto dev = DeviceComp::make(g, r.tr);
    auto red = ReduceScratch::make();
    if (!dev || !red) return fail(6, "GPU allocation failed");
    if (!dev->enableSpecies(r.sp)) return fail(6, "species allocation failed");
    auto ds = DeviceCompState::make(g.size(), 0, false, r.sp.nY);
    if (!ds) return fail(6, "GPU allocation failed");

    DeviceCompNscbc::Config nc;
    nc.rhoInf = 1.0;
    nc.uInf = r.uIn;
    nc.pInf = r.pB; // the IC's burnt pressure: the IC is the fixed point
    nc.gamma = r.tr.gamma;
    nc.etaIn = r.etaIn;
    nc.Rgas = r.tr.Rgas;
    nc.thermalInflow = r.thermalInflow;
    nc.TInf = r.Tu;
    auto nscm = DeviceCompNscbc::make(g, nc);
    if (!nscm) return fail(6, "NSCBC init failed");
    DeviceCompNscbc nsc = std::move(*nscm);
    nsc.setFreestream(r.uIn);
    std::optional<DeviceCompSink> snk;
    if (r.sinkNb > 0) {
        auto m = DeviceCompSink::make(
            g, {.nb = r.sinkNb, .alpha = r.sinkAmp,
                .cInf = std::sqrt(r.tr.gamma * r.tr.Rgas * r.Tu)});
        if (!m) return fail(6, "sink init failed");
        snk = std::move(*m);
    }
    dev->setRhsHook([&](const DeviceCompState& s, DeviceCompState& o,
                        const StageCombine& cmb) {
        nsc.apply(*dev, s, o, cmb);
        dev->speciesOpenConsistency(s, o, cmb);
        if (snk) {
            snk->apply(s, o, cmb);
            snk->applySpecies(s, o, cmb);
        }
    });

    if (ds->upload(ic) != hipSuccess || ds->uploadSpecies(0, rhoY) != hipSuccess)
        return fail(6, "initial condition upload failed");
    if (auto h = internal::checkDevice("the initial condition"); !h)
        return std::unexpected(h.error());

    const double hx = g.hx, dV = g.cellVolume();
    const int nb = r.sinkNb;
    const double plane = double(g.ny) * g.nz;
    const double Tmid = 0.5 * (1 + r.sigma) * r.Tu;
    const double dTflame = (r.sigma - 1) * r.Tu;

    const std::string cols[] = {"t",      "dt",     "dtChem", "SL",
                                "SLdisp", "xf",     "dth",    "Tmax",
                                "mass",   "energy", "omegaEdge", "yHits",
                                "uFresh", "pFresh", "TFresh"};
    // The fresh reference column: 2 delta_th inside the inflow band edge,
    // ~10 delta_th upstream of the flame. The consumption speed is normalised
    // by ITS rho Y (the fresh state the flame actually burns), and the
    // displacement speed uses its velocity.
    const int iRef = std::min(g.nx - 1, nb + int(std::lround(2 * r.dth / hx)));
    if (auto e = out.declareSeries("stats", cols); !e) return fail(4, e.error());
    const std::string pcols[] = {"t", "x", "T", "rho", "u", "p", "Y", "omega"};
    if (auto e = out.declareSeries("profiles", pcols); !e)
        return fail(4, e.error());

    DeviceBuffer<double> colScratch;
    std::vector<double> ps;
    struct Meas
    {
        double SL = 0, xf = 0, dth = 0, Tmax = 0, mass = 0, energy = 0,
               omegaEdge = 0, yHits = 0, uRef = 0, pRef = 0, TRef = 0;
    };
    auto measure = [&]() -> std::expected<Meas, RunError> {
        dev->updateChem(*ds);
        if (flameColumnStats(*dev, *ds, colScratch, ps) != hipSuccess)
            return fail(6, "flame column stats download failed");
        Meas m;
        auto col = [&](int i, int q) { return ps[std::size_t(i) * kFlameSums + q]; };
        double wmax = 0, wedge = 0, gmax = -1;
        int kmax = -1;
        bool found = false;
        for (int i = 0; i < g.nx; ++i) {
            const double w = col(i, 4) / plane;
            m.SL += w * hx;
            m.mass += col(i, 0) * dV;
            m.energy += col(i, 5) * dV;
            m.yHits += col(i, 8);
            m.Tmax = std::max(m.Tmax, col(i, 2) / plane);
            wmax = std::max(wmax, w);
            const bool band = nb > 0 ? (i < nb || i >= g.nx - nb)
                                     : (i == 0 || i == g.nx - 1);
            if (band) wedge = std::max(wedge, w);
            if (i + 1 < g.nx) {
                const double Ta = col(i, 2) / plane, Tb = col(i + 1, 2) / plane;
                if (!found && Ta < Tmid && Tb >= Tmid) {
                    m.xf = (i + 0.5 + (Tmid - Ta) / (Tb - Ta)) * hx;
                    found = true;
                }
                const double s = (Tb - Ta) / hx;
                if (s > gmax) {
                    gmax = s;
                    kmax = i;
                }
            }
        }
        // Parabolic peak of the face slopes around the discrete maximum.
        double peak = gmax;
        if (kmax > 0 && kmax + 2 < g.nx) {
            auto sl = [&](int i) {
                return (col(i + 1, 2) - col(i, 2)) / plane / hx;
            };
            const double a = sl(kmax - 1), b = sl(kmax), c = sl(kmax + 1);
            const double den = a - 2 * b + c;
            if (den < 0) peak = b - (c - a) * (c - a) / (8 * den);
        }
        m.dth = peak > 0 ? dTflame / peak : 0;
        m.omegaEdge = wmax > 0 ? wedge / wmax : 0;
        // Consumption speed: integral of omega over the fresh rho Y.
        const double ryRef = col(iRef, 3) / plane;
        m.SL = ryRef > 0 ? m.SL / ryRef : NAN;
        m.uRef = col(iRef, 6) / plane;
        m.pRef = col(iRef, 7) / plane;
        m.TRef = col(iRef, 2) / plane;
        if (!found) m.xf = NAN;
        return m;
    };

    auto emitProfiles = [&](double t) {
        for (int i = 0; i < g.nx; ++i) {
            auto col = [&](int q) { return ps[std::size_t(i) * kFlameSums + q]; };
            const double row[] = {t,
                                  (i + 0.5) * hx,
                                  col(2) / plane,
                                  col(0) / plane,
                                  col(6) / plane,
                                  col(7) / plane,
                                  col(3) / col(0),
                                  col(4) / plane};
            out.row("profiles", row);
        }
    };

    const float tempScale = float(dTflame);
    const float omegaScale = float(1.0 / r.dth); // ~ peak omega of S_L = 1
    internal::FrameScratch fs;
    auto writeFrame = [&](double t) {
        if (!fs.ensure(g.size(), 3, false)) return;
        dev->updateChem(*ds);
        fs.take(0, dev->T(), r.Tu);
        fs.take(1, dev->speciesY(0));
        fs.take(2, dev->omega());
        const FrameField ff[3]{{"temp", fs.channel(0), 0, tempScale, false},
                               {"fuel", fs.channel(1), 0, 1.0f, false},
                               {"omega", fs.channel(2), 0, omegaScale, false}};
        fs.emit(out, t, ff);
    };

    RunStats st;
    double t = 0.0;
    internal::StatsCadence cad{r.statsEvery};
    double dtStab = 0.0, dtChem = 0.0;
    int sinceDt = 1 << 30;
    internal::ProfileSchedule prof(r.profileDt);
    double xfPrev = NAN, tPrev = 0;
    const double guard = nb * hx + 2 * r.dth;
    double lastHits = 0;
    const auto wall0 = std::chrono::steady_clock::now();
    bool bindChem = false;

    while (t < r.tEnd) {
        if (stopRequested()) break;
        if (auto e = internal::recomputeDt(
                sinceDt, dtStab, r.dtRecompute, r.dtMargin, 1e-9,
                [&] {
                    const double a = dev->stableDt(*ds, *red, r.cfl);
                    dtChem = chemStableDt(*dev, *ds, *red);
                    bindChem = dtChem < a;
                    return std::min(a, dtChem);
                });
            !e)
            return fail(7, e.error());
        const double dt = std::min(r.tEnd - t, dtStab);
        dev->step(*ds, Real(dt));
        t += dt;
        ++st.steps;

        const bool statsDue = cad.due(st.steps, t, r.tEnd);
        const bool profDue = prof.due(t, r.tEnd);
        if (statsDue || profDue) {
            auto m = measure();
            if (!m) return std::unexpected(m.error());
            if (auto h = internal::checkDevice("a stats row", &*red); !h)
                return std::unexpected(h.error());
            if (!std::isfinite(m->xf) || !std::isfinite(m->mass))
                return fail(7, "flame state is not finite (diverged)");
            if (statsDue) {
                const double dxf = std::isfinite(xfPrev) && t > tPrev
                                       ? (m->xf - xfPrev) / (t - tPrev)
                                       : 0.0;
                const double row[] = {t,            dt,
                                      dtChem,       m->SL,
                                      m->uRef - dxf, m->xf,
                                      m->dth,       m->Tmax,
                                      m->mass,      m->energy,
                                      m->omegaEdge, m->yHits,
                                      m->uRef,      m->pRef,
                                      m->TRef};
                out.row("stats", row);
                xfPrev = m->xf;
                tPrev = t;
                lastHits = m->yHits;
                if (cad.printDue(st.steps, t, r.tEnd)) {
                    std::printf("t=%.3f/%.3f SL=%.6f xf=%.4f dth=%.4f "
                                "Tmax/Tb=%.5f omegaEdge=%.2e yHits=%g dt=%.2e%s\n",
                                t, r.tEnd, m->SL, m->xf, m->dth,
                                m->Tmax / (r.sigma * r.Tu), m->omegaEdge,
                                m->yHits, dt, bindChem ? " (chem dt)" : "");
                    std::fflush(stdout);
                    cad.printed(st.steps);
                }
                cad.logged(st.steps);
            }
            if (profDue) {
                emitProfiles(t);
                prof.done(t);
            }
            if (m->xf < guard || m->xf > g.nx * hx - guard)
                return fail(8, std::format(
                                   "flame not stabilized: drifted to the "
                                   "{} band at t={:.4g} (x_f={:.4g}; adjust "
                                   "--inflow-speed)",
                                   m->xf < guard ? "inflow" : "outflow", t,
                                   m->xf));
        }
        if (out.wantsAnyFrame(t)) writeFrame(t);
    }
    if (auto h = internal::checkDevice("the end of the run", &*red); !h)
        return std::unexpected(h.error());
    if (lastHits > 0)
        std::fprintf(stderr,
                     "dns: warning: %g cells with Y outside [-1e-6, 1+1e-6] "
                     "at the last stats row (counted, never clamped)\n",
                     lastHits);
    std::printf("label: %s\n", r.label.c_str());
    finishTiming(st, wall0, out);
    st.tStop = t;
    return st;
}

} // namespace dns::gpu
