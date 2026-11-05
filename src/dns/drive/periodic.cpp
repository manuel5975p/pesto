// dns tgv, dns hit.
#include "dns/drive/internal.hpp"

#include "dns/config.hpp"
#include "dns/diagnostics.hpp"
#include "dns/hit_ic.hpp"
#include "dns/interrupt.hpp"
#include "dns/operators.hpp"
#include "dns/output.hpp"
#include "dns/taylor_green.hpp"
#include "dns/timestepper.hpp"
#include "dns/units.hpp"
#include "dns/vtk_writer.hpp"

#if DNS_WITH_GPU
#include "dns/gpu/device_util.hpp"
#include "dns/gpu/periodic_frames.hpp"
#include "dns/gpu/runner_periodic.hpp"
#endif

#include <cmath>
#include <cstdio>
#include <numbers>
#include <optional>
#include <string>

namespace dns::drive {

using std::numbers::pi;

int runTgv(const Case& c)
{
    if (c.fvIncomp) return runFvIncompTgv(c);
    const auto& g = c.grid;
    const double nu = c.nu;

    if (g.nz == 1)
    {
        // The 2D analytic case: an exact solution exists, so this is a
        // convergence check rather than a benchmark. CPU reference stepper.
        auto U = VectorField::zeros(g);
        taylorGreen(g, nu, 0.0, U);
        TimeStepper ts(g, {.nu = nu, .cfl = 0.5});
        const double ke0 = kineticEnergy(g, U);
        double t = 0.0;
        long steps = 0;
        while (t < c.tEnd)
        {
            if (stopRequested()) break;
            const double dt = ts.suggestDt(U, c.tEnd - t);
            if (!(dt > 0) || !std::isfinite(dt))
                return die(exitNumerical, "stable dt is not finite and positive "
                              "(the state diverged)");
            ts.step(U, dt);
            t += dt;
            ++steps;
        }
        const double err = taylorGreenError(g, nu, t, U);
        const double ke = kineticEnergy(g, U);
        std::printf("n=%d nu=%g T=%g steps=%ld\n", g.nx, nu, t, steps);
        if (c.emit.contains(Emit::Error))
        {
            std::printf("max error vs analytic: %.3e\n", err);
            std::printf("KE ratio: %.6f (analytic %.6f)\n", ke / ke0,
                        std::exp(-4.0 * nu * t));
        }
        if (c.emit.contains(Emit::Vtk))
        {
            if (auto r = writeVtk(g, U, c.outPrefix + ".vtk"); !r)
                return die(exitOutput, r.error());
        }
        return 0;
    }

    // 3D Re1600 benchmark: a dissipation time series on stdout, the format the
    // spectral reference curves are compared against.
    const double vol = g.cellVolume() * double(g.size());
    const StepParams sp{nu, c.cfl};
    auto U = VectorField::zeros(g);
    taylorGreen3D(g, U);

    const double outEvery = 0.1;

#if DNS_WITH_GPU
    if (haveGpu(c))
    {
        std::optional<RunOutput> slices;
        std::optional<gpu::PeriodicFrames> frames;
        if (c.emit.contains(Emit::Slices))
        {
            auto oc = outputConfig(c, "");
            oc.kinds = {Emit::Slices};
            auto o = RunOutput::make(oc, c.meta, g,
                                     FrameGeometry::centred(g, 2 * pi));
            if (!o) return die(exitOutput, o.error());
            slices = std::move(*o);
            auto f = gpu::PeriodicFrames::make(g, *slices);
            if (!f) return die(exitOutput, f.error());
            frames = std::move(*f);
        }
        std::printf("# 3D TGV n=%d Re=%g cfl=%g backend=%s\n", g.nx, c.re, c.cfl,
                    gpu::backendName);
        gpu::TgvRun tr;
        tr.nu = nu;
        tr.cfl = sp.cfl;
        tr.tEnd = c.tEnd;
        tr.outEvery = outEvery;
        tr.volume = vol;
        auto st = gpu::runTgv3d(g, U, tr, frames ? &*frames : nullptr);
        if (!st) return die(st.error().code, st.error().message);
        if (slices)
            if (auto e = slices->finish(); !e) return die(exitOutput, e.error());
        std::fprintf(stderr, "done: %ld steps (%s)\n", st->steps, gpu::backendName);
        return 0;
    }
#endif
    TimeStepper ts(g, sp);
    std::printf("# 3D TGV n=%d Re=%g cfl=%g backend=cpu\n", g.nx, c.re, c.cfl);
    TgvSeries::header();
    TgvSeries series;
    double t = 0.0, nextOut = 0.0;
    long steps = 0;
    auto probe = [&] {
        series.row(t, kineticEnergy(g, U) / vol, nu * velocityGradientSq(g, U));
    };
    while (t < c.tEnd)
    {
        if (stopRequested()) break;
        if (t >= nextOut) { probe(); nextOut += outEvery; }
        const double dt = ts.suggestDt(U, c.tEnd - t);
        if (!(dt > 0) || !std::isfinite(dt))
            return die(exitNumerical, "stable dt is not finite and positive "
                          "(the state diverged)");
        ts.step(U, dt);
        t += dt;
        ++steps;
    }
    probe();
    std::fprintf(stderr, "done: %ld steps (cpu)\n", steps);
    return 0;
}

int runHit(const Case& c)
{
    if (!haveGpu(c)) return die(exitNoGpu, "no GPU device (dns hit is GPU-only)");
#if DNS_WITH_GPU
    const auto& g = c.grid;
    const auto d = HitDesign::make(g.nx, c.nu, c.urms);
    const double nu = d.nuBox(), epsF = d.epsBox(), tE = d.tEddyBox();
    std::printf("# HIT n=%d  SI: u'=%g m/s nu=%g  L=%.3e m  eps=%.3e W/kg  "
                "eta=%.3e m  Re_lambda=%.1f  T_e=%.3e s\n",
                g.nx, c.urms, c.nu, d.lBoxSI, d.epsSI, d.etaSI, d.reLambda,
                d.tEddySI);
    std::printf("# box units: nu=%.6e eps_f=%.6e T_e=%.4f  spin=%g Te stats=%g Te\n",
                nu, epsF, tE, c.spin, c.stats);

    auto U = VectorField::zeros(g);
    hitInitialCondition(g, 1.0, c.seed, U);

    const LinearForcing forcing{epsF, 1.5 / 5.0}; // kFloor = k_stationary/5
    const double tEnd = (c.spin + c.stats) * tE, tSpin = c.spin * tE;
    // Phase 3c: ONE sink for everything hit writes -- stats, slices, spectra
    // and snapshots. The spectra and snapshots used to go straight to
    // std::ofstream behind RunOutput's back, so a failed write was silent and
    // a full destination surfaced hours into a run instead of in the first
    // second.
    const auto oc = outputConfig(c, "");
    auto outOr = RunOutput::make(oc, c.meta, g, FrameGeometry::centred(g, 2 * pi));
    if (!outOr) return die(exitOutput, outOr.error());
    RunOutput out = std::move(*outOr);
    const std::string statsCols[] = {"t",   "dt",       "A",        "kPrime",
                                     "uPrime", "epsVisc", "epsInj",  "eta",
                                     "reLambda", "hOverEta", "cfl", "maxDiv"};
    if (auto e = out.declareSeries("stats", statsCols); !e)
        return die(exitOutput, e.error());

    std::optional<gpu::PeriodicFrames> frames;
    if (c.emit.contains(Emit::Slices))
    {
        auto f = gpu::PeriodicFrames::make(g, out);
        if (!f) return die(exitOutput, f.error());
        frames = std::move(*f);
    }

    gpu::HitRun hr;
    hr.nu = nu;
    hr.epsF = epsF;
    hr.kFloor = forcing.kFloor;
    hr.tEnd = tEnd;
    hr.tSpin = tSpin;
    hr.tEddy = tE;
    hr.spectrum = c.emit.contains(Emit::Spectrum);
    hr.snapshot = c.emit.contains(Emit::Snapshot);
    hr.lBoxSI = d.lBoxSI;
    hr.timeToSI = [&d](double t) { return d.units().timeToSI(t); };
    gpu::RunStats st;
    auto avg = gpu::runHit(g, U, hr, out, frames ? &*frames : nullptr, st);
    if (!avg) return die(avg.error().code, avg.error().message);
    if (auto e = out.finish(); !e) return die(exitOutput, e.error());
    const long steps = st.steps;

    const double uPrime = std::sqrt(2.0 * avg->kPrime / 3.0);
    std::printf("# %ld steps.  Stationary averages over %.1f Te:\n", steps, c.stats);
    std::printf("#   eps_visc/eps_f = %.4f   u'/target = %.4f   Re_lambda = %.1f "
                "(design %.1f)\n",
                avg->epsVisc / epsF, uPrime, avg->reLambda, d.reLambda);
    std::printf("#   SI: u' = %.2f m/s  eps = %.4e W/kg\n",
                d.units().velocityToSI(uPrime), d.units().epsToSI(avg->epsVisc));
    return 0;
#else
    return exitNoGpu;
#endif
}


} // namespace dns::drive
