#include "dns/gpu/runner_periodic.hpp"

#include "dns/diagnostics.hpp"
#include "dns/interrupt.hpp"
#include "dns/gpu/derived.hpp"
#include "dns/gpu/device_util.hpp"
#include "dns/gpu/internal/run_loop.hpp"
#include "dns/gpu/kernels.hpp"
#include "dns/gpu/periodic_frames.hpp"
#include "dns/gpu/reductions.hpp"
#include "dns/gpu/timestepper.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <format>
#include <string>
#include <vector>

namespace dns::gpu {

using internal::fail;

std::expected<RunStats, RunError>
runTgv3d(const Grid& g, const VectorField& ic, const TgvRun& r,
         PeriodicFrames* frames)
{
    auto ts = TimeStepper::make(g, {r.nu, r.cfl});
    auto D = DeviceField::make(g.size());
    auto red = ReduceScratch::make();
    if (!ts || !D || !red) return fail(6, "GPU init failed");
    if (uploadField(ic, *D) != hipSuccess)
        return fail(6, "initial condition upload failed");
    if (auto h = internal::checkDevice("the initial condition"); !h)
        return std::unexpected(h.error());
    const auto gg = GpuGrid::from(g);

    // The dissipation series IS this subcommand's output, so it
    // goes to stdout in the reference curves' format, not to a diagnostic.
    TgvSeries::header();
    TgvSeries series;
    double t = 0.0, nextOut = 0.0;
    long steps = 0;
    auto probe = [&] {
        series.row(t, kineticEnergy(gg, *D, *red) / r.volume,
                   r.nu * velocityGradientSq(gg, *D, *red));
    };
    while (t < r.tEnd)
    {
        // A stop signal ends the run the way reaching tEnd does.
        if (stopRequested()) break;
        if (t >= nextOut) { probe(); nextOut += r.outEvery; }
        if (frames && frames->wants(t)) frames->write(gg, *D, t);
        const double dt = ts->suggestDt(*D, r.tEnd - t);
        if (!(dt > 0) || !std::isfinite(dt))
            return fail(7, "stable dt is not finite and positive (the state "
                           "diverged)");
        ts->step(*D, dt);
        t += dt;
        ++steps;
    }
    probe();
    if (auto h = internal::checkDevice("the end of the run", &*red); !h)
        return std::unexpected(h.error());
    RunStats st;
    st.steps = steps;
    return st;
}

std::expected<HitAverages, RunError>
runHit(const Grid& g, const VectorField& ic, const HitRun& r, RunOutput& out,
       PeriodicFrames* frames, RunStats& st)
{
    auto ts = TimeStepper::make(g, {r.nu, 0.9});
    auto D = DeviceField::make(g.size());
    auto red = ReduceScratch::make();
    if (!ts || !D || !red) return fail(6, "GPU init failed");
    VectorField U = ic;
    if (uploadField(U, *D) != hipSuccess)
        return fail(6, "initial condition upload failed");
    if (auto h = internal::checkDevice("the initial condition"); !h)
        return std::unexpected(h.error());
    const auto gg = GpuGrid::from(g);

    double t = 0.0, dt = 0.0;
    long steps = 0;
    int spectrumIdx = 0, snapIdx = 0;
    double nextSpectrum = r.tSpin, nextSnap = r.tSpin;
    double epsSum = 0.0, kSum = 0.0, reSum = 0.0, wSum = 0.0;

    while (t < r.tEnd)
    {
        if (stopRequested()) break;
        const auto mean = meanVelocity(gg, *D, *red);
        const double kPrime = fluctuationEnergy(gg, *D, *red);
        // max(NaN, floor) is NaN: a diverged field would drive the forcing
        // with a NaN amplitude rather than stop.
        if (!std::isfinite(kPrime))
            return fail(7, "the fluctuation energy is not finite (the state "
                           "diverged)");
        const double A = r.epsF / (2.0 * std::max(kPrime, r.kFloor));
        ts->setForcing({A, {mean.x, mean.y, mean.z}});
        if (steps % 100 == 0)
            subtractMean(gg, Real(mean.x), Real(mean.y),
                              Real(mean.z), *D);

        if (frames && frames->wants(t)) frames->write(gg, *D, t);
        dt = ts->suggestDt(*D, r.tEnd - t);
        if (!(dt > 0) || !std::isfinite(dt))
            return fail(7, "stable dt is not finite and positive (the state "
                           "diverged)");
        ts->step(*D, dt);
        t += dt;
        ++steps;

        if (steps % r.statsEvery == 0)
        {
            if (auto h = internal::checkDevice("a stats row", &*red); !h)
                return std::unexpected(h.error());
            FlowStats fs{};
            fs.kPrime = fluctuationEnergy(gg, *D, *red);
            fs.uPrime = std::sqrt(2.0 * fs.kPrime / 3.0);
            fs.epsVisc = r.nu * velocityGradientSq(gg, *D, *red);
            if (fs.epsVisc > 0)
            {
                fs.eta = std::pow(r.nu * r.nu * r.nu / fs.epsVisc, 0.25);
                fs.reLambda =
                    fs.uPrime * fs.uPrime * std::sqrt(15.0 / (r.nu * fs.epsVisc));
                fs.hOverEta = double(g.hx) / fs.eta;
            }
            fs.courantPerDt = maxCourantPerDt(gg, *D, *red);
            fs.maxDiv = maxAbsDivergence(gg, *D, *red);
            const double row[] = {t, dt, A, fs.kPrime, fs.uPrime, fs.epsVisc,
                                  2.0 * A * fs.kPrime, fs.eta, fs.reLambda,
                                  fs.hOverEta, fs.courantPerDt * dt, fs.maxDiv};
            out.row("stats", row);
            if (fs.maxDiv * g.hx > 1e-3)
                std::fprintf(stderr, "WARN t=%.3f: maxDiv*h = %.3e\n", t,
                             fs.maxDiv * g.hx);
            if (t > r.tSpin)
            {
                epsSum += dt * r.statsEvery * fs.epsVisc;
                kSum += dt * r.statsEvery * fs.kPrime;
                reSum += dt * r.statsEvery * fs.reLambda;
                wSum += dt * r.statsEvery;
            }
        }

#if DNS_WITH_FFTW
        if (r.spectrum && t >= nextSpectrum)
        {
            (void)downloadField(*D, U);
            // std::format, NOT std::to_string: to_string is 6 DECIMALS, so
            // a shell energy of 1e-8 would be written as "0.000000".
            std::string csv = "k,E\n";
            const auto shellE = energySpectrum(g, U);
            for (std::size_t s = 1; s < shellE.size(); ++s)
                csv += std::format("{},{}\n", s, shellE[s]);
            if (auto e = out.auxText("spec", spectrumIdx++, ".csv", csv); !e)
                return fail(4, e.error());
            nextSpectrum += 0.25 * r.tEddy;
        }
#endif
        if (r.snapshot && t >= nextSnap)
        {
            (void)downloadField(*D, U);
            std::vector<float> raw;
            raw.reserve(3 * g.size());
            for (const Scalar* q : {&U.u, &U.v, &U.w})
                raw.insert(raw.end(), q->begin(), q->end());
            const int idx = snapIdx++;
            if (auto e = out.auxBinary("snap", idx, ".raw", raw); !e)
                return fail(4, e.error());
            const std::string meta = std::format(
                "{{\"n\":{},\"tBox\":{},\"tSI\":{},\"lBoxSI\":{},"
                "\"layout\":\"u,v,w float32 i-fastest MAC\"}}\n",
                g.nx, t, r.timeToSI(t), r.lBoxSI);
            if (auto e = out.auxText("snap", idx, ".json", meta); !e)
                return fail(4, e.error());
            nextSnap += r.tEddy;
        }
    }
    st.steps = steps;
    if (auto h = internal::checkDevice("the end of the run", &*red); !h)
        return std::unexpected(h.error());
    HitAverages a;
    if (wSum > 0)
    {
        a.epsVisc = epsSum / wSum;
        a.kPrime = kSum / wSum;
        a.reLambda = reSum / wSum;
    }
    return a;
}

} // namespace dns::gpu
