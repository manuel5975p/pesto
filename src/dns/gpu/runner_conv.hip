#include "dns/interrupt.hpp"
#include "dns/gpu/runner_conv.hpp"

#include "dns/gpu/comp.hpp"
#include "dns/gpu/comp_conv.hpp"
#include "dns/gpu/derived.hpp"
#include "dns/gpu/device_util.hpp"
#include "dns/gpu/internal/run_loop.hpp"
#include "dns/gpu/reductions.hpp"
#include "dns/transport.hpp"

#include <hip/hip_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <vector>

namespace dns::gpu {

using internal::fail;
using internal::finishTiming;
using internal::wallGradientPinned;

std::expected<RunStats, RunError>
runConv(const Grid& g, const CompState& ic, const ConvRun& r, RunOutput& out)
{
    const auto gg = GpuGrid::from(g);
    const auto tr = Transport::air(r.mu, r.Tref, r.tinfK, r.Pr, r.gamma,
                                   r.Rgas, r.sutherland);

    auto dev = DeviceComp::make(g, tr);
    auto ds = DeviceCompState::make(g.size());
    auto red = ReduceScratch::make();
    if (!dev || !ds || !red) return fail(6, "GPU allocation failed");

    auto wall = DeviceConvWall::make(
        g, {.gamma = r.gamma, .Rgas = r.Rgas, .Tbot = r.Tbot, .Ttop = r.Ttop,
            .patch = r.patch, .closedSides = r.closedSides});
    if (!wall) return fail(6, "convection wall init failed");

    dev->setStageHook([&](DeviceCompState& s) { wall->apply(s); });
    // Gravity as an RHS term on the live stage state: it rides SSP-RK3
    // exactly, like the channel body force, and leaves the fused stage kernel
    // untouched (never a state edit).
    dev->setRhsHook([&](const DeviceCompState& s, DeviceCompState& o,
                        const StageCombine& cmb) {
        addGravityY(gg, r.gravity, cmb.cr, r.closedSides, s, o);
    });

    (void)ds->upload(ic);
    wall->apply(*ds);

    const int jLo = 1, jHi = g.ny - 2;              // fluid rows
    const double dT = r.Tbot - r.Ttop;              // the imposed drop, > 0
    // Thermal diffusivity at the reference state, kappa / (rho cp) = nu / Pr
    // with rho_ref = 1: the conduction scale every Nusselt number is measured
    // against.
    const double kapDiff = r.mu / r.Pr;
    // Plane means run over the FLUID (i,k): a lateral ghost column holds the
    // wall mirror, so the divisor must match the kernel's range.
    const int ox = sideGhostX(r.closedSides), oz = sideGhostZ(gg, r.closedSides);
    const double planeCells = double(g.nx - 2 * ox) * (g.nz - 2 * oz);
    // Area-mean floor temperature: the wall-gradient Nusselt number is built
    // from plane averages, so the pinned value must be the mean of the
    // per-column wall temperatures (see the Boussinesq twin).
    const double patchFrac = convPatchAreaFraction(gg, r.patch, r.closedSides);
    const double TwallBot = patchFrac * r.Tbot + (1.0 - patchFrac) * r.Ttop;

    auto du = DeviceScalar::make(g.size());
    auto dv = DeviceScalar::make(g.size());
    auto dw = DeviceScalar::make(g.size());
    auto dVort = DeviceScalar::make(g.size());
    auto dSpeed = DeviceScalar::make(g.size());
    if (!du || !dv || !dw || !dVort || !dSpeed)
        return fail(6, "GPU allocation failed");

    const std::string cols[] = {"t",      "dt",   "nuBot", "nuTop",
                                "nuFlux", "urms", "KE"};
    out.declareSeries("stats", cols);
    const std::string pcols[] = {"t", "y", "T", "rho", "v", "vT", "nu"};
    out.declareSeries("profiles", pcols);

    // Frame scales. Temperature rides the DIVERGING map as T - Tref, so the
    // mid-layer reads as the neutral background and the two plumes separate by
    // sign; +/- dT/2 is exactly the imposed range, so the walls saturate and
    // nothing in between clips.
    // Centred on the area-mean WALL temperature, not the mid-layer one: with a
    // hot patch the bulk sits near the lid, and a mid-layer-centred diverging
    // map would render the whole box one colour. Identical at patch = 1.
    const double tCentre = 0.5 * (TwallBot + r.Ttop);
    const float tempRef = float(tCentre);
    const float tempScale = float(std::max(std::abs(r.Tbot - tCentre),
                                           std::abs(r.Ttop - tCentre)));
    const float speedMax = float(1.6 * r.uFf);
    const float vortScale = float(r.uFf > 0 ? 8.0 * r.uFf / r.H : 8.0);
    std::vector<Real> hVort(g.size()), hSpeed(g.size()), hTemp(g.size());
    std::vector<float> vortF(g.size()), speedF(g.size()), tempF(g.size());
    auto writeFrame = [&](double t) {
        dev->primitives(*ds);
        primitiveVelocity(gg, ds->rho, ds->mx, ds->my, ds->mz, *du, *dv, *dw);
        if (g.nz == 1) vorticityZCentred(gg, *du, *dv, *dVort);
        else curlMagCentred(gg, *du, *dv, *dw, *dVort);
        speedCentred(gg, *du, *dv, *dw, *dSpeed);
        (void)dVort->download(hVort.data());
        (void)dSpeed->download(hSpeed.data());
        (void)dev->T().download(hTemp.data());
        for (std::size_t c = 0; c < g.size(); ++c)
        {
            vortF[c] = float(hVort[c]);
            speedF[c] = float(hSpeed[c]);
            tempF[c] = float(double(hTemp[c]) - tempRef);
        }
        const FrameField ff[3]{
            {g.nz == 1 ? "vort" : "curl", vortF.data(), 0, vortScale,
             g.nz == 1},
            {"speed", speedF.data(), 0, speedMax, false},
            {"temp", tempF.data(), -tempScale, tempScale, true}};
        if (auto e = out.frame(t, FrameViews{ff}); !e)
            std::fprintf(stderr, "frame write failed: %s\n", e.error().c_str());
    };

    DeviceBuffer<double> planeScratch;
    std::vector<double> ps; // g.ny * 6 plane moment sums

    struct ConvState
    {
        double nuBot = 0, nuTop = 0, nuFlux = 0, urms = 0, ke = 0;
    };
    // Row j's mean temperature and the local Nusselt number
    //   Nu(y) = (<rho v T>/<rho> - kapDiff d<T>/dy) / (kapDiff dT/H),
    // i.e. the total vertical heat flux over the conductive one. Quiescent
    // conduction gives exactly 1 at every height.
    auto measure = [&]() -> std::expected<ConvState, std::string> {
        dev->primitives(*ds);
        if (auto e = convPlaneStats(gg, r.closedSides, ds->rho, dev->u(),
                                    dev->v(), dev->w(), dev->T(), planeScratch,
                                    ps);
            e != hipSuccess)
            return std::unexpected("plane stats download failed");
        ConvState c;
        auto Tm = [&](int j) { return ps[j * 6 + 1] / planeCells; };
        const double d1 = 0.5 * g.hy, d2 = 1.5 * g.hy;
        c.nuBot = -wallGradientPinned(Tm(jLo) - TwallBot, Tm(jLo + 1) - TwallBot,
                                      d1, d2) * r.H / dT;
        c.nuTop = wallGradientPinned(Tm(jHi) - r.Ttop, Tm(jHi - 1) - r.Ttop,
                                     d1, d2) * r.H / dT;

        double mass = 0, keSum = 0, nuSum = 0;
        int nuRows = 0;
        for (int j = jLo; j <= jHi; ++j)
        {
            mass += ps[j * 6 + 0];
            keSum += ps[j * 6 + 4];
            // Centred d<T>/dy needs both neighbours, so the two rows adjacent
            // to a wall are left to the wall gradients above.
            if (j == jLo || j == jHi) continue;
            const double rhoBar = ps[j * 6 + 0] / planeCells;
            const double conv = ps[j * 6 + 5] / planeCells / rhoBar;
            const double dTdy = (Tm(j + 1) - Tm(j - 1)) / (2 * g.hy);
            nuSum += (conv - kapDiff * dTdy) / (kapDiff * dT / r.H);
            ++nuRows;
        }
        c.nuFlux = nuRows > 0 ? nuSum / nuRows : 0;
        c.urms = mass > 0 ? std::sqrt(keSum / mass) : 0;
        c.ke = 0.5 * g.cellVolume() * keSum;
        return c;
    };

    auto emitProfiles = [&](double t) {
        for (int j = jLo; j <= jHi; ++j)
        {
            const double rhoBar = ps[j * 6 + 0] / planeCells;
            const double Tbar = ps[j * 6 + 1] / planeCells;
            const double vBar = ps[j * 6 + 2] / planeCells;
            const double vT = ps[j * 6 + 5] / planeCells / rhoBar;
            double nuLocal = 0;
            if (j > jLo && j < jHi)
            {
                const double dTdy = (ps[(j + 1) * 6 + 1] - ps[(j - 1) * 6 + 1])
                                  / (planeCells * 2 * g.hy);
                nuLocal = (vT - kapDiff * dTdy) / (kapDiff * dT / r.H);
            }
            const double row[] = {t,     (j - jLo + 0.5) * g.hy, Tbar, rhoBar,
                                  vBar,  vT,                     nuLocal};
            (void)out.row("profiles", row);
        }
    };

    RunStats st;
    double t = 0.0;
    long lastLog = 0;
    // The PRINT cadence needs its own counter. Sharing lastLog with the CSV
    // cadence made the progress line unreachable: lastLog is reset every
    // statsEvery steps, so `steps - lastLog >= 10 * statsEvery` was never true
    // and the line appeared only at tEnd -- which left dns-ui's live plots
    // empty for the whole run.
    long lastPrint = 0;
    double dtStab = 0.0;
    int sinceDt = 1 << 30;
    double nextProf = r.profileDt > 0 ? r.profileDt : -1;
    const auto wall0 = std::chrono::steady_clock::now();

    while (t < r.tEnd)
    {
        // A stop signal ends the run the SAME way reaching tEnd does: return
        // normally, so every destructor runs.
        if (stopRequested()) break;
        // 1e-9: the convection cell runs on the channel's dt scale (a unit box
        // and an O(1) sound speed), not the body runs' -- so it takes the
        // channel's collapse threshold, not comp's 1e-6.
        if (auto e = internal::recomputeDt(
                sinceDt, dtStab, r.dtRecompute, r.dtMargin, 1e-9,
                [&] { return dev->stableDt(*ds, *red, r.cfl); });
            !e)
            return fail(7, e.error());
        const double dt = std::min(r.tEnd - t, dtStab);

        dev->step(*ds, Real(dt));
        wall->apply(*ds); // leave the state wall-consistent between steps
        t += dt;
        ++st.steps;

        const bool statsDue = st.steps - lastLog >= r.statsEvery || t >= r.tEnd;
        const bool profDue = nextProf >= 0 && (t >= nextProf || t >= r.tEnd);
        if (statsDue || profDue)
        {
            auto c = measure();
            if (!c) return fail(7, c.error());
            if (statsDue)
            {
                const double row[] = {t,          dt,      c->nuBot, c->nuTop,
                                      c->nuFlux,  c->urms, c->ke};
                (void)out.row("stats", row);
                if (st.steps - lastPrint >= 10 * r.statsEvery || t >= r.tEnd)
                {
                    std::printf("t=%.2f/%.2f Nu=%.3f/%.3f (flux %.3f) "
                                "urms/uff=%.3f dt=%.2e\n",
                                t, r.tEnd, c->nuBot, c->nuTop, c->nuFlux,
                                r.uFf > 0 ? c->urms / r.uFf : 0.0, dt);
                    std::fflush(stdout);
                    lastPrint = st.steps;
                }
                lastLog = st.steps;
            }
            if (profDue)
            {
                emitProfiles(t);
                while (nextProf <= t) nextProf += r.profileDt;
            }
        }
        if (out.wantsFrame(t) || out.wantsPng(t) || out.wantsWeb(t)
            || out.wantsShm(t))
            writeFrame(t);
    }
    (void)hipDeviceSynchronize();
    finishTiming(st, wall0, out);
    return st;
}

} // namespace dns::gpu
