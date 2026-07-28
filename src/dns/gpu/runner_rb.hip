#include "dns/interrupt.hpp"
#include "dns/gpu/runner_rb.hpp"

#include "dns/gpu/boussinesq.hpp"
#include "dns/gpu/derived.hpp"
#include "dns/gpu/device_util.hpp"
#include "dns/gpu/internal/run_loop.hpp"
#include "dns/gpu/kernels.hpp"
#include "dns/gpu/poisson.hpp"
#include "dns/gpu/reductions.hpp"

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
runRb(const Grid& g, const VectorField& icU, const Scalar& icT, const RbRun& r,
      RunOutput& out)
{
    const auto gg = GpuGrid::from(g);

    auto dU0 = DeviceField::make(g.size());
    auto dU = DeviceField::make(g.size());
    auto dS = DeviceField::make(g.size());
    auto dT0 = DeviceScalar::make(g.size());
    auto dT = DeviceScalar::make(g.size());
    auto dTs = DeviceScalar::make(g.size());
    auto dDiv = DeviceScalar::make(g.size());
    auto dPhi = DeviceScalar::make(g.size());
    auto red = ReduceScratch::make();
    if (!dU0 || !dU || !dS || !dT0 || !dT || !dTs || !dDiv || !dPhi || !red)
        return fail(6, "GPU allocation failed");

    // A closed side is a zero-flux face for the pressure exactly as the plates
    // are, so the whole box becomes the all-Neumann one and its own solver
    // covers it -- no mixed transform is needed. Periodic sides keep the
    // FFT-in-x/z + exact-tridiagonal-in-y route.
    auto poisson = DevicePoissonSolver::make(
        gg, r.closedSides ? PoissonBc::NeumannBox : PoissonBc::PeriodicXZWallY);
    if (!poisson) return fail(6, "pressure solver init failed");

    {
        std::vector<Real> h(g.size());
        auto up = [&](const Scalar& src, DeviceScalar& dst) {
            for (std::size_t c = 0; c < g.size(); ++c) h[c] = Real(src[c]);
            (void)dst.upload(h.data());
        };
        up(icU.u, dU->u);
        up(icU.v, dU->v);
        up(icU.w, dU->w);
        up(icT, *dT);
    }
    applyRbWall(gg, Real(r.Tbot), Real(r.Ttop), Real(r.patch), r.closedSides,
                *dU, *dT);

    const double dT_imposed = r.Tbot - r.Ttop;
    const int jLo = 1, jHi = g.ny - 2;
    // Lateral ghost columns are the wall's, so every plane mean is over the
    // FLUID (i,k) only and the divisor has to agree with the kernel's range.
    const int ox = sideGhostX(r.closedSides), oz = sideGhostZ(gg, r.closedSides);
    const double planeCells = double(g.nx - 2 * ox) * (g.nz - 2 * oz);
    // Area-mean FLOOR temperature. The wall-gradient Nusselt number is built
    // from plane-averaged temperatures, and the gradient operator is linear,
    // so the plane average of the per-column gradients is the gradient of the
    // plane averages -- provided the pinned wall value is the area MEAN of the
    // per-column wall temperatures, not Tbot. Counted in cells, so it is the
    // discrete patch the kernel actually applies.
    const double patchFrac = rbPatchAreaFraction(gg, r.patch, r.closedSides);
    const double TwallBot = patchFrac * r.Tbot + (1.0 - patchFrac) * r.Ttop;

    // One SSP-RK3 stage: out = a*U0 + (1-a)*(in + dt*rhs), the temperature
    // riding the SAME combination so the two stay on one time level, then
    // project. Buoyancy enters the momentum rhs (hence the (1-a)*dt weight)
    // and is the only place the temperature acts on the flow.
    auto stage = [&](const DeviceField& inU, const DeviceScalar& inT,
                     DeviceField& outU, DeviceScalar& outT, double a,
                     double dt) {
        fusedRhsStage(gg, Real(r.nu), Real(0), 0, 0, 0, Real(a), *dU0,
                      Real(1.0 - a), inU, Real(dt), outU);
        addBuoyancy(gg, Real(r.gAlpha), Real(r.Tref), Real((1.0 - a) * dt), inT,
                    outU);
        stageTemperature(gg, Real(r.kappa), Real(a), *dT0, Real(1.0 - a), inU,
                         inT, Real(dt), outT);
        applyRbWall(gg, Real(r.Tbot), Real(r.Ttop), Real(r.patch), r.closedSides,
                    outU, outT);
        divergence(gg, outU, *dDiv, Real(1.0 / dt));
        poisson->solve(*dDiv, *dPhi);
        // dp/dn = 0 at a wall. The solve writes the ghost cells as zero, and
        // subtractGradient would read that as a real pressure and push a
        // velocity straight through the wall face.
        mirrorPressureWalls(gg, r.closedSides, *dPhi);
        subtractGradient(gg, *dPhi, Real(dt), outU);
        applyRbWall(gg, Real(r.Tbot), Real(r.Ttop), Real(r.patch), r.closedSides,
                    outU, outT);
    };

    const std::string cols[] = {"t",      "dt",   "nuBot", "nuTop",
                                "nuFlux", "urms", "KE",    "maxDiv"};
    out.declareSeries("stats", cols);
    const std::string pcols[] = {"t", "y", "T", "v", "vT", "nu"};
    out.declareSeries("profiles", pcols);

    auto du = DeviceScalar::make(g.size());
    auto dv = DeviceScalar::make(g.size());
    auto dw = DeviceScalar::make(g.size());
    auto dVort = DeviceScalar::make(g.size());
    auto dSpeed = DeviceScalar::make(g.size());
    if (!du || !dv || !dw || !dVort || !dSpeed)
        return fail(6, "GPU allocation failed");

    // Temperature frames ride the diverging map, so they need a reference that
    // reads as the NEUTRAL background. On a uniformly heated plate that is the
    // mid-layer temperature; with a hot patch the bulk equilibrates near the
    // LID instead, and a map still centred on mid-layer paints the whole box
    // one colour. Centring on the area-mean wall temperature tracks that, and
    // the half-range is stretched to whichever plate is further from it so
    // nothing clips. At patch = 1 this is exactly (Tref, dT/2) again.
    const double tCentre = 0.5 * (TwallBot + r.Ttop);
    const float tempRef = float(tCentre);
    const float tempScale = float(std::max(std::abs(r.Tbot - tCentre),
                                           std::abs(r.Ttop - tCentre)));
    const float speedMax = 1.6f;
    const float vortScale = 8.0f;
    std::vector<Real> hVort(g.size()), hSpeed(g.size()), hTemp(g.size());
    std::vector<float> vortF(g.size()), speedF(g.size()), tempF(g.size());
    auto writeFrame = [&](double t) {
        // MAC faces -> cell centres, the same averaging the plane stats use.
        centreVelocity(gg, *dU, *du, *dv, *dw);
        if (g.nz == 1) vorticityZCentred(gg, *du, *dv, *dVort);
        else curlMagCentred(gg, *du, *dv, *dw, *dVort);
        speedCentred(gg, *du, *dv, *dw, *dSpeed);
        (void)dVort->download(hVort.data());
        (void)dSpeed->download(hSpeed.data());
        (void)dT->download(hTemp.data());
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
    std::vector<double> ps;

    struct RbState
    {
        double nuBot = 0, nuTop = 0, nuFlux = 0, urms = 0, ke = 0, maxDiv = 0;
    };
    auto measure = [&]() -> std::expected<RbState, std::string> {
        if (auto e = rbPlaneStats(gg, r.closedSides, *dU, *dT, planeScratch, ps);
            e != hipSuccess)
            return std::unexpected("plane stats download failed");
        RbState s;
        auto Tm = [&](int j) { return ps[j * 4 + 0] / planeCells; };
        const double d1 = 0.5 * g.hy, d2 = 1.5 * g.hy;
        s.nuBot = -wallGradientPinned(Tm(jLo) - TwallBot, Tm(jLo + 1) - TwallBot,
                                      d1, d2) * r.H / dT_imposed;
        s.nuTop = wallGradientPinned(Tm(jHi) - r.Ttop, Tm(jHi - 1) - r.Ttop, d1,
                                     d2) * r.H / dT_imposed;
        double keSum = 0, nuSum = 0;
        int rows = 0;
        for (int j = jLo; j <= jHi; ++j)
        {
            keSum += ps[j * 4 + 3];
            if (j == jLo || j == jHi) continue;
            const double conv = ps[j * 4 + 2] / planeCells;
            const double dTdy = (Tm(j + 1) - Tm(j - 1)) / (2 * g.hy);
            nuSum += (conv - r.kappa * dTdy) / (r.kappa * dT_imposed / r.H);
            ++rows;
        }
        s.nuFlux = rows > 0 ? nuSum / rows : 0;
        const double cells = planeCells * double(jHi - jLo + 1);
        s.urms = std::sqrt(keSum / cells);
        s.ke = 0.5 * g.cellVolume() * keSum;
        return s;
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
    double nextProf = r.profileDt > 0 ? r.profileDt : -1;
    const double hmin = std::min({g.hx, g.hy, g.nz == 1 ? 1e30 : g.hz});
    // Explicit diffusive bound on the LARGER of the two diffusivities, and the
    // free-fall bound sqrt(h / (g alpha dT)) -- the latter is what limits the
    // very first steps, when the layer is at rest and the advective bound is
    // unbounded.
    const double dtDiff = 0.25 * hmin * hmin / std::max(r.nu, r.kappa);
    const double dtBuoy =
        std::sqrt(hmin / std::max(r.gAlpha * dT_imposed, 1e-30));
    const auto wall0 = std::chrono::steady_clock::now();

    while (t < r.tEnd)
    {
        if (stopRequested()) break;
        const double cpd = maxCourantPerDt(gg, *dU, *red);
        const double dtAdv = cpd > 0 ? r.cfl / cpd : 1e30;
        double dt = std::min({dtAdv, dtDiff, dtBuoy, r.tEnd - t});
        if (!(dt > 0) || !std::isfinite(dt))
            return fail(7, "stable dt is not finite and positive (the state "
                           "diverged)");

        (void)hipMemcpy(dU0->u.data(), dU->u.data(), g.size() * sizeof(Real),
                        hipMemcpyDeviceToDevice);
        (void)hipMemcpy(dU0->v.data(), dU->v.data(), g.size() * sizeof(Real),
                        hipMemcpyDeviceToDevice);
        (void)hipMemcpy(dU0->w.data(), dU->w.data(), g.size() * sizeof(Real),
                        hipMemcpyDeviceToDevice);
        (void)hipMemcpy(dT0->data(), dT->data(), g.size() * sizeof(Real),
                        hipMemcpyDeviceToDevice);

        stage(*dU0, *dT0, *dU, *dT, 0.0, dt);
        stage(*dU, *dT, *dS, *dTs, 0.75, dt);
        stage(*dS, *dTs, *dU, *dT, 1.0 / 3.0, dt);
        t += dt;
        ++st.steps;

        const bool statsDue = st.steps - lastLog >= r.statsEvery || t >= r.tEnd;
        const bool profDue = nextProf >= 0 && (t >= nextProf || t >= r.tEnd);
        if (statsDue || profDue)
        {
            auto s = measure();
            if (!s) return fail(7, s.error());
            if (statsDue)
            {
                divergence(gg, *dU, *dDiv, Real(1));
                zeroGhostCells(gg, r.closedSides, *dDiv);
                s->maxDiv = maxAbsScalar(*dDiv, *red);
                const double row[] = {t,        dt,      s->nuBot, s->nuTop,
                                      s->nuFlux, s->urms, s->ke,    s->maxDiv};
                (void)out.row("stats", row);
                if (st.steps - lastPrint >= 10 * r.statsEvery || t >= r.tEnd)
                {
                    std::printf("t=%.2f/%.2f Nu=%.3f/%.3f (flux %.3f) "
                                "urms=%.3f dt=%.2e\n",
                                t, r.tEnd, s->nuBot, s->nuTop, s->nuFlux,
                                s->urms, dt);
                    std::fflush(stdout);
                    lastPrint = st.steps;
                }
                lastLog = st.steps;
            }
            if (profDue)
            {
                for (int j = jLo; j <= jHi; ++j)
                {
                    const double Tbar = ps[j * 4 + 0] / planeCells;
                    const double vBar = ps[j * 4 + 1] / planeCells;
                    const double vT = ps[j * 4 + 2] / planeCells;
                    double nuLocal = 0;
                    if (j > jLo && j < jHi)
                    {
                        const double dTdy = (ps[(j + 1) * 4 + 0]
                                             - ps[(j - 1) * 4 + 0])
                                          / (planeCells * 2 * g.hy);
                        nuLocal = (vT - r.kappa * dTdy)
                                / (r.kappa * dT_imposed / r.H);
                    }
                    const double row[] = {t,  (j - jLo + 0.5) * g.hy, Tbar,
                                          vBar, vT,                  nuLocal};
                    (void)out.row("profiles", row);
                }
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
