#include "dns/gpu/runner_channel.hpp"

#include "dns/curvmesh.hpp"
#include "dns/gpu/comp.hpp"
#include "dns/gpu/comp_channel.hpp"
#include "dns/gpu/derived.hpp"
#include "dns/gpu/internal/derived_remap.hpp"
#include "dns/gpu/internal/run_loop.hpp"
#include "dns/gpu/device_util.hpp"
#include "dns/gpu/reductions.hpp"
#include "dns/remap.hpp"

#include <hip/hip_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <vector>

namespace dns::gpu {

using internal::fail;
using internal::finishTiming;

namespace {


// One-sided 2nd-order wall gradient from the first two off-wall values with
// u(0) = 0 pinned: du/dy|0 = (u1 d2^2 - u2 d1^2) / (d1 d2 (d2 - d1)).
double wallGradient(double u1, double u2, double d1, double d2)
{
    return (u1 * d2 * d2 - u2 * d1 * d1) / (d1 * d2 * (d2 - d1));
}

} // namespace

std::expected<RunStats, RunError>
runChannel(const Grid& g, const CompState& ic, const ChannelRun& r,
           RunOutput& out)
{
    const auto gg = GpuGrid::from(g);
    const ChannelStretchY remap{double(g.ny) * g.hy, g.ny, r.stretchBeta,
                                r.delta};

    auto m = CurvMetrics::build(CurvMesh::channelY(g, r.stretchBeta, r.delta));
    if (!m) return fail(2, "channel mesh: " + m.error());
    auto dev = DeviceComp::make(*m, r.gamma, r.mu);
    auto ds = DeviceCompState::make(g.size());
    auto red = ReduceScratch::make();
    if (!dev || !ds || !red) return fail(6, "GPU allocation failed");

    auto wall = DeviceChannelWall::make(
        g, {r.gamma, r.Rgas, r.Twall, r.isothermal});
    if (!wall) return fail(6, "channel wall init failed");

    dev->setStageHook([&](DeviceCompState& s) { wall->apply(s); });
    // Body force as an RHS term on the live stage primitives: it rides
    // SSP-RK3 exactly, like the NSCBC rhs overwrite (never a state edit).
    dev->setRhsHook([&](const DeviceCompState&, DeviceCompState& k) {
        addBodyForceX(gg, r.fBody, dev->u(), k);
    });

    (void)ds->upload(ic);
    wall->apply(*ds);

    // Host-side row geometry: physical cell-centre y and row heights.
    const double nu = r.mu; // rho_wall = 1 in wall units
    std::vector<double> yc(g.ny), dy(g.ny);
    for (int j = 0; j < g.ny; ++j)
    {
        yc[j] = remap.mapY((j + 0.5) * g.hy);
        dy[j] = remap.mapY((j + 1.0) * g.hy) - remap.mapY(double(j) * g.hy);
    }
    const int jLo = 1, jHi = g.ny - 2; // fluid rows

    auto du = DeviceScalar::make(g.size());
    auto dv = DeviceScalar::make(g.size());
    auto dw = DeviceScalar::make(g.size());
    auto dVort = DeviceScalar::make(g.size());
    auto dSpeed = DeviceScalar::make(g.size());
    if (!du || !dv || !dw || !dVort || !dSpeed)
        return fail(6, "GPU allocation failed");

    const std::string cols[] = {"t",       "dt",      "ubulk", "utauBot",
                                "utauTop", "retau",   "KE"};
    out.declareSeries("stats", cols);
    const std::string pcols[] = {"t",    "y",    "yplus", "uplus", "urms",
                                 "vrms", "wrms", "uv",    "rho",   "T"};
    out.declareSeries("profiles", pcols);

    // Log-law bulk-velocity estimate, resolved once in validate() (it also
    // sets pRef from the bulk Mach there); re-deriving it here would let the
    // two drift apart.
    const double ubEst = r.ubEst;
    const float vortScale = float(r.curlMax > 0 ? r.curlMax : r.reTau / 4.0);
    const float speedMax = float(1.6 * ubEst);
    std::vector<Real> hVort(g.size()), hSpeed(g.size());
    std::vector<float> vortF(g.size()), speedF(g.size());
    auto writeFrame = [&](double t) {
        primitiveVelocity(gg, ds->rho, ds->mx, ds->my, ds->mz, *du, *dv, *dw);
        if (g.nz == 1) vorticityZRemap(gg, remap, *du, *dv, *dVort);
        else curlMagCentred(gg, *du, *dv, *dw, *dVort);
        speedCentred(gg, *du, *dv, *dw, *dSpeed);
        (void)dVort->download(hVort.data());
        (void)dSpeed->download(hSpeed.data());
        for (std::size_t c = 0; c < g.size(); ++c)
        {
            vortF[c] = float(hVort[c]);
            speedF[c] = float(hSpeed[c]);
        }
        const FrameField ff[2]{
            {g.nz == 1 ? "vort" : "curl", vortF.data(), 0, vortScale,
             g.nz == 1},
            {"speed", speedF.data(), 0, speedMax, false}};
        if (auto e = out.frame(t, FrameViews{ff}); !e)
            std::fprintf(stderr, "frame write failed: %s\n", e.error().c_str());
    };

    DeviceBuffer<double> planeScratch;
    std::vector<double> ps; // g.ny * 8 plane moment sums

    struct WallState
    {
        double ubulk = 0, utauBot = 0, utauTop = 0, ke = 0;
    };
    auto measure = [&]() -> std::expected<WallState, std::string> {
        dev->primitives(*ds);
        if (auto e = channelPlaneStats(gg, ds->rho, dev->u(), dev->v(),
                                       dev->w(), dev->T(), planeScratch, ps);
            e != hipSuccess)
            return std::unexpected("plane stats download failed");
        WallState w;
        double mass = 0, mom = 0, ke = 0;
        for (int j = jLo; j <= jHi; ++j)
        {
            mass += dy[j] * ps[j * 8 + 0];
            mom += dy[j] * ps[j * 8 + 1];
            ke += dy[j] * (ps[j * 8 + 3] + ps[j * 8 + 4] + ps[j * 8 + 5]);
        }
        w.ubulk = mass > 0 ? mom / mass : 0;
        w.ke = 0.5 * g.hx * g.hz * ke;
        auto favreU = [&](int j) { return ps[j * 8 + 1] / ps[j * 8 + 0]; };
        w.utauBot = std::sqrt(std::max(
            nu * wallGradient(favreU(jLo), favreU(jLo + 1), yc[jLo],
                              yc[jLo + 1]),
            0.0));
        const double dTop1 = 2 * r.delta - yc[jHi], dTop2 = 2 * r.delta - yc[jHi - 1];
        w.utauTop = std::sqrt(std::max(
            nu * wallGradient(favreU(jHi), favreU(jHi - 1), dTop1, dTop2),
            0.0));
        return w;
    };

    auto emitProfiles = [&](double t, const WallState& w) {
        const double ut = std::max(0.5 * (w.utauBot + w.utauTop), 1e-12);
        for (int j = jLo; j <= jHi; ++j)
        {
            const double rhoM = ps[j * 8 + 0] / (double(g.nx) * g.nz);
            const double uF = ps[j * 8 + 1] / ps[j * 8 + 0];
            const double vF = ps[j * 8 + 2] / ps[j * 8 + 0];
            const double uu = ps[j * 8 + 3] / ps[j * 8 + 0] - uF * uF;
            const double vv = ps[j * 8 + 4] / ps[j * 8 + 0] - vF * vF;
            const double ww = ps[j * 8 + 5] / ps[j * 8 + 0];
            const double uv = ps[j * 8 + 6] / ps[j * 8 + 0] - uF * vF;
            const double TM = ps[j * 8 + 7] / ps[j * 8 + 0];
            const double dWall = std::min(yc[j], 2 * r.delta - yc[j]);
            const double row[] = {t,
                                  yc[j],
                                  dWall * ut / nu,
                                  uF / ut,
                                  std::sqrt(std::max(uu, 0.0)) / ut,
                                  std::sqrt(std::max(vv, 0.0)) / ut,
                                  std::sqrt(std::max(ww, 0.0)) / ut,
                                  uv / (ut * ut),
                                  rhoM,
                                  TM};
            (void)out.row("profiles", row);
        }
    };

    RunStats st;
    double t = 0.0;
    long lastLog = 0;
    double dtStab = 0.0;
    int sinceDt = 1 << 30;
    double nextProf = r.profileDt > 0 ? r.profileDt : -1;
    const auto wall0 = std::chrono::steady_clock::now();

    while (t < r.tEnd)
    {
        // 1e-9: the channel's own collapse threshold (wall units put dt on a
        // different scale than the body runs). NOT shared with comp's 1e-6.
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
            auto w = measure();
            if (!w) return fail(7, w.error());
            if (statsDue)
            {
                const double retau =
                    0.5 * (w->utauBot + w->utauTop) * r.delta / nu;
                const double row[] = {t,          dt,         w->ubulk,
                                      w->utauBot, w->utauTop, retau,
                                      w->ke};
                (void)out.row("stats", row);
                if (st.steps - lastLog >= 10 * r.statsEvery || t >= r.tEnd)
                {
                    std::printf("t=%.2f/%.2f ub=%.2f Re_tau=%.1f "
                                "(bot %.3f top %.3f) dt=%.2e\n",
                                t, r.tEnd, w->ubulk, retau, w->utauBot,
                                w->utauTop, dt);
                    std::fflush(stdout);
                }
                lastLog = st.steps;
            }
            if (profDue)
            {
                emitProfiles(t, *w);
                while (nextProf <= t) nextProf += r.profileDt;
            }
        }
        if (out.wantsFrame(t) || out.wantsPng(t) || out.wantsWeb(t))
            writeFrame(t);
    }
    (void)hipDeviceSynchronize();
    finishTiming(st, wall0, out);
    return st;
}

} // namespace dns::gpu
