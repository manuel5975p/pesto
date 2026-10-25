#include "dns/gpu/runner_channel.hpp"

#include "dns/interrupt.hpp"

#include "dns/curvmesh.hpp"
#include "dns/gpu/comp.hpp"
#include "dns/gpu/comp_channel.hpp"
#include "dns/gpu/derived.hpp"
#include "dns/gpu/device_util.hpp"
#include "dns/gpu/internal/derived_remap.hpp"
#include "dns/gpu/internal/frame_scratch.hpp"
#include "dns/gpu/internal/run_loop.hpp"
#include "dns/gpu/reductions.hpp"
#include "dns/remap.hpp"
#include "dns/steady_stop.hpp"

#include <hip/hip_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <format>
#include <span>
#include <string>
#include <vector>

namespace dns::gpu {

using internal::fail;
using internal::finishTiming;

using internal::wallGradientPinned;

std::expected<RunStats, RunError>
runChannel(const Grid& g, const CompState& ic, const ChannelRun& r,
           RunOutput& out)
{
    const auto gg = GpuGrid::from(g);
    const ChannelStretchY remap{double(g.ny) * g.hy, g.ny, r.stretchBeta,
                                r.delta};
    const bool rans = r.rans != RansModel::None;
    const bool sst = ransSst(r.rans);
    const bool steady = rans && r.steadyTol > 0;

    auto m = CurvMetrics::build(CurvMesh::channelY(g, r.stretchBeta, r.delta));
    if (!m) return fail(2, "channel mesh: " + m.error());
    auto dev = DeviceComp::make(*m, r.gamma, r.mu);
    // Steady: the compensated state, so pseudo-time updates below ulp(q)/2
    // still accumulate (docs/rans.md, "The FP32 update floor"). The channel
    // wall is the only state-overwriting hook here, and it keeps the low words.
    auto ds = DeviceCompState::make(g.size(), ransScalars(r.rans), steady);
    auto red = ReduceScratch::make();
    if (!dev || !ds || !red) return fail(6, "GPU allocation failed");
    // Both plates are walls: mu_t is mirrored odd on both ghost rows, so it
    // is exactly 0 on the wall faces (docs/rans.md).
    if (rans && !dev->enableRans({.model = r.rans,
                                  .wallDist = r.wallDist,
                                  .mirrorYLo = true,
                                  .mirrorYHi = true}))
        return fail(6, "GPU allocation failed (RANS)");
    if (steady && !dev->enableCompensation())
        return fail(6, "GPU allocation failed (compensated state)");

    // SST's wall omega: TMR's 60 nu_w / (beta1 d1^2) with d1 the first cell
    // centre's distance (the same on both plates: the mesh is a reflection).
    const double d1 = sst ? r.wallDist[g.idx(0, 1, 0)] : 0.0;
    auto wall = DeviceChannelWall::make(
        g, {r.gamma, r.Rgas, r.Twall, r.isothermal,
            sst ? sst::omegaWallFactor * r.mu / (d1 * d1) : 0.0});
    if (!wall) return fail(6, "channel wall init failed");

    dev->setStageHook([&](DeviceCompState& s) { wall->apply(s); });
    // Body force as an RHS term on the live stage primitives: it rides
    // SSP-RK3 exactly, like the NSCBC rhs overwrite (never a state edit).
    dev->setRhsHook([&](const DeviceCompState&, DeviceCompState& out,
                        const StageCombine& cmb) {
        addBodyForceX(gg, r.fBody, cmb, dev->u(), out);
    });

    bool upOk = ds->upload(ic) == hipSuccess;
    for (int i = 0; i < ds->nT; ++i)
        upOk = upOk && ds->uploadTurb(i, r.icTurb[i]) == hipSuccess;
    if (!upOk) return fail(6, "initial condition upload failed");
    if (auto h = internal::checkDevice("the initial condition"); !h)
        return std::unexpected(h.error());
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

    // RANS appends columns; the DNS channel's series are untouched.
    std::vector<std::string> cols = {"t",       "dt",      "ubulk", "utauBot",
                                     "utauTop", "retau",   "KE"};
    if (rans) cols.emplace_back("mutMax");
    if (sst) cols.emplace_back("floorHits");
    if (steady)
        cols.insert(cols.end(), {"iter", "resid", "resU", "resE", "resT",
                                 "Tbulk", "remaining"});
    if (auto e = out.declareSeries("stats", cols); !e)
        return fail(4, e.error());
    std::vector<std::string> pcols = {"t",    "y",    "yplus", "uplus", "urms",
                                      "vrms", "wrms", "uv",    "rho",   "T"};
    if (rans) pcols.emplace_back("mutRatio");
    if (rans && !sst) pcols.emplace_back("nutilde");
    if (sst) pcols.insert(pcols.end(), {"kplus", "omegaplus"});
    if (auto e = out.declareSeries("profiles", pcols); !e)
        return fail(4, e.error());

    // Log-law bulk-velocity estimate, resolved once in validate() (it also
    // sets pRef from the bulk Mach there); re-deriving it here would let the
    // two drift apart.
    const double ubEst = r.ubEst;
    const float vortScale = float(r.curlMax > 0 ? r.curlMax : r.reTau / 4.0);
    const float speedMax = float(1.6 * ubEst);
    internal::FrameScratch fs;
    auto writeFrame = [&](double t) {
        if (!fs.ensure(g.size(), 2, true)) return;
        fs.centreFromConserved(gg, *ds);
        if (g.nz == 1) vorticityZRemap(gg, remap, fs.u(), fs.v(), fs.vort());
        else curlMagCentred(gg, fs.u(), fs.v(), fs.w(), fs.vort());
        speedCentred(gg, fs.u(), fs.v(), fs.w(), fs.speed());
        fs.take(0, fs.vort());
        fs.take(1, fs.speed());
        const FrameField ff[2]{
            {g.nz == 1 ? "vort" : "curl", fs.channel(0), 0, vortScale,
             g.nz == 1},
            {"speed", fs.channel(1), 0, speedMax, false}};
        fs.emit(out, t, ff);
    };

    DeviceBuffer<double> planeScratch, mutScratch, ntScratch, nt2Scratch;
    std::vector<double> ps;          // g.ny * 8 plane moment sums
    // g.ny * 2 {sum, max}: mu_t, and the conserved scalars (rho nu~ | rho k,
    // rho omega)
    std::vector<double> pmut, pnt, pnt2;

    struct WallState
    {
        double ubulk = 0, utauBot = 0, utauTop = 0, ke = 0, mutMax = 0;
        // Favre bulk T - Twall and the y-integrated conserved scalars
        // (rho nu~ | rho k, rho omega): the steady stop's other integrals.
        double tExcess = 0, turb[2] = {};
    };
    auto measure = [&]() -> std::expected<WallState, RunError> {
        dev->primitives(*ds);
        if (auto e = channelPlaneStats(gg, ds->rho, dev->u(), dev->v(),
                                       dev->w(), dev->T(), planeScratch, ps);
            e != hipSuccess)
            return fail(6, "plane stats download failed");
        WallState w;
        double mass = 0, mom = 0, ke = 0, heat = 0;
        for (int j = jLo; j <= jHi; ++j)
        {
            mass += dy[j] * ps[j * 8 + 0];
            mom += dy[j] * ps[j * 8 + 1];
            ke += dy[j] * (ps[j * 8 + 3] + ps[j * 8 + 4] + ps[j * 8 + 5]);
            heat += dy[j] * ps[j * 8 + 7];
        }
        w.ubulk = mass > 0 ? mom / mass : 0;
        w.tExcess = mass > 0 ? heat / mass - r.Twall : 0;
        w.ke = 0.5 * g.hx * g.hz * ke;
        auto favreU = [&](int j) { return ps[j * 8 + 1] / ps[j * 8 + 0]; };
        w.utauBot = std::sqrt(std::max(
            nu * wallGradientPinned(favreU(jLo), favreU(jLo + 1), yc[jLo],
                                    yc[jLo + 1]),
            0.0));
        const double dTop1 = 2 * r.delta - yc[jHi], dTop2 = 2 * r.delta - yc[jHi - 1];
        w.utauTop = std::sqrt(std::max(
            nu * wallGradientPinned(favreU(jHi), favreU(jHi - 1), dTop1, dTop2),
            0.0));
        if (!rans) return w;
        // mu_t of THIS state (the stage buffer holds the last stage's input).
        dev->updateMut(*ds);
        if (channelPlaneScalarSums(gg, dev->mut(), mutScratch, pmut) !=
                hipSuccess ||
            channelPlaneScalarSums(gg, ds->t[0], ntScratch, pnt) != hipSuccess ||
            (sst && channelPlaneScalarSums(gg, ds->t[1], nt2Scratch, pnt2) !=
                        hipSuccess))
            return fail(6, "RANS plane sums download failed");
        for (int j = jLo; j <= jHi; ++j)
        {
            w.mutMax = std::max(w.mutMax, pmut[j * 2 + 1] / r.mu);
            w.turb[0] += dy[j] * pnt[j * 2];
            if (sst) w.turb[1] += dy[j] * pnt2[j * 2];
        }
        return w;
    };

    auto emitProfiles = [&](double t, const WallState& w) {
        const double ut = std::max(0.5 * (w.utauBot + w.utauTop), 1e-12);
        const double nxz = double(g.nx) * g.nz;
        for (int j = jLo; j <= jHi; ++j)
        {
            const double rhoM = ps[j * 8 + 0] / nxz;
            const double uF = ps[j * 8 + 1] / ps[j * 8 + 0];
            const double vF = ps[j * 8 + 2] / ps[j * 8 + 0];
            const double uu = ps[j * 8 + 3] / ps[j * 8 + 0] - uF * uF;
            const double vv = ps[j * 8 + 4] / ps[j * 8 + 0] - vF * vF;
            const double ww = ps[j * 8 + 5] / ps[j * 8 + 0];
            const double uv = ps[j * 8 + 6] / ps[j * 8 + 0] - uF * vF;
            const double TM = ps[j * 8 + 7] / ps[j * 8 + 0];
            const double dWall = std::min(yc[j], 2 * r.delta - yc[j]);
            std::vector<double> row = {t,
                                       yc[j],
                                       dWall * ut / nu,
                                       uF / ut,
                                       std::sqrt(std::max(uu, 0.0)) / ut,
                                       std::sqrt(std::max(vv, 0.0)) / ut,
                                       std::sqrt(std::max(ww, 0.0)) / ut,
                                       uv / (ut * ut),
                                       rhoM,
                                       TM};
            // Plane means of mu_t/mu and nu~/nu = (rho nu~)/mu (constant mu);
            // SST: Favre k+ = k/u_tau^2 and omega+ = omega nu/u_tau^2.
            if (rans) row.push_back(pmut[j * 2] / nxz / r.mu);
            if (rans && !sst) row.push_back(pnt[j * 2] / nxz / r.mu);
            if (sst)
                row.insert(row.end(),
                           {pnt[j * 2] / ps[j * 8 + 0] / (ut * ut),
                            pnt2[j * 2] / ps[j * 8 + 0] * nu / (ut * ut)});
            out.row("profiles", row);
        }
    };

    RunStats st;
    double t = 0.0;
    internal::StatsCadence cad{r.statsEvery};
    double dtStab = 0.0;
    int sinceDt = 1 << 30;
    internal::ProfileSchedule prof(r.profileDt);
    const auto wall0 = std::chrono::steady_clock::now();
    const std::string ransLabel = rans ? "RANS " + std::string(ransName(r.rans))
                                       : std::string();

    if (steady)
    {
        // Steady RANS in local-time-stepping PSEUDO-time: every cell marches
        // at its own stable dt (the lts ratios on top of dtRef). It stops on
        // the integrals (ub, bulk T, the turbulence scalars) extrapolated to
        // their fixed point, not on the residual: the FP32 residual floors
        // above the slow mode's (docs/rans.md, "Stopping on the integrals").
        // t accumulates dtRef and is NOT a physical time.
        dev->setLts(true);
        bool converged = false;
        double resid = HUGE_VAL, remaining = HUGE_VAL;
        SteadyStop stop(sst ? 4 : 3);
        // Residual in wall units: L = delta = 1, U = u_tau = 1, and the
        // energy scaled by the bulk estimate (docs/rans.md, D11).
        auto statsRow = [&](double dt) -> std::expected<void, RunError> {
            const ResidualNorms n = dev->residual(*ds, jLo, jHi, *red);
            auto w = measure();
            if (!w) return std::unexpected(w.error());
            if (auto h = internal::checkDevice("a stats row", &*red); !h)
                return std::unexpected(h.error());
            // SST: k and ln(omega) rates, the larger of the two.
            const double resU = n.mom, resE = n.E / ubEst,
                         resT = sst ? std::max(n.t[0], n.t[1]) : n.t[0];
            resid = std::max({resU, resE, resT});
            if (!std::isfinite(resid))
                return fail(7, "the residual is not finite (the state "
                               "diverged)");
            // Bulk T, not T - Twall: the excess is a 0.6% difference of two
            // FP32-resolved numbers, its relative noise is 5e-6, not 3e-8.
            const double tBulk = r.Twall + w->tExcess;
            // Equally spaced samples only: the cap row is off the cadence.
            if (st.steps % cad.every == 0)
            {
                const double q[4] = {w->ubulk, tBulk, w->turb[0], w->turb[1]};
                stop.push(std::span(q, sst ? 4 : 3));
                remaining = stop.remaining(r.steadyTol);
            }
            const double retau = 0.5 * (w->utauBot + w->utauTop) * r.delta / nu;
            std::vector<double> row = {t, dt, w->ubulk, w->utauBot, w->utauTop,
                                       retau, w->ke, w->mutMax};
            if (sst) row.push_back(double(dev->floorHits()));
            row.insert(row.end(), {double(st.steps), resid, resU, resE, resT,
                                   tBulk, remaining});
            out.row("stats", row);
            converged = remaining < r.steadyTol;
            if (converged || st.steps >= r.steadySteps ||
                st.steps - cad.lastPrint >= 10 * cad.every)
            {
                std::printf("iter=%ld ub=%.4f Re_tau=%.2f mutMax=%.2f "
                            "dtRef=%.2e %s remaining=%.2e resid=%.3e "
                            "(U %.2e E %.2e T %.2e)\n",
                            st.steps, w->ubulk, retau, w->mutMax, dt,
                            ransLabel.c_str(), remaining, resid, resU, resE,
                            resT);
                std::fflush(stdout);
                cad.printed(st.steps);
            }
            cad.logged(st.steps);
            if (converged || st.steps >= r.steadySteps) emitProfiles(t, *w);
            return {};
        };
        while (!converged && st.steps < r.steadySteps)
        {
            if (stopRequested()) break;
            if (auto e = internal::recomputeDt(
                    sinceDt, dtStab, r.dtRecompute, r.dtMargin, 1e-9,
                    [&] { return dev->localDt(*ds, *red, r.cfl); });
                !e)
                return fail(7, e.error());
            dev->step(*ds, Real(dtStab));
            wall->apply(*ds);
            t += dtStab;
            ++st.steps;
            if (st.steps - cad.lastLog >= cad.every || st.steps >= r.steadySteps)
                if (auto e = statsRow(dtStab); !e)
                    return std::unexpected(e.error());
        }
        if (auto h = internal::checkDevice("the end of the run", &*red); !h)
            return std::unexpected(h.error());
        finishTiming(st, wall0, out);
        if (!converged && !stopRequested())
            return fail(8, std::format("steady state not reached in {} "
                                       "iterations (remaining change {:.3e} "
                                       "> {:.3g})",
                                       st.steps, remaining, r.steadyTol));
        st.converged = converged;
        return st;
    }

    while (t < r.tEnd)
    {
        // A stop signal ends the run the SAME way reaching tEnd does: return
        // normally, so every destructor runs -- the stats file is flushed and
        // the shared-memory segment is unlinked. Terminating mid-step instead
        // leaves a segment that outlives its own run, flagged as live.
        if (stopRequested()) break;
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

        const bool statsDue = cad.due(st.steps, t, r.tEnd);
        const bool profDue = prof.due(t, r.tEnd);
        if (statsDue || profDue)
        {
            auto w = measure();
            if (!w) return std::unexpected(w.error());
            if (auto h = internal::checkDevice("a stats row", &*red); !h)
                return std::unexpected(h.error());
            if (statsDue)
            {
                const double retau =
                    0.5 * (w->utauBot + w->utauTop) * r.delta / nu;
                std::vector<double> row = {t,          dt,         w->ubulk,
                                           w->utauBot, w->utauTop, retau,
                                           w->ke};
                if (rans) row.push_back(w->mutMax);
                if (sst) row.push_back(double(dev->floorHits()));
                out.row("stats", row);
                if (cad.printDue(st.steps, t, r.tEnd))
                {
                    std::printf("t=%.2f/%.2f ub=%.2f Re_tau=%.1f "
                                "(bot %.3f top %.3f) dt=%.2e%s%s\n",
                                t, r.tEnd, w->ubulk, retau, w->utauBot,
                                w->utauTop, dt, rans ? " " : "",
                                ransLabel.c_str());
                    std::fflush(stdout);
                    cad.printed(st.steps);
                }
                cad.logged(st.steps);
            }
            if (profDue)
            {
                emitProfiles(t, *w);
                prof.done(t);
            }
        }
        if (out.wantsAnyFrame(t)) writeFrame(t);
    }
    if (auto h = internal::checkDevice("the end of the run", &*red); !h)
        return std::unexpected(h.error());
    finishTiming(st, wall0, out);
    return st;
}

} // namespace dns::gpu
