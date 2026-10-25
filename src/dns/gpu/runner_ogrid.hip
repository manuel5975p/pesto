#include "dns/gpu/runner_ogrid.hpp"

#include "dns/interrupt.hpp"

#include "dns/curvmesh.hpp"
#include "dns/gpu/comp.hpp"
#include "dns/gpu/derived.hpp"
#include "dns/gpu/device_util.hpp"
#include "dns/gpu/internal/comp_flux.hpp"
#include "dns/gpu/internal/frame_scratch.hpp"
#include "dns/gpu/internal/run_loop.hpp"
#include "dns/gpu/reductions.hpp"

#include <hip/hip_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <format>
#include <numbers>
#include <string>
#include <vector>

namespace dns::gpu {

using internal::fail;
using internal::finishTiming;
using internal::rampFrac;

namespace {

// Curl of the cell-centred velocity through the O-grid's own metrics: index
// space central differences contracted with the cell-centre inverse Jacobian
// (gpu/internal/comp_flux.hpp), so the values are TRUE curl components on a
// body-fitted mesh -- the uniform-spacing kVorticityZCentred stencil would
// report the index-space derivative and be wrong by the local radial spacing.
// At nz == 1 the k stencil wraps onto itself and every z column is exactly
// zero, so the signed wz branch is the only one with content.
//
// The two ghost rows are written as zero: their j stencil reaches across the
// annulus (row 0's lo neighbour is row ny-1) and means nothing.
__global__ void kOGridCurl(GpuGrid g, CurvView m, int signedWz, const Real* u,
                           const Real* v, const Real* w, Real* out)
{
    const std::size_t c = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (c >= g.size()) return;
    int i, j, k;
    decompose(g, c, i, j, k);
    if (j == 0 || j == g.ny - 1)
    {
        out[c] = Real(0);
        return;
    }
    const Offsets o = offsetsAt(g, i, j, k);
    const Real* vc[3] = {u, v, w};
    Real gd[3][3];
    curvVelGradCentre(m, o, c,
                      [&](std::size_t cell, int comp) { return vc[comp][cell]; },
                      gd);
    const Real wz = gd[1][0] - gd[0][1];
    if (signedWz)
    {
        out[c] = wz;
        return;
    }
    const Real wx = gd[2][1] - gd[1][2], wy = gd[0][2] - gd[2][0];
    out[c] = sqrt(wx * wx + wy * wy + wz * wz);
}

} // namespace

std::expected<RunStats, RunError>
runOGrid(const Grid& g, const OGrid& og, const CompState& ic, const OGridRun& r,
         RunOutput& out)
{
    const auto gg = GpuGrid::from(g);

    auto mtr = CurvMetrics::build(og.mesh);
    if (!mtr) return fail(2, "O-grid metrics: " + mtr.error());
    auto dev = DeviceComp::make(*mtr, r.tr, r.les, r.sgs);
    auto ds = DeviceCompState::make(g.size());
    auto red = ReduceScratch::make();
    if (!dev || !ds || !red) return fail(6, "GPU allocation failed");

    if (r.rotor.active() && !r.rotorMrf)
        return fail(2, "the O-grid has no moving-geometry rotor: spinning the "
                       "body would spin the mesh. --rotor-frame mrf is the "
                       "supported form (the validator enforces this)");
    if (r.rotor.active())
    {
        // --rotor-frame mrf on the body-fitted mesh. The real tip radius is
        // EXACT here -- the wall is a node row -- so the tip-Mach guard uses
        // it directly; omega stays quoted on the nominal D/2 like the
        // Cartesian path (one definition, config.cpp).
        const double* cs = r.rotor.shaft.data();
        const double tipR = og.rWallMax;
        const double cInf = std::sqrt(r.tr.gamma * r.pInf / r.rhoInf);
        if (auto e = internal::checkTipMach(r.rotor.omega, tipR, r.rotor.radius,
                                            cInf, std::abs(r.uInf) / cInf,
                                            "wall tip radius", "wall", "", "");
            !e)
            return std::unexpected(e.error());

        // Zone: f = 1 over the whole wall (that is what cancels the moving
        // wall's normal flux against the transport correction exactly), the
        // blend inside the annulus, and psi CONSTANT before the far rows so
        // the characteristic far field never sees the frame. rFar and the
        // nominal fluid radial spacing come off the mesh itself.
        const auto pf = og.mesh.node(0, g.ny - 1, 0);
        const double rFar = std::hypot(pf[0] - cs[0], pf[1] - cs[1]);
        const double hr = (rFar - og.rWallMax) / std::max(1, g.ny - 3);
        MrfZone z;
        z.axis = 2;
        z.omega = r.rotor.omega;
        for (int d = 0; d < 3; ++d) z.center[d] = cs[d];
        z.r1 = tipR + std::max(0.05 * r.D, 2.0 * hr);
        z.r2 = z.r1 + std::max(0.15 * r.D, 4.0 * hr);
        // The O-grid is z-extruded with a periodic span: no axial boundary
        // exists, so the axial blend is degenerate exactly as the 2D
        // Cartesian zone's (setMrfCurv requires it).
        z.a1 = 1e30;
        z.a2 = 2e30;
        // Margin 4 nominal cells: with radial clustering the OUTER cells are
        // the widest, so the nominal spacing under-counts there.
        if (z.r2 >= rFar - 4.0 * hr)
            return fail(2, std::format(
                "MRF blend (r2 {:.3g}) reaches the far rows (rFar {:.3g}): "
                "the rotating frame must vanish before the characteristic "
                "boundary. Increase --ogrid-far",
                z.r2, rFar));
        std::printf("rotor: MRF zone about the shaft: f=1 to r %.4g, 0 past "
                    "%.4g (frozen rotor -- exact for a circular section, an "
                    "APPROXIMATION otherwise)\n",
                    z.r1, z.r2);
        std::fflush(stdout);
        if (dev->setMrfCurv(z, og.mesh) != hipSuccess)
            return fail(6, "MRF zone init failed");
    }

    DeviceOGridBc::Config bc;
    bc.gamma = r.tr.gamma;
    bc.Rgas = r.tr.Rgas;
    bc.thermal = r.thermal;
    bc.Twall = r.Twall;
    bc.rhoInf = r.rhoInf;
    bc.uInf = r.uInf;
    bc.pInf = r.pInf;
    bc.sigma = r.farSigma;
    bc.band = r.farBand;
    if (r.rotor.active())
    {
        bc.omega = r.rotor.omega;
        bc.cx = r.rotor.shaft[0];
        bc.cy = r.rotor.shaft[1];
        bc.wallPos = ogridWallMidpoints(og);
        bc.wallSpan = ogridWallSpans(og);
    }
    auto wall = DeviceOGridBc::make(g, bc);
    if (!wall) return fail(6, "O-grid boundary init failed");

    // Both boundary rows ride the stage hook: they carry no solution, so the
    // rhs computed on them is discarded and re-derived every stage.
    dev->setStageHook([&](DeviceCompState& s) { wall->apply(s); });

    if (ds->upload(ic) != hipSuccess)
        return fail(6, "initial condition upload failed");
    if (auto h = internal::checkDevice("the initial condition"); !h)
        return std::unexpected(h.error());
    const double u0 = r.ramp > 0 ? 0.0 : r.uInf;
    wall->setFreestream(u0);
    wall->apply(*ds);

    // heat: the wall energy flux INTO the body, raw (not a coefficient) --
    // a Nusselt number needs the caller's own kappa and temperature scale.
    // wmax: the acoustic wave speed the dt bound actually saw, put back on a
    // physical length (the mesh's narrowest cell width); the index spacings of
    // a curvilinear grid are not lengths.
    // A rotor adds thrust, shaft torque and their propeller coefficients
    // (rho n^2 D^4 / D^5, n = rev per unit time), exactly runner_comp's
    // columns; a static run keeps the columns it has always had.
    const std::vector<std::string> cols =
        r.rotor.active()
            ? std::vector<std::string>{"t",    "dt",   "cd",     "cl",
                                       "cs",   "heat", "wmax",   "thrust",
                                       "torque", "ct", "cq"}
            : std::vector<std::string>{"t",  "dt",   "cd",  "cl",
                                       "cs", "heat", "wmax"};
    if (auto e = out.declareSeries("stats", cols); !e)
        return fail(4, e.error());
    const auto [ctDen, cqDen] = internal::propellerDen(r.rhoInf, r.rotor.omega, r.D);

    const float vortScale =
        float(r.curlMax > 0 ? r.curlMax
                            : (r.signedVorticity ? 8.0 : 15.0) * r.uInf / r.D);
    const float speedMax = float(1.6 * r.uInf);
    const char* vortName = r.signedVorticity ? "vort" : "curl";
    const bool signedWz = r.signedVorticity && g.nz == 1;
    // --temp: T - Tinf on the diverging map. Same convention as runner_comp,
    // and the same clip (internal/frame_scratch.hpp).
    const double TInf = r.pInf / (r.rhoInf * r.tr.Rgas);
    const double machSq = r.uInf * r.uInf * r.rhoInf / (r.tr.gamma * r.pInf);
    const float tempScale = internal::thermalFrameScale(
        TInf, machSq, r.tr.gamma, r.thermal == OGridThermal::Isothermal,
        r.Twall);
    internal::FrameScratch fs;
    auto writeFrame = [&](double t) {
        if (!fs.ensure(g.size(), r.tempFrames ? 3 : 2, true)) return;
        if (r.tempFrames) dev->primitives(*ds);
        fs.centreFromConserved(gg, *ds);
        kOGridCurl<<<launchBlocks(g.size()), blockSize>>>(
            gg, dev->curvTables()->view(), signedWz ? 1 : 0, fs.u().data(),
            fs.v().data(), fs.w().data(), fs.vort().data());
        checkLastKernel("kOGridCurl");
        speedCentred(gg, fs.u(), fs.v(), fs.w(), fs.speed());
        fs.take(0, fs.vort());
        fs.take(1, fs.speed(), r.speedDelta ? r.uInf : 0.0);
        if (r.tempFrames) fs.take(2, dev->T(), TInf);
        const FrameField ff[3]{
            {vortName, fs.channel(0), 0, vortScale, signedWz,
             r.vortAsinh && signedWz ? vortScale / 32.0f : 0.0f},
            internal::speedField(fs.channel(1), r.speedDelta, r.uInf, speedMax),
            {"temp", fs.channel(2), -tempScale, tempScale, true}};
        fs.emit(out, t, {ff, r.tempFrames ? 3u : 2u});
    };

    RunStats st;
    double heat = 0.0;
    double t = 0.0;
    internal::StatsCadence cad{r.statsEvery};
    double dtStab = 0.0;
    int sinceDt = 1 << 30;
    const double span = g.nz > 1 ? g.nz * g.hz : 1.0;
    const auto wall0 = std::chrono::steady_clock::now();

    while (t < r.tEnd)
    {
        // A stop signal ends the run the SAME way reaching tEnd does: return
        // normally, so every destructor runs -- the stats file is flushed and
        // the shared-memory segment is unlinked. Terminating mid-step instead
        // leaves a segment that outlives its own run, flagged as live.
        if (stopRequested()) break;
        // 1e-6: the compressible collapse threshold, shared with runner_comp
        // because this is the same physics on a different mesh. NOT the
        // channel's 1e-9 -- that one lives in wall units.
        if (auto e = internal::recomputeDt(
                sinceDt, dtStab, r.dtRecompute, r.dtMargin, 1e-6,
                [&] { return dev->stableDt(*ds, *red, r.cfl); });
            !e)
            return fail(7, e.error());
        const double dt = std::min(r.tEnd - t, dtStab);

        dev->step(*ds, Real(dt));
        if (t < r.ramp)
        {   // Quiet start: boost the frame, then retarget the far field. The
            // boost is an exact Galilean frame change -- no divergence, no
            // vorticity, no acoustic wave -- so the no-slip wall never sees an
            // impulsive start.
            const double uNew = r.uInf * rampFrac(t + dt, r.ramp);
            boostX(gg, *ds, uNew - wall->freestream());
            wall->setFreestream(uNew);
        }
        wall->apply(*ds); // leave the state boundary-consistent between steps
        t += dt;
        ++st.steps;

        if (cad.due(st.steps, t, r.tEnd))
        {
            const auto F = wall->surface(*dev, *ds);
            if (auto h = internal::checkDevice("a stats row", &*red); !h)
                return std::unexpected(h.error());
            st.cd = F.fx / (r.qRef * span);
            st.cl = F.fy / (r.qRef * span);
            st.cs = F.fz / (r.qRef * span);
            heat = F.heat;
            const double wmax = r.cfl / dtStab * mtr->hmin;
            if (r.rotor.active())
            {
                // z shaft: thrust is the force along it, torque the moment
                // about it, both signed like runner_comp's.
                const double row[] = {t,     dt,   st.cd, st.cl,  st.cs,
                                      heat,  wmax, F.fz,  F.tz,
                                      ctDen > 0 ? F.fz / ctDen : 0.0,
                                      cqDen > 0 ? F.tz / cqDen : 0.0};
                out.row("stats", row);
            }
            else
            {
                const double row[] = {t, dt, st.cd, st.cl, st.cs, heat, wmax};
                out.row("stats", row);
            }
            if (cad.printDue(st.steps, t, r.tEnd))
            {
                if (r.rotor.active())
                    std::printf("t=%.2f/%.2f torque=%+.4e cd=%.3f cl=%+.3f "
                                "wmax=%.2f dt=%.2e\n",
                                t, r.tEnd, F.tz, st.cd, st.cl, wmax, dt);
                else
                    std::printf("t=%.2f/%.2f cd=%.3f cl=%+.3f cs=%+.3f "
                                "q=%+.3e wmax=%.2f dt=%.2e\n",
                                t, r.tEnd, st.cd, st.cl, st.cs, heat, wmax,
                                dt);
                std::fflush(stdout);
                cad.printed(st.steps);
            }
            cad.logged(st.steps);
        }
        if (out.wantsAnyFrame(t)) writeFrame(t);
    }
    if (auto h = internal::checkDevice("the end of the run", &*red); !h)
        return std::unexpected(h.error());
    st.tStop = t;
    finishTiming(st, wall0, out);
    return st;
}

} // namespace dns::gpu
