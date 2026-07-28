#include "dns/interrupt.hpp"
#include "dns/gpu/runner_ogrid.hpp"

#include "dns/curvmesh.hpp"
#include "dns/gpu/comp.hpp"
#include "dns/gpu/derived.hpp"
#include "dns/gpu/device_util.hpp"
#include "dns/gpu/internal/comp_flux.hpp"
#include "dns/gpu/internal/run_loop.hpp"
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

namespace {

// Quiet-start freestream schedule: smoothstep 0 -> 1 over [0, ramp].
double rampFrac(double t, double ramp)
{
    if (ramp <= 0 || t >= ramp) return 1.0;
    const double s = t / ramp;
    return s * s * (3 - 2 * s);
}

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
    auto wall = DeviceOGridBc::make(g, bc);
    if (!wall) return fail(6, "O-grid boundary init failed");

    // Both boundary rows ride the stage hook: they carry no solution, so the
    // rhs computed on them is discarded and re-derived every stage.
    dev->setStageHook([&](DeviceCompState& s) { wall->apply(s); });

    (void)ds->upload(ic);
    const double u0 = r.ramp > 0 ? 0.0 : r.uInf;
    wall->setFreestream(u0);
    wall->apply(*ds);

    auto du = DeviceScalar::make(g.size());
    auto dv = DeviceScalar::make(g.size());
    auto dw = DeviceScalar::make(g.size());
    auto dVort = DeviceScalar::make(g.size());
    auto dSpeed = DeviceScalar::make(g.size());
    if (!du || !dv || !dw || !dVort || !dSpeed)
        return fail(6, "GPU allocation failed");

    // heat: the wall energy flux INTO the body, raw (not a coefficient) --
    // a Nusselt number needs the caller's own kappa and temperature scale.
    // wmax: the acoustic wave speed the dt bound actually saw, put back on a
    // physical length (the mesh's narrowest cell width); the index spacings of
    // a curvilinear grid are not lengths.
    const std::string cols[] = {"t",  "dt", "cd",   "cl",
                                "cs", "heat", "wmax"};
    out.declareSeries("stats", cols);

    const float vortScale =
        float(r.curlMax > 0 ? r.curlMax
                            : (r.signedVorticity ? 8.0 : 15.0) * r.uInf / r.D);
    const float speedMax = float(1.6 * r.uInf);
    const char* vortName = r.signedVorticity ? "vort" : "curl";
    const bool signedWz = r.signedVorticity && g.nz == 1;
    std::vector<Real> hVort(g.size()), hSpeed(g.size());
    std::vector<float> vortF(g.size()), speedF(g.size());
    // --temp: T - Tinf, clipped at whichever thermal excursion the case can
    // produce -- the isothermal wall's own offset, or the stagnation rise an
    // adiabatic body still generates. Same convention as runner_comp.
    const double TInf = r.pInf / (r.rhoInf * r.tr.Rgas);
    const double machSq = r.uInf * r.uInf * r.rhoInf / (r.tr.gamma * r.pInf);
    const float tempScale = float(std::max(
        r.thermal == OGridThermal::Isothermal ? std::abs(r.Twall - TInf) : 0.0,
        0.5 * (r.tr.gamma - 1.0) * machSq * TInf));
    std::vector<Real> hTemp(r.tempFrames ? g.size() : 0);
    std::vector<float> tempF(r.tempFrames ? g.size() : 0);
    auto writeFrame = [&](double t) {
        if (r.tempFrames) dev->primitives(*ds);
        primitiveVelocity(gg, ds->rho, ds->mx, ds->my, ds->mz, *du, *dv, *dw);
        kOGridCurl<<<launchBlocks(g.size()), blockSize>>>(
            gg, dev->curvTables()->view(), signedWz ? 1 : 0, du->data(),
            dv->data(), dw->data(), dVort->data());
        checkLastKernel("kOGridCurl");
        speedCentred(gg, *du, *dv, *dw, *dSpeed);
        (void)dVort->download(hVort.data());
        (void)dSpeed->download(hSpeed.data());
        const float uOff = r.speedDelta ? float(r.uInf) : 0.0f;
        for (std::size_t c = 0; c < g.size(); ++c)
        {
            vortF[c] = float(hVort[c]);
            speedF[c] = float(hSpeed[c]) - uOff;
        }
        if (r.tempFrames)
        {
            (void)dev->T().download(hTemp.data());
            for (std::size_t c = 0; c < g.size(); ++c)
                tempF[c] = float(double(hTemp[c]) - TInf);
        }
        const FrameField ff[3]{
            {vortName, vortF.data(), 0, vortScale, signedWz,
             r.vortAsinh && signedWz ? vortScale / 32.0f : 0.0f},
            r.speedDelta
                ? FrameField{"dspeed", speedF.data(), 0, float(0.6 * r.uInf),
                             true}
                : FrameField{"speed", speedF.data(), 0, speedMax, false},
            {"temp", tempF.data(), -tempScale, tempScale, true}};
        const std::span<const FrameField> fv(ff, r.tempFrames ? 3u : 2u);
        if (auto e = out.frame(t, FrameViews{fv}); !e)
            std::fprintf(stderr, "frame write failed: %s\n", e.error().c_str());
    };

    RunStats st;
    double heat = 0.0;
    double t = 0.0;
    long lastLog = 0;
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

        if (st.steps - lastLog >= r.statsEvery || t >= r.tEnd)
        {
            const auto F = wall->surface(*dev, *ds);
            st.cd = F.fx / (r.qRef * span);
            st.cl = F.fy / (r.qRef * span);
            st.cs = F.fz / (r.qRef * span);
            heat = F.heat;
            const double wmax = r.cfl / dtStab * mtr->hmin;
            const double row[] = {t, dt, st.cd, st.cl, st.cs, heat, wmax};
            (void)out.row("stats", row);
            if (st.steps - lastLog >= 10 * r.statsEvery || t >= r.tEnd)
            {
                std::printf("t=%.2f/%.2f cd=%.3f cl=%+.3f cs=%+.3f q=%+.3e "
                            "wmax=%.2f dt=%.2e\n",
                            t, r.tEnd, st.cd, st.cl, st.cs, heat, wmax, dt);
                std::fflush(stdout);
            }
            lastLog = st.steps;
        }
        if (out.wantsFrame(t) || out.wantsPng(t) || out.wantsWeb(t)
            || out.wantsShm(t))
            writeFrame(t);
    }
    (void)hipDeviceSynchronize();
    st.tStop = t;
    finishTiming(st, wall0, out);
    return st;
}

} // namespace dns::gpu
