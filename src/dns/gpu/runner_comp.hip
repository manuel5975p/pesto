#include "dns/interrupt.hpp"
#include "dns/gpu/runner_comp.hpp"

#include "dns/curvmesh.hpp"
#include "dns/gpu/comp.hpp"
#include "dns/gpu/internal/derived_remap.hpp"
#include "dns/gpu/internal/run_loop.hpp"
#include "dns/gpu/comp_far.hpp"
#include "dns/gpu/comp_ghost.hpp"
#include "dns/gpu/comp_nscbc.hpp"
#include "dns/gpu/derived.hpp"
#include "dns/gpu/device_util.hpp"
#include "dns/gpu/reductions.hpp"

#include <hip/hip_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <optional>
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

} // namespace

std::expected<RunStats, RunError>
runCompressible(const Grid& g, const CompState& ic, const Sdf& sdf,
                const std::array<double, 6>* bbox, const CompRun& r,
                RunOutput& out)
{
    const auto gg = GpuGrid::from(g);
    const bool nscbc = r.far == FarField::Nscbc;
    const bool stretched = r.stretchBeta > 0;
    const bool focal = r.focalActive;

    // The PointRemaps of the two curvilinear meshes; both are the exact
    // identity when their parameters are off, so building them unconditionally
    // costs nothing. NSCBC and the derived frames evaluate the active one on
    // device (kernel template argument) so differences use physical, not
    // reference, distances.
    const TanhStretchY remap{double(g.ny) * g.hy, r.stretchBeta};
    const double axisL[3] = {double(g.nx) * g.hx, double(g.ny) * g.hy,
                             double(g.nz) * g.hz};
    AxisRemap3 fremap;
    {
        Focal1D* f[3] = {&fremap.fx, &fremap.fy, &fremap.fz};
        for (int a = 0; a < 3; ++a)
            *f[a] = r.focalActive && r.focalHi[a] > r.focalLo[a]
                        ? Focal1D::make(axisL[a], r.focalLo[a], r.focalHi[a],
                                        r.focalW[a], r.focalRatio)
                        : Focal1D::identity(axisL[a]);
    }

    // Curvilinear mesh: nodes from the PointRemap, finite-volume tables
    // derived from the nodes (discrete face closure), solver on the
    // curvilinear path. LES rides along on both paths (the SGS operators read
    // the mesh metrics); the validator guarantees at most one of the two
    // meshes is active.
    std::optional<CurvMetrics> mtr;
    if (stretched)
    {
        auto m = CurvMetrics::build(CurvMesh::stretchedY(g, r.stretchBeta));
        if (!m) return fail(2, "stretched mesh: " + m.error());
        mtr = std::move(*m);
    }
    else if (focal)
    {
        auto m = CurvMetrics::build(CurvMesh::fromRemap(g, fremap));
        if (!m) return fail(2, "focal mesh: " + m.error());
        mtr = std::move(*m);
    }
    auto dev = mtr ? DeviceComp::make(*mtr, r.tr, r.les, r.sgs)
                   : DeviceComp::make(g, r.tr, r.les, r.sgs);
    auto ds = DeviceCompState::make(g.size());
    auto red = ReduceScratch::make();
    if (!dev || !ds || !red) return fail(6, "GPU allocation failed");

    // --penalize: device lambda for the band decay (DeviceCutGeometry::make
    // reuses the standard upload; the aperture buffers ride along unused).
    std::optional<DeviceCutGeometry> dPen;
    if (r.penalize > 0)
    {
        auto p = DeviceCutGeometry::make(*r.cutGeom);
        if (!p) return fail(6, "GPU allocation failed");
        dPen = std::move(*p);
    }

    DeviceCompFar::Config fc;
    fc.rhoInf = r.rhoInf;
    fc.uInf = r.uInf;
    fc.pInf = r.pInf;
    fc.gamma = r.gamma;
    // Resolved by validate(); not re-derived here (see runner_incomp).
    if (r.spongeCells <= 0)
        return fail(6, "far-field band width was not resolved");
    fc.ns = r.spongeCells;
    fc.openY = r.open;
    fc.openZ = r.open && g.nz > 1;
    fc.noslipY = r.noslipY;
    if (nscbc)
    {   // x is handled characteristically; an optional thin strip inside the
        // NSCBC rows absorbs the residual reflection that pumps box modes above
        // Re 800 (pure NSCBC diverges at Re2000 / 2048^2).
        fc.sigMax = r.nscbcStrip > 0 ? 0.5 : 0.0;
        if (r.nscbcStrip > 0) fc.ns = r.nscbcStrip;
    }
    auto far = DeviceCompFar::make(g, fc);
    if (!far) return fail(6, "far-field init failed");

    std::optional<DeviceCompNscbc> nsc;
    if (nscbc)
    {
        DeviceCompNscbc::Config nc;
        nc.rhoInf = r.rhoInf;
        nc.uInf = r.uInf;
        nc.pInf = r.pInf;
        nc.gamma = r.gamma;
        nc.betaTrans = r.nscbcBeta;
        nc.openY = r.open;
        nc.openZ = r.open && g.nz > 1;
        nc.potCorrect = r.nscbcPot;
        nc.xc = r.cx;
        nc.yc = r.cy;
        nc.zc = r.cz;
        nc.aBody = 0.5 * r.D;
        nc.floorRows = r.floorRows;
        nc.thermalInflow = r.nscbcThermalInflow;
        nc.Rgas = r.tr.Rgas;
        nc.TInf = r.pInf / (r.rhoInf * r.tr.Rgas);
        auto m = DeviceCompNscbc::make(g, nc);
        if (!m) return fail(6, "NSCBC init failed");
        nsc = std::move(*m);
    }

    // The ghost wall classifies against the physical cell centres of whichever
    // mesh is active; the focal remap is the exact identity when off, so the
    // Cartesian tables are unchanged.
    const GhostWallThermal wallBc{
        .mode = r.wallIsothermal ? GhostWallThermal::Mode::Isothermal
                                 : GhostWallThermal::Mode::Adiabatic,
        .Twall = r.Twall,
        .Rgas = r.tr.Rgas};
    auto ghost = DeviceCompGhost::make(g, sdf, r.gamma, r.rhoInf, r.pInf,
                                       bbox ? bbox->data() : nullptr, fremap,
                                       wallBc);
    if (!ghost) return fail(6, "ghost wall init failed");

    // --penalize: the band decay rides the stage hook (runs once per RK
    // stage, so each application carries dt/3; penDt is set per step below).
    // It precedes the ghost fill so the mirrors sample the penalized field.
    double penDt = 0;
    auto applyBc = [&](DeviceCompState& s) {
        if (dPen && penDt > 0)
            penalizeMomentum(GpuGrid::from(g), dPen->lambda, Real(r.penalize),
                             Real(penDt), s);
        far->apply(s);
        ghost->apply(s);
    };
    dev->setStageHook(applyBc);
    if (nsc)
        dev->setRhsHook([&](const DeviceCompState& s, DeviceCompState& out,
                            const StageCombine& cmb) {
            if (stretched) nsc->apply(*dev, s, out, cmb, remap);
            else if (focal) nsc->apply(*dev, s, out, cmb, fremap);
            else nsc->apply(*dev, s, out, cmb);
        });

    (void)ds->upload(ic);
    const double u0 = r.ramp > 0 ? 0.0 : r.uInf;
    far->setFreestream(u0);
    if (nsc) nsc->setFreestream(u0);
    applyBc(*ds);

    auto du = DeviceScalar::make(g.size());
    auto dv = DeviceScalar::make(g.size());
    auto dw = DeviceScalar::make(g.size());
    auto dVort = DeviceScalar::make(g.size());
    auto dSpeed = DeviceScalar::make(g.size());
    if (!du || !dv || !dw || !dVort || !dSpeed)
        return fail(6, "GPU allocation failed");

    // qw is the wall heat flux (an energy RATE into the body, comp_ghost.hpp);
    // it is identically zero on the default adiabatic wall.
    const std::string cols[] = {"t", "dt", "cd", "cl", "cs", "wmax", "qw"};
    out.declareSeries("stats", cols);

    const float vortScale =
        float(r.curlMax > 0 ? r.curlMax
                            : (r.signedVorticity ? 8.0 : 15.0) * r.uInf / r.D);
    const float speedMax = float(1.6 * r.uInf);
    const char* vortName = r.signedVorticity ? "vort" : "curl";
    std::vector<Real> hVort(g.size()), hSpeed(g.size());
    std::vector<float> vortF(g.size()), speedF(g.size());
    // --temp: T - Tinf on the diverging map, so the freestream is the neutral
    // background exactly as --speed-map delta does for velocity. The clip is
    // whichever thermal excursion the case can actually produce -- an
    // isothermal wall's own offset, or the stagnation rise (gamma-1)/2 M^2
    // Tinf that an adiabatic body still generates.
    const double TInf = r.pInf / (r.rhoInf * r.tr.Rgas);
    const double machSq = r.uInf * r.uInf * r.rhoInf / (r.tr.gamma * r.pInf);
    const float tempScale = float(std::max(
        r.wallIsothermal ? std::abs(r.Twall - TInf) : 0.0,
        0.5 * (r.tr.gamma - 1.0) * machSq * TInf));
    std::vector<Real> hTemp(r.tempFrames ? g.size() : 0);
    std::vector<float> tempF(r.tempFrames ? g.size() : 0);
    // Display raster: on a clustered mesh the frame arrays are resampled onto
    // the uniform physical grid RunOutput was built with (dns/framemap.hpp).
    const bool resample = r.raster && r.raster->active();
    const std::size_t rn = resample ? r.raster->renderGrid().size() : 0;
    std::vector<float> vortR(rn), speedR(rn);
    std::vector<float> tempR(r.tempFrames ? rn : 0);
    auto writeFrame = [&](double t) {
        if (r.tempFrames) dev->primitives(*ds);
        primitiveVelocity(gg, ds->rho, ds->mx, ds->my, ds->mz, *du, *dv, *dw);
        if (r.signedVorticity && stretched)
            vorticityZRemap(gg, remap, *du, *dv, *dVort);
        else if (r.signedVorticity && focal)
            vorticityZRemap(gg, fremap, *du, *dv, *dVort);
        else if (r.signedVorticity) vorticityZCentred(gg, *du, *dv, *dVort);
        else curlMagCentred(gg, *du, *dv, *dw, *dVort);
        speedCentred(gg, *du, *dv, *dw, *dSpeed);
        (void)dVort->download(hVort.data());
        (void)dSpeed->download(hSpeed.data());
        const float uOff = r.speedDelta ? float(r.uInf) : 0.0f;
        for (std::size_t c = 0; c < g.size(); ++c)
        {
            vortF[c] = float(hVort[c]);
            speedF[c] = float(hSpeed[c]) - uOff;
        }
        const float* vortOut = vortF.data();
        const float* speedOut = speedF.data();
        const float* tempOut = nullptr;
        if (r.tempFrames)
        {
            (void)dev->T().download(hTemp.data());
            for (std::size_t c = 0; c < g.size(); ++c)
                tempF[c] = float(double(hTemp[c]) - TInf);
            tempOut = tempF.data();
        }
        if (resample)
        {
            r.raster->apply(vortF.data(), vortR.data());
            r.raster->apply(speedF.data(), speedR.data());
            vortOut = vortR.data();
            speedOut = speedR.data();
            if (r.tempFrames)
            {
                r.raster->apply(tempF.data(), tempR.data());
                tempOut = tempR.data();
            }
        }
        const FrameField ff[3]{
            {vortName, vortOut, 0, vortScale, r.signedVorticity,
             r.vortAsinh && r.signedVorticity ? vortScale / 32.0f : 0.0f},
            r.speedDelta
                ? FrameField{"dspeed", speedOut, 0, float(0.6 * r.uInf), true}
                : FrameField{"speed", speedOut, 0, speedMax, false},
            {"temp", tempOut, -tempScale, tempScale, true}};
        const std::span<const FrameField> fv(ff, r.tempFrames ? 3u : 2u);
        if (auto e = out.frame(t, FrameViews{fv}); !e)
            std::fprintf(stderr, "frame write failed: %s\n", e.error().c_str());
    };

    RunStats st;
    double t = 0.0;
    long lastLog = 0;
    double dtStab = 0.0;
    int sinceDt = 1 << 30;
    // Potential-correction wake source: EMA of F_drag/(rhoInf uInf) over
    // ~2 D/uInf so the shedding oscillation does not feed back into the
    // far-field target. Fed only past the quiet-start ramp (the ramp's
    // inertial force is not drag).
    double qEma = 0.0, tQ = r.ramp;
    const auto wall0 = std::chrono::steady_clock::now();

    while (t < r.tEnd)
    {
        // A stop signal ends the run the SAME way reaching tEnd does: return
        // normally, so every destructor runs -- the stats file is flushed and
        // the shared-memory segment is unlinked. Terminating mid-step instead
        // leaves a segment that outlives its own run, flagged as live.
        if (stopRequested()) break;
        // 1e-6: the compressible collapse threshold. NOT shared with the
        // channel's 1e-9 -- the dt scales differ.
        if (auto e = internal::recomputeDt(
                sinceDt, dtStab, r.dtRecompute, r.dtMargin, 1e-6,
                [&] { return dev->stableDt(*ds, *red, r.cfl); });
            !e)
            return fail(7, e.error());
        const double dt = std::min(r.tEnd - t, dtStab);

        penDt = dt / 3.0;
        dev->step(*ds, Real(dt));
        if (t < r.ramp)
        {   // Quiet start: boost the frame, then retarget the far field.
            const double uNew = r.uInf * rampFrac(t + dt, r.ramp);
            if (nsc)
            {
                boostX(gg, *ds, uNew - nsc->freestream());
                nsc->setFreestream(uNew);
                far->setFreestream(uNew);
            }
            else
            {
                far->boost(*ds, uNew - far->freestream());
                far->setFreestream(uNew);
            }
        }
        applyBc(*ds);   // leave the state BC/ghost-consistent between steps
        t += dt;
        ++st.steps;

        if (st.steps - lastLog >= r.statsEvery || t >= r.tEnd)
        {
            const auto F = ghost->force(*dev, *ds);
            const double span = g.nz > 1 ? g.nz * g.hz : 1.0;
            st.cd = F.fx / (r.qRef * span);
            st.cl = F.fy / (r.qRef * span);
            st.cs = F.fz / (r.qRef * span);
            if (nsc && r.nscbcPot && t > r.ramp)
            {
                const double Qinst = std::clamp(F.fx / (r.rhoInf * r.uInf),
                                                0.0, 2.0 * r.D * r.uInf * span);
                const double alpha =
                    std::min(1.0, (t - tQ) / (2.0 * r.D / r.uInf));
                qEma += alpha * (Qinst - qEma);
                tQ = t;
                nsc->setWakeSource(qEma);
            }
            const double wmax =
                r.cfl / dtStab * std::min({g.hx, g.hy, g.hz});
            const double row[] = {t, dt, st.cd, st.cl, st.cs, wmax, F.heat};
            (void)out.row("stats", row);
            if (st.steps - lastLog >= 10 * r.statsEvery || t >= r.tEnd)
            {
                std::printf("t=%.2f/%.2f cd=%.3f cl=%+.3f cs=%+.3f wmax=%.2f "
                            "dt=%.2e\n",
                            t, r.tEnd, st.cd, st.cl, st.cs, wmax, dt);
                std::fflush(stdout);
            }
            lastLog = st.steps;
        }
        // The output schedule is independent of the quiet-start ramp: frames
        // during spin-up show the (physical, ramped) transient. The ramp gate
        // that used to suppress them hid the first `ramp` time units of every
        // run for no numerical reason -- the sinks pace themselves via
        // fdt/pdt/wdt either way.
        if (out.wantsFrame(t) || out.wantsPng(t) || out.wantsWeb(t)
            || out.wantsShm(t))
            writeFrame(t);
    }
    (void)hipDeviceSynchronize();
    finishTiming(st, wall0, out);
    return st;
}

} // namespace dns::gpu
