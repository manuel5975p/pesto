#include "dns/gpu/runner_incomp.hpp"

#include "dns/interrupt.hpp"

#include "dns/gpu/cutflux.hpp"
#include "dns/gpu/cutpoisson.hpp"
#include "dns/gpu/derived.hpp"
#include "dns/gpu/device_util.hpp"
#include "dns/gpu/ghostfill.hpp"
#include "dns/gpu/internal/frame_scratch.hpp"
#include "dns/gpu/internal/run_loop.hpp"
#include "dns/gpu/kernels.hpp"
#include "dns/gpu/les.hpp"
#include "dns/gpu/reductions.hpp"
#include "dns/gpu/tunnel.hpp"

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
using internal::rampFrac;

std::expected<RunStats, RunError>
runIncompressible(const Grid& g, const CutGeometry& cut, const Sdf& sdf,
                  const VectorField& ic, const IncompRun& r, RunOutput& out,
                  VectorField* finalField)
{
    const auto gg = GpuGrid::from(g);
    const double h = g.hx;

    auto dCut = DeviceCutGeometry::make(cut);
    // --cut-flux: the dispatch mask for the apertured momentum kernel. Built
    // only when requested; flag-off runs never touch the cutflux TU.
    std::optional<DeviceCutFlag> dFlag;
    if (r.cutFlux)
    {
        auto hostFlag = buildCutFlag(g, cut);
        clearBoundaryFlag(g, hostFlag, 4); // tunnel/fringe rows own the ring
        auto f = DeviceCutFlag::make(hostFlag);
        if (!f) return fail(6, "GPU allocation failed");
        dFlag = std::move(*f);
    }
    // Sharp viscous wall: the diffusion term reads a ghost-filled copy of the
    // stage field (mirror values, no-slip at the true sub-cell surface) while
    // convection keeps the masked zeros. Mirror values in the skew-symmetric
    // convective form INJECT energy -- never feed dG to convection.
    auto dGhost = DeviceGhostFill::make(g, cut, sdf);
    auto dU = DeviceField::make(g.size());
    auto dU0 = DeviceField::make(g.size());
    auto dS = DeviceField::make(g.size());
    auto dG = DeviceField::make(g.size());
    auto red = ReduceScratch::make();
    auto scr = DeviceCutProjectionScratch::make(g.size());
    if (!dCut || !dGhost || !dU || !dU0 || !dS || !dG || !red || !scr)
        return fail(6, "GPU allocation failed");

    // Steady mode only: the field as of the previous stats row. Differencing
    // adjacent steps loses the signal to FP32 cancellation.
    std::optional<DeviceField> dSnap;
    if (r.steadyTol > 0)
    {
        auto s = DeviceField::make(g.size());
        if (!s) return fail(6, "GPU allocation failed");
        dSnap = std::move(*s);
    }
    // Outlet flux-correction staging; the fringe far field never asks for it.
    std::optional<DeviceScalar> dTmp;
    if (!r.fringe)
    {
        auto s = DeviceScalar::make(g.size());
        if (!s) return fail(6, "GPU allocation failed");
        dTmp = std::move(*s);
    }

    std::optional<DeviceSmagorinsky> les;
    if (r.les > 0)
    {
        auto m = DeviceSmagorinsky::make(gg, r.les, r.sgs);
        if (!m) return fail(6, "LES model init failed");
        les = std::move(*m);
    }

    auto poisson = DeviceCutPoissonSolver::make(
        gg, *dCut, r.maxIters, r.tol,
        r.fringe ? PoissonBc::Periodic : PoissonBc::NeumannBox);
    if (!poisson) return fail(6, "pressure solver init failed");

    auto bodyForce = DeviceBodyForce::make(gg, r.box);
    if (!bodyForce) return fail(6, "body-force tally init failed");

    // Per-object tallies (scenes): the same surface integral, restricted to
    // each object's own index box.
    std::vector<DeviceBodyForce> objForce;
    objForce.reserve(r.tallies.size());
    for (const auto& tb : r.tallies)
    {
        auto f = DeviceBodyForce::make(gg, tb.box);
        if (!f) return fail(6, "per-object force tally init failed");
        objForce.push_back(std::move(*f));
    }

    auto tunnel = DeviceTunnelBc::make(gg, {r.uInf, r.open, r.pot, r.cx, r.cy,
                                            r.cz, 0.5 * r.D, r.floorRows,
                                            r.floorFrac});
    // Both come RESOLVED from validate(); the runner does not second-guess
    // them. (The fallbacks that used to live here disagreed with the
    // validator's rule for a thin 3D box -- nz > 1 but too thin to hold a z
    // band -- and were only dead because dns_main always passes the resolved
    // values through.)
    if (r.fringeCells <= 0 || (r.fringeAxes != 2 && r.fringeAxes != 3))
        return fail(6, "far-field band width/axes were not resolved");
    auto fringe =
        DeviceFringe::make(gg, {r.uInf, 0.5, r.fringeCells, r.fringeAxes});
    if (!tunnel || !fringe) return fail(6, "far-field init failed");

    // Quiet start: the inlet/fringe target starts at rest and is
    // smoothstepped up to uInf over [0, ramp] each step below. uF is the
    // CURRENT freestream; the outlet convection speed follows it.
    double uF = r.ramp > 0 ? 0.0 : r.uInf;
    if (r.ramp > 0)
    {
        tunnel->setFreestream(0.0);
        fringe->setFreestream(0.0);
    }

    if (uploadField(ic, *dU) != hipSuccess)
        return fail(6, "initial condition upload failed");
    maskSolidFaces(gg, *dCut, *dU);
    (void)hipMemset(scr->phi.data(), 0, g.size() * sizeof(Real)); // warm start
    if (!r.fringe) tunnel->apply(*dU, *dU, 0.0); // seed ghosts/prescribed faces
    if (auto h = internal::checkDevice("the initial condition"); !h)
        return std::unexpected(h.error());

    // One SSP-RK3 stage. The order below carries physics -- see the masterplan
    // keep-list; in particular the ghost view feeds ONLY the viscous term, and
    // the outlet flux correction must precede the divergence so the projection
    // RHS is compatible.
    int pcgIters = 0;
    // Solves that missed the target since the last stats row (warned there),
    // and a residual that went non-finite (the field diverged: exit 7).
    int pcgUnconverged = 0, pcgSolves = 0;
    bool pcgDiverged = false;
    auto stage = [&](const DeviceField& in, DeviceField& outF, double a, double dt) {
        dGhost->fillFrom(in, *dG);
        // The solid-face mask rides in the stage kernel's tail instead of
        // costing a second full read-modify-write pass over the three arrays
        // the stage has just written -- and so does the Brinkman decay, WHEN
        // nothing runs between them. With LES it cannot: the eddy-diffusion
        // add sits between the RHS and the penalization, so both stay
        // separate passes there and the folded mask ahead of them is a no-op
        // (the trailing mask re-zeroes the same faces).
        const StageTail tail{
            .lam = (r.penalize > 0 && !les) ? dCut->lambda.data() : nullptr,
            .idte = r.penalize > 0 ? Real((1.0 - a) * dt) / Real(r.penalize)
                                   : Real(0),
            .ax = dCut->ax.data(),
            .ay = dCut->ay.data(),
            .az = dCut->az.data()};
        if (dFlag)
            fusedRhsStageCut(gg, Real(r.nu), Real(0), 0, 0, 0, Real(a), *dU0,
                             Real(1.0 - a), in, *dG, *dCut, *dFlag, Real(dt),
                             outF, tail);
        else
            fusedRhsStage(gg, Real(r.nu), Real(0), 0, 0, 0, Real(a), *dU0,
                          Real(1.0 - a), in, *dG, Real(dt), outF, tail);
        if (les)
        {
            les->computeNut(*dG, *dCut);
            les->addEddyDiffusion(*dG, Real((1.0 - a) * dt), outF);
            if (r.penalize > 0)
                penalizeSolidBand(gg, *dCut, Real(r.penalize),
                                  Real((1.0 - a) * dt), outF);
            maskSolidFaces(gg, *dCut, outF);
        }
        if (r.fringe) fringe->apply(outF);
        else
        {
            tunnel->apply(in, outF, uF * dt / h);
            tunnel->correctOutletFlux(outF, *red, *dTmp);
        }
        divergenceCut(gg, *dCut, outF, scr->div, Real(1.0 / dt));
        if (!r.fringe) tunnel->addFluxRhs(outF, 1.0 / dt, scr->div);
        const PcgResult pr = poisson->solve(scr->div, scr->phi, true);
        pcgIters += pr.iters;
        pcgUnconverged += pr.converged ? 0 : 1;
        ++pcgSolves;
        pcgDiverged = pcgDiverged || !pr.finite;
        subtractGradientCut(gg, *dCut, scr->phi, Real(dt), outF);
        if (!r.fringe) // refresh the mirrors; keep the projected lateral fluxes
            tunnel->apply(outF, outF, 0.0, /*refreshOnly=*/true);
    };

    const std::string cols[] = {"t",   "dt",  "K",   "cd",      "cl",
                                "cdf", "cfl", "maxDiv", "pcgIters", "resid"};
    if (auto e = out.declareSeries("stats", cols); !e)
        return fail(4, e.error());
    if (!objForce.empty())
    {
        std::vector<std::string> fcols{"t"};
        for (const auto& tb : r.tallies)
            for (const char* q : {"cd_", "cl_", "cdf_"})
                fcols.push_back(q + tb.name);
        if (auto e = out.declareSeries("forces", fcols); !e)
            return fail(4, e.error());
    }

    const float vortScale =
        float(r.vortMax > 0 ? r.vortMax : 8.0 * r.uInf / r.D);
    const float speedMax = float(1.6 * r.uInf);
    const float vortToe = r.vortAsinh ? vortScale / 32.0f : 0.0f;
    // The MAC field is differenced straight into the two derived scalars, so
    // this runner needs no cell-centred velocity trio.
    internal::FrameScratch fs;
    auto writeFrame = [&](double t) {
        if (!fs.ensure(g.size(), 2, false)) return;
        vorticityZ(gg, *dU, fs.vort());
        speedMac(gg, *dU, fs.speed());
        fs.take(0, fs.vort());
        fs.take(1, fs.speed(), r.speedDelta ? r.uInf : 0.0);
        const FrameField ff[2]{
            {"vort", fs.channel(0), 0, vortScale, true, vortToe},
            internal::speedField(fs.channel(1), r.speedDelta, r.uInf, speedMax)};
        fs.emit(out, t, ff);
    };

    RunStats st;
    double t = 0.0, dt = 1e-3;
    // Potential-corrected inlet wake source: EMA of F_drag / uInf, fed only
    // after the impulsive-start transient.
    internal::WakeEma wake{.tLast = r.ramp + 2.0 * r.D / r.uInf};
    // Steady-residual snapshot interval, in PHYSICAL time -- not in steps. The
    // metric differences two FP32 fields, so its noise floor is ~eps*u/interval;
    // a fixed step count shrinks the interval with dt and pushes the floor UP on
    // finer grids (measured: 1.2e-6 at 256^2 would become ~1.8e-5 at 2048^2,
    // above any useful tolerance). A fixed time span keeps the floor grid-free.
    const double residDt = 0.5 * r.D / r.uInf;
    double tSnap = 0.0;   // time of the steady-residual snapshot; 0 = none yet
    bool residValid = false;  // a residual needs TWO snapshots; the first tally
                              // only takes one, and 0.0 must never read as
                              // "converged" (it did: stopped at step 1)
    // The stable-dt bound and the Courant number it came from, carried across
    // the steps that reuse them (--dt-recompute > 1). `conv` also feeds the
    // reported cfl column, which is then the Courant number of the step the
    // bound was measured on -- the same one that sets dt.
    double dtStab = 0.0, conv = 0.0;
    int sinceDt = 1 << 30;
    const auto wall0 = std::chrono::steady_clock::now();
    while (t < r.tEnd)
    {
        // A stop signal ends the run the SAME way reaching tEnd does: return
        // normally, so every destructor runs -- the stats file is flushed and
        // the shared-memory segment is unlinked. Terminating mid-step instead
        // leaves a segment that outlives its own run, flagged as live.
        if (stopRequested()) break;
        // 1e-4: the INCOMPRESSIBLE collapse threshold, and it is not the
        // compressible 1e-6 or the channel's 1e-9 -- the dt scales differ.
        // Step 0 is exempt (the initial condition has not been projected into
        // a stable state yet), as it always was.
        if (auto e = internal::recomputeDt(
                sinceDt, dtStab, r.dtRecompute, r.dtMargin,
                st.steps > 0 ? 1e-4 : 0.0,
                [&] {
                    conv = dFlag ? maxCourantPerDtCut(gg, *dU, dCut->ax, dCut->ay,
                                                      dCut->az, dCut->lambda, *red)
                                 : maxCourantPerDt(gg, *dU, *red);
                    const double diff = r.nu > 0 ? 6.0 * r.nu / (h * h) : 0.0;
                    return r.cfl / std::max(conv, diff);
                });
            !e)
        {
            // The CFL limiter (not the end-of-run clamp) collapsed dt: the
            // field has diverged. Dump the argmax cells and stop with exit 7.
            auto host = VectorField::zeros(g);
            (void)downloadField(*dU, host);
            reportBlowup(g, host, cut.lambda, t);
            return fail(7, "the velocity field diverged (" + e.error() + ")");
        }
        dt = std::min(r.tEnd - t, dtStab);

        if (r.ramp > 0 && t < r.ramp)
        {
            uF = r.uInf * rampFrac(t + dt, r.ramp);
            tunnel->setFreestream(uF);
            fringe->setFreestream(uF);
        }
        std::swap(*dU0, *dU);
        pcgIters = 0;
        stage(*dU0, *dU, 0.0, dt);

        // The end-of-run clamp can leave a sliver of a step. The surface
        // pressure is phi/dt, so a sliver dt turns the force tally into
        // garbage (measured: cd ~1e6 in the last row). Never tally on it.
        const bool clamped = dt < 0.5 * dtStab;
        const bool tally = st.steps % r.statsEvery == 0 && !clamped;
        if (tally)
        {
            // Surface pressure from the FIRST stage's phi: that stage advances a
            // full dt, so phi is the static pressure. Later stages have shorter
            // effective substeps and would bias the force low. Friction is read
            // from the time-t field through the ghost view the viscous term saw.
            (void)dGhost->fillFrom(*dU0, *dG);
            const auto co = bodyForce->measure(*dCut, scr->phi, *dG, r.nu, r.qRef,
                                               les ? &les->nut() : nullptr, *red);
            st.cd = co.cd;
            st.cl = co.cl;
            st.cs = co.cs;
            st.cdf = co.cdf;
            if (!objForce.empty())
            {
                // Same phi / ghost view as the total: the per-object rows are
                // a decomposition of it, not a second measurement scheme.
                std::vector<double> frow{t};
                for (auto& f : objForce)
                {
                    const auto oc = f.measure(*dCut, scr->phi, *dG, r.nu, r.qRef,
                                              les ? &les->nut() : nullptr, *red);
                    frow.insert(frow.end(), {oc.cd, oc.cl, oc.cdf});
                }
                out.row("forces", frow);
            }
            if (r.pot && t > wake.tLast)
            {
                const double Qinst = std::clamp(st.cd * r.qRef / r.uInf, 0.0,
                                                2.0 * r.D * r.uInf);
                tunnel->setWakeSource(wake.update(t, Qinst, r.D, r.uInf));
            }
        }
        stage(*dU, *dS, 0.75, dt);
        stage(*dS, *dU, 1.0 / 3.0, dt);
        if (pcgDiverged)
            return fail(7, "the pressure solve diverged (non-finite residual)");

        if (out.wantsAnyFrame(t)) writeFrame(t);

        if (tally)
        {
            // Steady residual: RMS change since the previous stats row, over
            // that row's elapsed time, scaled to uInf^2/D so the threshold is
            // dt- and grid-free. The interval (not one step) is what makes it
            // measurable: the per-step change reaches the FP32 rounding level
            // (~eps*u/dt) long before the flow is stationary, which floors a
            // step-to-step residual at ~1e-4 on fine grids -- measured.
            if (r.steadyTol > 0 && t + dt >= tSnap + residDt)
            {
                if (tSnap > 0)
                {
                    st.resid = rmsDiffField(*dU, *dSnap, *red) / (t + dt - tSnap)
                               * r.D / (r.uInf * r.uInf);
                    residValid = true;
                }
                if (copyField(*dSnap, *dU) != hipSuccess)
                    return fail(6, "steady snapshot copy failed");
                tSnap = t + dt;
            }
            const double K = kineticEnergy(gg, *dU, *red);
            // Continuity residual; the tunnel adds the prescribed in/out fluxes,
            // the fringe box is plain periodic divergence.
            divergenceCut(gg, *dCut, *dU, scr->div, Real(1));
            if (!r.fringe) tunnel->addFluxRhs(*dU, 1.0, scr->div);
            const double maxDiv = maxAbsScalar(scr->div, *red);
            if (auto h = internal::checkDevice("a stats row", &*red,
                                               poisson->status());
                !h)
                return std::unexpected(h.error());
            if (pcgUnconverged > 0)
                std::fprintf(stderr,
                             "dns: warning t=%.4g: %d of %d pressure solves "
                             "stopped short of --tol since the last row\n",
                             t, pcgUnconverged, pcgSolves);
            pcgUnconverged = pcgSolves = 0;
            const double row[] = {t,       dt,     K,      st.cd,
                                  st.cl,   st.cdf, dt * conv, maxDiv,
                                  double(pcgIters), st.resid};
            out.row("stats", row);
            if (st.steps % (10 * r.statsEvery) == 0)
            {
                std::printf("t=%.2f/%.2f cd=%.3f (cdf=%.3f) cl=%+.3f K=%.1f "
                            "maxDiv=%.2e pcg=%d resid=%.2e\n",
                            t, r.tEnd, st.cd, st.cdf, st.cl, K, maxDiv, pcgIters,
                            st.resid);
                std::fflush(stdout);
            }
        }
        t += dt;
        ++st.steps;

        if (r.steadyTol > 0 && tally && residValid && st.resid < r.steadyTol)
        {
            st.converged = true;
            std::printf("steady: resid=%.3e < %.3e at t=%.2f (%ld steps)\n",
                        st.resid, r.steadyTol, t, st.steps);
            break;
        }
    }
    st.tStop = t;
    if (finalField) (void)downloadField(*dU, *finalField);
    if (r.steadyTol > 0 && !st.converged)
        return fail(8, "steady state not reached by tEnd (residual "
                       + std::to_string(st.resid) + " > "
                       + std::to_string(r.steadyTol)
                       + "); raise --ft or relax --steady");
    if (auto h = internal::checkDevice("the end of the run", &*red,
                                       poisson->status());
        !h)
        return std::unexpected(h.error());
    finishTiming(st, wall0, out);
    return st;
}

} // namespace dns::gpu
