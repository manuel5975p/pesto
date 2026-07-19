#include "dns/gpu/runner_incomp.hpp"

#include "dns/gpu/cutpoisson.hpp"
#include "dns/gpu/derived.hpp"
#include "dns/gpu/device_util.hpp"
#include "dns/gpu/ghostfill.hpp"
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

namespace {

std::unexpected<RunError> fail(int code, std::string m)
{
    return std::unexpected(RunError{code, std::move(m)});
}

} // namespace

std::expected<RunStats, RunError>
runIncompressible(const Grid& g, const CutGeometry& cut, const Sdf& sdf,
                  const VectorField& ic, const IncompRun& r, RunOutput& out)
{
    const auto gg = GpuGrid::from(g);
    const double h = g.hx;

    auto dCut = DeviceCutGeometry::make(cut);
    // Sharp viscous wall: the diffusion term reads a ghost-filled copy of the
    // stage field (mirror values, no-slip at the true sub-cell surface) while
    // convection keeps the masked zeros. Mirror values in the skew-symmetric
    // convective form INJECT energy -- never feed dG to convection.
    auto dGhost = DeviceGhostFill::make(g, cut, sdf);
    auto dU = DeviceField::make(g.size());
    auto dU0 = DeviceField::make(g.size());
    auto dS = DeviceField::make(g.size());
    auto dG = DeviceField::make(g.size());
    auto dTmp = DeviceScalar::make(g.size());
    auto dVort = DeviceScalar::make(g.size());
    auto dSpeed = DeviceScalar::make(g.size());
    auto red = ReduceScratch::make();
    auto scr = DeviceCutProjectionScratch::make(g.size());
    if (!dCut || !dGhost || !dU || !dU0 || !dS || !dG || !dTmp || !dVort
        || !dSpeed || !red || !scr)
        return fail(6, "GPU allocation failed");

    std::optional<DeviceSmagorinsky> les;
    if (r.les > 0)
    {
        auto m = DeviceSmagorinsky::make(gg, r.les);
        if (!m) return fail(6, "LES model init failed (--les requires --nz 1)");
        les = std::move(*m);
    }

    auto poisson = DeviceCutPoissonSolver::make(
        gg, *dCut, r.maxIters, r.tol,
        r.fringe ? PoissonBc::Periodic : PoissonBc::NeumannBox);
    if (!poisson) return fail(6, "pressure solver init failed");

    auto bodyForce = DeviceBodyForce::make(gg, r.box);
    if (!bodyForce) return fail(6, "body-force tally init failed");

    auto tunnel = DeviceTunnelBc::make(gg, {r.uInf});
    const int nsF = r.fringeCells > 0
                        ? r.fringeCells
                        : std::clamp(int(0.1 * g.nx), 8, g.nx / 2 - 1);
    const int axes = r.fringeAxes > 0 ? r.fringeAxes : (g.nz == 1 ? 2 : 3);
    auto fringe = DeviceFringe::make(gg, {r.uInf, 0.5, nsF, axes});
    if (!tunnel || !fringe) return fail(6, "far-field init failed");

    (void)uploadField(ic, *dU);
    maskSolidFaces(gg, *dCut, *dU);
    (void)hipMemset(scr->phi.data(), 0, g.size() * sizeof(Real)); // warm start
    if (!r.fringe) tunnel->apply(*dU, *dU, 0.0); // seed ghosts/prescribed faces

    // One SSP-RK3 stage. The order below carries physics -- see the masterplan
    // keep-list; in particular the ghost view feeds ONLY the viscous term, and
    // the outlet flux correction must precede the divergence so the projection
    // RHS is compatible.
    int pcgIters = 0;
    auto stage = [&](const DeviceField& in, DeviceField& outF, double a, double dt) {
        (void)dGhost->fillFrom(in, *dG);
        fusedRhsStage(gg, Real(r.nu), Real(0), 0, 0, 0, Real(a), *dU0,
                      Real(1.0 - a), in, *dG, Real(dt), outF);
        if (les)
        {
            les->computeNut(*dG, *dCut);
            les->addEddyDiffusion(*dG, Real((1.0 - a) * dt), outF);
        }
        maskSolidFaces(gg, *dCut, outF);
        if (r.fringe) fringe->apply(outF);
        else
        {
            tunnel->apply(in, outF, r.uInf * dt / h);
            tunnel->correctOutletFlux(outF, *red, *dTmp);
        }
        divergenceCut(gg, *dCut, outF, scr->div, Real(1.0 / dt));
        if (!r.fringe) tunnel->addFluxRhs(outF, 1.0 / dt, scr->div);
        pcgIters += poisson->solve(scr->div, scr->phi, true);
        subtractGradientCut(gg, *dCut, scr->phi, Real(dt), outF);
        if (!r.fringe) tunnel->apply(outF, outF, 0.0); // refresh the mirrors
    };

    const std::string cols[] = {"t",   "dt",  "K",   "cd",      "cl",
                                "cdf", "cfl", "maxDiv", "pcgIters"};
    out.declareSeries("stats", cols);

    const float vortScale = float(8.0 * r.uInf / r.D), speedMax = float(1.6 * r.uInf);
    std::vector<Real> hVort(g.size()), hSpeed(g.size());
    std::vector<float> vortF(g.size()), speedF(g.size());
    auto writeFrame = [&](double t) {
        vorticityZ(gg, *dU, *dVort);
        speedMac(gg, *dU, *dSpeed);
        (void)dVort->download(hVort.data());
        (void)dSpeed->download(hSpeed.data());
        for (std::size_t c = 0; c < g.size(); ++c)
        {
            vortF[c] = float(hVort[c]);
            speedF[c] = float(hSpeed[c]);
        }
        const FrameField ff[2]{{"vort", vortF.data(), 0, vortScale, true},
                               {"speed", speedF.data(), 0, speedMax, false}};
        if (auto e = out.frame(t, FrameViews{ff}); !e)
            std::fprintf(stderr, "frame write failed: %s\n", e.error().c_str());
    };

    RunStats st;
    double t = 0.0, dt = 1e-3;
    const auto wall0 = std::chrono::steady_clock::now();
    while (t < r.tEnd)
    {
        const double conv = maxCourantPerDt(gg, *dU, *red);
        const double diff = r.nu > 0 ? 6.0 * r.nu / (h * h) : 0.0;
        const double dtStab = r.cfl / std::max(conv, diff);
        dt = std::min(r.tEnd - t, dtStab);

        // If the CFL limiter (not the end-of-run clamp) has collapsed dt, the
        // field has diverged: dump the argmax cells and stop with exit 7.
        if (dtStab < 1e-4 && st.steps > 0)
        {
            auto host = VectorField::zeros(g);
            (void)downloadField(*dU, host);
            reportBlowup(g, host, cut.lambda, t);
            return fail(7, "the velocity field diverged (dt collapsed)");
        }

        std::swap(*dU0, *dU);
        pcgIters = 0;
        stage(*dU0, *dU, 0.0, dt);

        const bool tally = st.steps % r.statsEvery == 0;
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
        }
        stage(*dU, *dS, 0.75, dt);
        stage(*dS, *dU, 1.0 / 3.0, dt);

        if (out.wantsFrame(t) || out.wantsPng(t) || out.wantsWeb(t))
            writeFrame(t);

        if (tally)
        {
            const double K = kineticEnergy(gg, *dU, *red);
            // Continuity residual; the tunnel adds the prescribed in/out fluxes,
            // the fringe box is plain periodic divergence.
            divergenceCut(gg, *dCut, *dU, scr->div, Real(1));
            if (!r.fringe) tunnel->addFluxRhs(*dU, 1.0, scr->div);
            const double maxDiv = maxAbsScalar(scr->div, *red);
            const double row[] = {t,       dt,     K,      st.cd,
                                  st.cl,   st.cdf, dt * conv, maxDiv,
                                  double(pcgIters)};
            (void)out.row("stats", row);
            if (st.steps % (10 * r.statsEvery) == 0)
            {
                std::printf("t=%.2f/%.2f cd=%.3f (cdf=%.3f) cl=%+.3f K=%.1f "
                            "maxDiv=%.2e pcg=%d\n",
                            t, r.tEnd, st.cd, st.cdf, st.cl, K, maxDiv, pcgIters);
                std::fflush(stdout);
            }
        }
        t += dt;
        ++st.steps;
    }
    (void)hipDeviceSynchronize();
    st.msPerStep = std::chrono::duration<double, std::milli>(
                       std::chrono::steady_clock::now() - wall0)
                       .count()
                   / double(st.steps ? st.steps : 1);
    out.setMsPerStep(st.msPerStep);
    return st;
}

} // namespace dns::gpu
