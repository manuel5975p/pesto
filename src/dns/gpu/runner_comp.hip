#include "dns/gpu/runner_comp.hpp"

#include "dns/gpu/comp.hpp"
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

namespace {

std::unexpected<RunError> fail(int code, std::string m)
{
    return std::unexpected(RunError{code, std::move(m)});
}

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

    auto dev = DeviceComp::make(g, r.gamma, r.mu, r.les);
    auto ds = DeviceCompState::make(g.size());
    auto red = ReduceScratch::make();
    if (!dev || !ds || !red) return fail(6, "GPU allocation failed");

    DeviceCompFar::Config fc;
    fc.rhoInf = r.rhoInf;
    fc.uInf = r.uInf;
    fc.pInf = r.pInf;
    fc.gamma = r.gamma;
    fc.ns = r.spongeCells > 0 ? r.spongeCells
                              : std::clamp(int(0.1 * g.nx), 8, g.nx / 2 - 1);
    fc.openY = r.open;
    fc.openZ = r.open && g.nz > 1;
    if (nscbc)
    {   // x is handled characteristically; an optional thin strip inside the
        // NSCBC rows absorbs the residual reflection that pumps box modes above
        // Re 800 (pure NSCBC blows up at Re2000 / 2048^2).
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
        auto m = DeviceCompNscbc::make(g, nc);
        if (!m) return fail(6, "NSCBC init failed");
        nsc = std::move(*m);
    }

    auto ghost = DeviceCompGhost::make(g, sdf, r.gamma, r.rhoInf, r.pInf,
                                       bbox ? bbox->data() : nullptr);
    if (!ghost) return fail(6, "ghost wall init failed");

    auto applyBc = [&](DeviceCompState& s) {
        far->apply(s);
        ghost->apply(s);
    };
    dev->setStageHook(applyBc);
    if (nsc)
        dev->setRhsHook([&](const DeviceCompState& s, DeviceCompState& k) {
            nsc->apply(*dev, s, k);
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

    const std::string cols[] = {"t", "dt", "cd", "cl", "cs", "wmax"};
    out.declareSeries("stats", cols);

    const float vortScale =
        float(r.curlMax > 0 ? r.curlMax
                            : (r.signedVorticity ? 8.0 : 15.0) * r.uInf / r.D);
    const float speedMax = float(1.6 * r.uInf);
    const char* vortName = r.signedVorticity ? "vort" : "curl";
    std::vector<Real> hVort(g.size()), hSpeed(g.size());
    std::vector<float> vortF(g.size()), speedF(g.size());
    auto writeFrame = [&](double t) {
        primitiveVelocity(gg, ds->rho, ds->mx, ds->my, ds->mz, *du, *dv, *dw);
        if (r.signedVorticity) vorticityZCentred(gg, *du, *dv, *dVort);
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
            {vortName, vortF.data(), 0, vortScale, r.signedVorticity},
            {"speed", speedF.data(), 0, speedMax, false}};
        if (auto e = out.frame(t, FrameViews{ff}); !e)
            std::fprintf(stderr, "frame write failed: %s\n", e.error().c_str());
    };

    RunStats st;
    double t = 0.0;
    long lastLog = 0;
    double dtStab = 0.0;
    int sinceDt = 1 << 30;
    const auto wall0 = std::chrono::steady_clock::now();

    while (t < r.tEnd)
    {
        if (++sinceDt >= r.dtRecompute)
        {
            const double s = dev->stableDt(*ds, *red, r.cfl);
            if (!std::isfinite(s) || s <= 0)
                return fail(7, "stable dt is not finite and positive "
                               "(the state diverged)");
            dtStab = r.dtMargin * s;
            sinceDt = 0;
        }
        const double dt = std::min(r.tEnd - t, dtStab);
        if (dtStab < 1e-6)
            return fail(7, "dt collapsed -- the state diverged");

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
            const double wmax =
                r.cfl / dtStab * std::min({g.hx, g.hy, g.hz});
            const double row[] = {t, dt, st.cd, st.cl, st.cs, wmax};
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
        if (out.wantsFrame(t) || out.wantsPng(t) || out.wantsWeb(t))
            writeFrame(t);
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
