#include "dns/gpu/runner_fvcomp.hpp"

#include "dns/gpu/chem.hpp"
#include "dns/gpu/hip_runtime.hpp"
#include "dns/gpu/internal/frame_scratch.hpp"
#include "dns/gpu/internal/run_loop.hpp"
#include "dns/interrupt.hpp"
#include "dns/steady_stop.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <format>
#include <optional>
#include <string>
#include <vector>

namespace dns::gpu {

using internal::fail;
using internal::finishTiming;
using internal::rampFrac;

namespace {

// Vorticity channel of a frame: signed wz on a 2D mesh, |curl| otherwise.
// grad rows [3 field + axis]; fields u, v, w.
std::vector<double> vorticity(const std::array<std::vector<double>, 12>& g, bool signedWz)
{
    const std::size_t n = g[0].size();
    std::vector<double> o(n);
    for (std::size_t c = 0; c < n; ++c) {
        const double wz = g[3][c] - g[1][c];
        if (signedWz) {
            o[c] = wz;
            continue;
        }
        const double wx = g[7][c] - g[5][c], wy = g[2][c] - g[6][c];
        o[c] = std::sqrt(wx * wx + wy * wy + wz * wz);
    }
    return o;
}

double dot3(const std::array<double, 3>& a, const std::array<double, 3>& b)
{
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

} // namespace

std::expected<RunStats, RunError> runFvComp(const FaceMesh& fm, const dns::FvState& ic, const FvCompRun& r,
                                            const FvRaster& raster, RunOutput& out)
{
    auto made = DeviceFvComp::make(fm, r.solver);
    if (!made) return fail(made.error().starts_with("FV device") ? 6 : 3, made.error());
    DeviceFvComp& dev = *made;
    if (auto u = dev.upload(ic); !u) return fail(6, u.error());
    // The solver validated the converged table; a ramp starts from frac 0.
    auto retarget = [&](double frac) -> std::expected<void, RunError> {
        for (std::size_t p = 0; p < fm.patches.size(); ++p) {
            if (fm.patches[p].type == FoamPatchType::Empty) continue;
            const auto it = std::ranges::find(r.spec.patches, fm.patches[p].name, &FvPatchSpec::name);
            if (it->kind != FvPatchKind::Far && it->kind != FvPatchKind::Inflow) continue;
            if (auto e = dev.setBc(int(p), fvBcOf(*it, r.fs, frac)); !e) return fail(6, e.error());
        }
        return {};
    };
    if (r.ramp > 0)
        if (auto e = retarget(0.0); !e) return std::unexpected(e.error());
    if (!r.quiet) {
        std::printf("# %s\n# FV path: %d cells, %d internal faces, %d boundary faces, %.1f MB on the device\n",
                    r.label.c_str(), fm.nC, fm.nI, fm.nB, double(dev.deviceBytes()) / 1048576.0);
        std::fflush(stdout);
    }

    const bool rans = r.solver.rans.model != RansModel::None, burning = r.flame || r.jet;
    std::vector<std::string> cols =
        r.jet ? std::vector<std::string>{"t",     "dt",    "dtChem", "xTip",   "xBase",   "half",      "Tmax",
                                         "TmaxIn", "burn", "fuelIn", "fuelOut", "burnRatio", "mass",     "energy",
                                         "pMean", "zMin",  "zMax",   "yfMin",  "yoMin",   "hits",      "uJet"} :
        r.flame ? std::vector<std::string>{"t",   "dt",    "dtChem", "SL",    "SLdisp", "xf",    "dth",
                                           "Tmax", "mass", "energy", "omegaEdge", "yHits", "uFresh", "pFresh",
                                           "TFresh", "yMin", "yMax",  "burn",  "fuel"}
                : std::vector<std::string>{"t",   "dt",  "cd",   "cl",     "cs",     "heat",
                                           "machMax", "cdf", "clf", "mass", "energy", "kinetic"};
    if (rans) {
        cols.emplace_back("mutMax");
        if (ransSst(r.solver.rans.model)) cols.emplace_back("floorHits");
    }
    if (auto e = out.declareSeries("stats", cols, r.label); !e) return fail(4, e.error());
    if (r.flame) {
        const std::string pcols[] = {"t", "x", "T", "rho", "u", "p", "Y", "omega"};
        if (auto e = out.declareSeries("profiles", pcols); !e) return fail(4, e.error());
    }

    const std::array<double, 3>& dir = r.fs.dir;
    const std::array<double, 3> lift = r.spec.lift;
    const std::array<double, 3> side = {dir[1] * lift[2] - dir[2] * lift[1], dir[2] * lift[0] - dir[0] * lift[2],
                                        dir[0] * lift[1] - dir[1] * lift[0]};
    const double qA = 0.5 * r.fs.rho * r.fs.U * r.fs.U * r.refArea;

    const bool signedWz = fm.dim == 2;
    const float vortScale = float(r.curlMax > 0 ? r.curlMax : (signedWz ? 8.0 : 15.0) * r.fs.U / r.D);
    const float speedMax = float(1.6 * r.fs.U);
    const Grid& rg = raster.grid();
    std::vector<float> chan0(rg.size()), chan1(rg.size());
    std::array<std::vector<double>, 7> prim;
    std::array<std::vector<double>, 12> grad;
    dns::FvState flameState;
    std::vector<double> flameOmega;
    std::vector<float> chan2;
    auto writeFlameFrame = [&](double t) -> std::expected<void, RunError> {
        const FvFlameSpec& fl = r.flame->spec();
        if (auto c = dev.speciesFields(flameState, flameOmega); !c) return fail(6, c.error());
        if (auto c = dev.cellFields(prim, grad); !c) return fail(6, c.error());
        chan2.resize(rg.size());
        std::vector<double> dT(prim[6]);
        std::ranges::for_each(dT, [&](double& v) { v -= fl.Tu; });
        const std::size_t n = dT.size();
        std::vector<double> Y(n);
        const double* ry = flameState.var(kFvMeanVars + fl.fuel);
        for (std::size_t c = 0; c < n; ++c) Y[c] = ry[c] / prim[0][c];
        raster.sample(dT, chan0);
        raster.sample(Y, chan1);
        raster.sample(flameOmega, chan2);
        const FrameField ff[3] = {{"temp", chan0.data(), 0, float((fl.sigma - 1.0) * fl.Tu), false},
                                  {"fuel", chan1.data(), 0, 1.0f, false},
                                  {"omega", chan2.data(), 0, float(1.0 / fl.dth), false}};
        if (auto e = out.frame(t, ff); !e) return fail(4, e.error());
        return {};
    };
    std::vector<float> chan3;
    auto writeJetFrame = [&](double t) -> std::expected<void, RunError> {
        const FvJetSpec& js = r.jet->spec();
        if (auto c = dev.speciesFields(flameState, flameOmega); !c) return fail(6, c.error());
        if (auto c = dev.cellFields(prim, grad); !c) return fail(6, c.error());
        chan2.resize(rg.size());
        chan3.resize(rg.size());
        const std::size_t n = prim[0].size();
        std::vector<double> dT(prim[6]), YF(n), YO(n);
        std::ranges::for_each(dT, [&](double& v) { v -= js.T0; });
        for (std::size_t c = 0; c < n; ++c) {
            YF[c] = flameState.var(kFvMeanVars)[c] / prim[0][c];
            YO[c] = flameState.var(kFvMeanVars + 1)[c] / prim[0][c];
        }
        raster.sample(dT, chan0);
        raster.sample(YF, chan1);
        raster.sample(YO, chan2);
        raster.sample(flameOmega, chan3);
        const FrameField ff[4] = {{"temp", chan0.data(), 0, float((js.model.sigma - 1.0) * js.T0), false},
                                  {"fuel", chan1.data(), 0, 1.0f, false},
                                  {"oxid", chan2.data(), 0, 1.0f, false},
                                  {"omega", chan3.data(), 0, 1.0f, false}};
        if (auto e = out.frame(t, ff); !e) return fail(4, e.error());
        return {};
    };
    auto writeFrame = [&](double t) -> std::expected<void, RunError> {
        if (r.jet) return writeJetFrame(t);
        if (r.flame) return writeFlameFrame(t);
        if (auto c = dev.cellFields(prim, grad); !c) return fail(6, c.error());
        raster.sample(vorticity(grad, signedWz), chan0);
        std::vector<double> sp(prim[0].size());
        for (std::size_t c = 0; c < sp.size(); ++c) {
            const double m = std::sqrt(prim[1][c] * prim[1][c] + prim[2][c] * prim[2][c] + prim[3][c] * prim[3][c]);
            sp[c] = r.speedDelta ? m - r.fs.U : m;
        }
        raster.sample(sp, chan1);
        const FrameField ff[2] = {{signedWz ? "vort" : "curl", chan0.data(), 0, vortScale, signedWz,
                                   r.vortAsinh && signedWz ? vortScale / 32.0f : 0.0f},
                                  internal::speedField(chan1.data(), r.speedDelta, r.fs.U, speedMax)};
        if (auto e = out.frame(t, ff); !e) return fail(4, e.error());
        return {};
    };

    RunStats st;
    double t = 0.0, dtStab = 0.0, uCur = r.ramp > 0 ? 0.0 : r.fs.U;
    int sinceDt = 1 << 30;
    long step = 0;
    double dtFirst = 0.0;
    internal::StatsCadence cad{r.statsEvery};
    internal::ProfileSchedule prof(r.profileDt);
    const auto wall0 = std::chrono::steady_clock::now();

    double dtChem = HUGE_VAL, xfPrev = NAN, tPrev = 0, lastHits = 0, lastMin = 0, lastMax = 1;
    bool bindChem = false;
    // One flame measure: the stats row and/or the profile, the drift guard.
    auto flameRow = [&](double dt, bool row, bool profile) -> std::expected<void, RunError> {
        if (auto c = dev.speciesFields(flameState, flameOmega); !c) return fail(6, c.error());
        const FvFlameMeasure m = r.flame->measure(flameState, flameOmega);
        if (!std::isfinite(m.mass) || !std::isfinite(m.Tmax) || !std::isfinite(m.xf))
            return fail(7, "flame state is not finite or has no mid-temperature crossing (diverged)");
        lastHits = m.hits;
        lastMin = m.yMin;
        lastMax = m.yMax;
        if (row) {
            const double dxf = std::isfinite(xfPrev) && t > tPrev ? (m.xf - xfPrev) / (t - tPrev) : 0.0;
            const double v[] = {t,      dt,          dtChem, m.SL,   m.uRef - dxf, m.xf,   m.dth,    m.Tmax, m.mass, m.energy,
                                m.omegaEdge, m.hits, m.uRef, m.pRef, m.TRef,       m.yMin, m.yMax, m.burn, m.fuel};
            out.row("stats", v);
            xfPrev = m.xf;
            tPrev = t;
            cad.logged(step);
        }
        if (profile) {
            for (const FvFlameSlab& k : m.slabs) {
                const double v[] = {t, k.s, k.T, k.rho, k.u, k.p, k.ry / k.rho, k.omega};
                out.row("profiles", v);
            }
            prof.done(t);
        }
        if (!r.quiet && cad.printDue(step, t, r.tEnd)) {
            std::printf("t=%.3f/%.3f SL=%.6f xf=%.4f dth=%.4f Tmax/Tb=%.5f omegaEdge=%.2e yHits=%g dt=%.2e%s\n", t, r.tEnd,
                        m.SL, m.xf, m.dth, m.Tmax / (r.flame->spec().sigma * r.flame->spec().Tu), m.omegaEdge, m.hits, dt,
                        bindChem ? " (chem dt)" : "");
            std::fflush(stdout);
            cad.printed(step);
        }
        const double guard = 2.0 * r.flame->spec().dth;
        if (m.xf < guard || m.xf > r.flame->length() - guard)
            return fail(8, std::format("flame not stabilized: drifted to the {} end at t={:.4g} (x_f={:.4g}; adjust "
                                       "--inflow-speed)",
                                       m.xf < guard ? "inflow" : "outflow", t, m.xf));
        return {};
    };

    // One jet row and the structured runner's guard. fuelIn: what the open patches
    // carry in (Far, Inflow), fuelOut: what Outflow patches carry out.
    auto jetRow = [&](double dt) -> std::expected<void, RunError> {
        if (auto c = dev.speciesFields(flameState, flameOmega); !c) return fail(6, c.error());
        const FvJetMeasure m = r.jet->measure(flameState, flameOmega);
        if (!std::isfinite(m.mass) || !std::isfinite(m.Tmax)) return fail(7, "flame state is not finite (diverged)");
        auto pf = dev.patchFlux(kFvMeanVars);
        if (!pf) return fail(6, pf.error());
        double fuelIn = 0, fuelOut = 0;
        for (std::size_t p = 0; p < fm.patches.size(); ++p) {
            const FvBcKind k = r.solver.bc[p].kind;
            if (k == FvBcKind::Far || k == FvBcKind::Inflow) fuelIn -= (*pf)[p];
            if (k == FvBcKind::Outflow) fuelOut += (*pf)[p];
        }
        lastHits = m.hits;
        lastMin = std::min(m.yfMin, m.yoMin);
        lastMax = m.zMax;
        const double v[] = {t,     dt,      dtChem, m.xTip,  m.xBase, m.half,  m.Tmax,   m.TmaxIn, m.burn,
                            fuelIn, fuelOut, fuelIn > 0 ? m.burn / fuelIn : 0.0, m.mass, m.energy, m.pMean,
                            m.zMin, m.zMax,  m.yfMin, m.yoMin, m.hits,  1.0};
        out.row("stats", v);
        cad.logged(step);
        const FvJetSpec& js = r.jet->spec();
        if (!r.quiet && cad.printDue(step, t, r.tEnd)) {
            std::printf("t=%.3f/%.3f xTip=%.3f half=%.3f TmaxIn/Tad=%.4f burn/in=%.4f p/p0=%.4f hits=%g dt=%.2e%s\n", t, r.tEnd,
                        m.xTip, m.half, m.TmaxIn / (js.model.sigma * js.T0), fuelIn > 0 ? m.burn / fuelIn : 0.0, m.pMean,
                        m.hits, dt, bindChem ? " (chem dt)" : "");
            std::fflush(stdout);
            cad.printed(step);
        }
        const double Tmid = 0.5 * (1.0 + js.model.sigma) * js.T0;
        std::string why;
        if (m.TmaxIn < Tmid)
            why = std::format("Tmax past the ends = {:.4g} T0 fell below the mid temperature {:.4g} T0 (extinction)",
                              m.TmaxIn / js.T0, Tmid / js.T0);
        else if (!(m.xBase <= 0.5 * r.jet->length()))
            why = std::format("the flame base x={:.4g} lifted past half the mesh (lift-off)", m.xBase);
        if (!why.empty()) return fail(8, std::format("diffusion flame lost at t={:.4g}: {} (raise --damkohler)", t, why));
        return {};
    };

    auto statsRow = [&](double dt) -> std::expected<void, RunError> {
        auto s = dev.surface();
        if (!s) return fail(7, s.error());
        auto tot = dev.totals();
        if (!tot) return fail(6, tot.error());
        const std::array<double, 3> F = {s->fp[0] + s->fv[0], s->fp[1] + s->fv[1], s->fp[2] + s->fv[2]};
        st.cd = dot3(F, dir) / qA;
        st.cl = dot3(F, lift) / qA;
        st.cs = dot3(F, side) / qA;
        st.cdf = dot3(s->fv, dir) / qA;
        std::vector<double> row = {t,           dt,       st.cd,           st.cl,      st.cs,      s->heat,
                                   tot->machMax, st.cdf,  dot3(s->fv, lift) / qA, tot->mass, tot->energy, tot->kinetic};
        if (rans) {
            auto m = dev.eddyViscosity();
            if (!m) return fail(6, m.error());
            row.push_back(*std::ranges::max_element(*m));
            if (ransSst(r.solver.rans.model)) {
                auto h = dev.floorHits();
                if (!h) return fail(6, h.error());
                row.push_back(double(*h));
            }
        }
        out.row("stats", row);
        return {};
    };

    // The pseudo-time march: one row per stats cadence, the floored stop.
    if (r.steadyTol > 0) {
        const int nT = ransScalars(r.solver.rans.model);
        SteadyStop stop(3 + nT, 2);
        double remaining = HUGE_VAL;
        for (; step < r.steadySteps; ) {
            if (stopRequested()) break;
            if (step % r.dtRecompute == 0) {
                auto d = dev.localDt({.cfl = r.cfl});
                if (!d) return fail(7, d.error());
                dtStab = r.dtMargin * *d;
            }
            dev.step(dtStab);
            t += dtStab;
            ++step;
            if (step % r.statsEvery != 0) continue;
            if (auto e = statsRow(dtStab); !e) return std::unexpected(e.error());
            dns::FvState s;
            if (auto e = dev.download(s); !e) return fail(6, e.error());
            std::vector<double> q = {st.cd, st.cdf, 0.0};
            for (std::size_t c = 0; c < std::size_t(fm.nC); ++c) {
                const double* qr = s.q.data();
                const std::size_t n = std::size_t(fm.nC);
                q[2] += 0.5 * (qr[n + c] * qr[n + c] + qr[2 * n + c] * qr[2 * n + c] + qr[3 * n + c] * qr[3 * n + c]) /
                        qr[c] * fm.V[c];
            }
            for (int e = 0; e < nT; ++e) {
                double v = 0.0;
                for (std::size_t c = 0; c < std::size_t(fm.nC); ++c) v += s.var(5 + e)[c] * fm.V[c];
                q.push_back(v);
            }
            stop.push(q);
            remaining = stop.remaining(r.steadyTol);
            if (!r.quiet && cad.printDue(step, double(step), 1e300)) {
                std::printf("iter %ld dtRef %.4g cd %.6f cl %.6f remaining %.3g\n", step, dtStab, st.cd, st.cl, remaining);
                std::fflush(stdout);
                cad.printed(step);
            }
            if (remaining < r.steadyTol) break;
        }
        st.steps = step;
        st.tStop = t;
        finishTiming(st, wall0, out);
        for (const SteadyFloor& f : stop.floors())
            std::printf("steady series: trend %.3g jitter %.3g remaining %.3g\n", f.trend, f.jitter, f.remaining);
        if (stopRequested()) return st;
        if (!(remaining < r.steadyTol))
            return fail(8, std::format("steady state not reached in {} iterations: remaining {:.3g}, tol {:.3g}", step,
                                       remaining, r.steadyTol));
        return st;
    }

    while (t < r.tEnd) {
        if (stopRequested()) break;
        // Collapse is relative: a real mesh's dt has no absolute scale.
        std::string diverged;
        if (auto e = internal::recomputeDt(sinceDt, dtStab, r.dtRecompute, r.dtMargin, 1e-4 * dtFirst, [&] {
                auto d = dev.stableDt({.cfl = r.cfl});
                if (!d) diverged = d.error();
                if (!d) return -1.0;
                if (!burning) return *d;
                auto dc = dev.chemDt(kChemDt);
                if (!dc) {
                    diverged = dc.error();
                    return -1.0;
                }
                dtChem = *dc;
                bindChem = dtChem < *d;
                return std::min(*d, dtChem);
            });
            !e)
            return fail(7, diverged.empty() ? e.error() : diverged);
        if (dtFirst == 0.0) dtFirst = dtStab;
        const double dt = std::min(r.tEnd - t, dtStab);
        dev.step(dt);
        if (t < r.ramp) {
            // Quiet start, as the structured O-grid: an exact Galilean boost,
            // then the open patches retarget.
            const double frac = rampFrac(t + dt, r.ramp), uNew = r.fs.U * frac;
            const double du = uNew - uCur;
            dev.boost({du * dir[0], du * dir[1], du * dir[2]});
            uCur = uNew;
            if (auto e = retarget(frac); !e) return std::unexpected(e.error());
        }
        t += dt;
        ++step;
        if (r.flame) {
            const bool row = cad.due(step, t, r.tEnd), profile = prof.due(t, r.tEnd);
            if (row || profile)
                if (auto e = flameRow(dt, row, profile); !e) return std::unexpected(e.error());
        } else if (r.jet) {
            if (cad.due(step, t, r.tEnd))
                if (auto e = jetRow(dt); !e) return std::unexpected(e.error());
        } else if (cad.due(step, t, r.tEnd)) {
            if (auto e = statsRow(dt); !e) return std::unexpected(e.error());
            cad.logged(step);
        }
        if (out.wantsAnyFrame(t))
            if (auto e = writeFrame(t); !e) return std::unexpected(e.error());
        if (!burning && !r.quiet && cad.printDue(step, t, r.tEnd)) {
            std::printf("step %ld t %.5g dt %.4g cd %.6f cl %.6f\n", step, t, dt, st.cd, st.cl);
            std::fflush(stdout);
            cad.printed(step);
        }
    }
    if (r.flame && lastHits > 0)
        std::fprintf(stderr,
                     "dns: warning: %g cells with Y outside [-1e-6, 1+1e-6] at the last stats row (counted, never "
                     "clamped): Y in [%.3g, 1%+.3g]\n",
                     lastHits, lastMin, lastMax - 1);
    if (r.jet && lastHits > 0)
        std::fprintf(stderr,
                     "dns: warning: %g cells with Z, Y_F or Y_O outside [-1e-6, 1+1e-6] at the last stats row (counted, "
                     "never clamped): min Y %.3g, max Z 1%+.3g\n",
                     lastHits, lastMin, lastMax - 1);
    if (burning) std::printf("label: %s\n", r.label.c_str());
    st.steps = step;
    st.tStop = t;
    finishTiming(st, wall0, out);
    return st;
}

} // namespace dns::gpu
