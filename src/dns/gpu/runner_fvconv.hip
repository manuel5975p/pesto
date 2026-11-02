#include "dns/gpu/runner_fvconv.hpp"

#include "dns/gpu/fv_incomp.hpp"
#include "dns/gpu/hip_runtime.hpp"
#include "dns/gpu/internal/run_loop.hpp"
#include "dns/interrupt.hpp"
#include "dns/la/graph.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <format>
#include <string>
#include <vector>

namespace dns::gpu {

using internal::fail;
using internal::finishTiming;
using internal::wallGradientPinned;

namespace {

// Cell -> height bin, in ascending cell order.
struct Bins
{
    int n = 0;
    std::vector<int> bin;
    std::vector<double> vol, y; // per bin: volume, V-weighted mean height
};

std::expected<Bins, std::string> binsOf(const FaceMesh& fm, int n, double H)
{
    Bins b{.n = n, .bin = {}, .vol = std::vector<double>(std::size_t(n), 0.0), .y = std::vector<double>(std::size_t(n), 0.0)};
    b.bin.resize(std::size_t(fm.nC));
    for (std::size_t c = 0; c < std::size_t(fm.nC); ++c) {
        const int j = std::clamp(int(std::floor(fm.C[1][c] / H * n)), 0, n - 1);
        b.bin[c] = j;
        b.vol[std::size_t(j)] += fm.V[c];
        b.y[std::size_t(j)] += fm.V[c] * fm.C[1][c];
    }
    for (std::size_t j = 0; j < std::size_t(n); ++j) {
        if (!(b.vol[j] > 0.0))
            return std::unexpected(std::format("--ny {}: height bin {} holds no cell centre: take fewer bins", n, j));
        b.y[j] /= b.vol[j];
    }
    return b;
}

// Plane statistics of one state.
struct Sample
{
    double nuBot = 0, nuTop = 0, nuFlux = 0, urms = 0, ke = 0;
    std::vector<double> rho, T, v, vT, nu; // per bin
};

// Runner_conv's measure on bins. rho null: unit density, runner_rb's.
Sample measure(const FaceMesh& fm, const Bins& b, const double* rho, const double* u, const double* v, const double* w,
               const double* T, const FvConvRun& r)
{
    const std::size_t n = std::size_t(b.n);
    Sample s{.rho = std::vector<double>(n, 0.0),
             .T = std::vector<double>(n, 0.0),
             .v = std::vector<double>(n, 0.0),
             .vT = std::vector<double>(n, 0.0),
             .nu = std::vector<double>(n, 0.0)};
    std::vector<double> rvT(n, 0.0);
    double mass = 0, keSum = 0;
    for (std::size_t c = 0; c < std::size_t(fm.nC); ++c) {
        const std::size_t j = std::size_t(b.bin[c]);
        const double V = fm.V[c], d = rho ? rho[c] : 1.0, t = T[c], vv = v[c];
        s.rho[j] += V * d;
        s.T[j] += V * t;
        s.v[j] += V * vv;
        rvT[j] += V * d * vv * t;
        mass += V * d;
        keSum += V * d * (u[c] * u[c] + vv * vv + w[c] * w[c]);
    }
    for (std::size_t j = 0; j < n; ++j) {
        s.rho[j] /= b.vol[j];
        s.T[j] /= b.vol[j];
        s.v[j] /= b.vol[j];
        s.vT[j] = rvT[j] / b.vol[j] / s.rho[j];
    }
    const double dT = r.Tbot - r.Ttop, cond = r.kapDiff * dT / r.H;
    const std::size_t hi = n - 1;
    s.nuBot = -wallGradientPinned(s.T[0] - r.Tbot, s.T[1] - r.Tbot, b.y[0], b.y[1]) * r.H / dT;
    s.nuTop = wallGradientPinned(s.T[hi] - r.Ttop, s.T[hi - 1] - r.Ttop, r.H - b.y[hi], r.H - b.y[hi - 1]) * r.H / dT;
    double sum = 0;
    for (std::size_t j = 1; j < hi; ++j) {
        const double dTdy = (s.T[j + 1] - s.T[j - 1]) / (b.y[j + 1] - b.y[j - 1]);
        s.nu[j] = (s.vT[j] - r.kapDiff * dTdy) / cond;
        sum += s.nu[j];
    }
    s.nuFlux = n > 2 ? sum / double(n - 2) : 0.0;
    s.urms = mass > 0 ? std::sqrt(keSum / mass) : 0.0;
    s.ke = 0.5 * keSum;
    return s;
}

std::vector<double> vorticityOf(const double* wzP, const double* wzM, const double* wxP, const double* wxM,
                                const double* wyP, const double* wyM, std::size_t n, bool signedWz)
{
    std::vector<double> o(n);
    for (std::size_t c = 0; c < n; ++c) {
        const double wz = wzP[c] - wzM[c];
        if (signedWz) {
            o[c] = wz;
            continue;
        }
        const double wx = wxP[c] - wxM[c], wy = wyP[c] - wyM[c];
        o[c] = std::sqrt(wx * wx + wy * wy + wz * wz);
    }
    return o;
}

double areaOf(const FaceMesh& fm, const std::vector<int>& patches)
{
    double a = 0;
    for (int p : patches) {
        const MeshPatch& m = fm.patches[std::size_t(p)];
        for (int i = 0; i < m.count; ++i) {
            const std::size_t f = std::size_t(fm.nI + m.first + i);
            a += std::hypot(fm.S[0][f], fm.S[1][f], fm.S[2][f]);
        }
    }
    return a;
}

// Series, profile rows, frames: what the two loops share.
class Sink
{
public:
    Sink(const FaceMesh& fm, const FvRaster& raster, const FvConvRun& r, RunOutput& out)
        : fm_(fm), raster_(raster), r_(r), out_(out), ch0_(raster.grid().size()), ch1_(raster.grid().size()),
          ch2_(raster.grid().size())
    {
    }

    std::expected<void, RunError> declare(std::vector<std::string> extra)
    {
        std::vector<std::string> cols = {"t", "dt", "nuBot", "nuTop", "nuFlux", "urms", "KE", "nuBotF", "nuTopF"};
        cols.insert(cols.end(), extra.begin(), extra.end());
        if (auto e = out_.declareSeries("stats", cols, r_.label); !e) return fail(4, e.error());
        const std::string pc[] = {"t", "y", "T", "rho", "v", "vT", "nu"};
        if (auto e = out_.declareSeries("profiles", pc); !e) return fail(4, e.error());
        return {};
    }

    void profiles(double t, const Sample& s, const Bins& b)
    {
        for (std::size_t j = 0; j < std::size_t(b.n); ++j) {
            const double row[] = {t, b.y[j], s.T[j], s.rho[j], s.v[j], s.vT[j], s.nu[j]};
            out_.row("profiles", row);
        }
    }

    std::expected<void, RunError> frame(double t, const std::vector<double>& vort, const std::vector<double>& speed,
                                        const std::vector<double>& T)
    {
        const double tC = 0.5 * (r_.Tbot + r_.Ttop);
        const float tS = float(0.5 * (r_.Tbot - r_.Ttop));
        std::vector<double> dT(T.size());
        std::ranges::transform(T, dT.begin(), [&](double x) { return x - tC; });
        raster_.sample(vort, ch0_);
        raster_.sample(speed, ch1_);
        raster_.sample(dT, ch2_);
        const bool sw = fm_.dim == 2;
        const FrameField ff[3] = {{sw ? "vort" : "curl", ch0_.data(), 0, float(8.0 * r_.uFf / r_.H), sw},
                                  {"speed", ch1_.data(), 0, float(1.6 * r_.uFf), false},
                                  {"temp", ch2_.data(), -tS, tS, true}};
        if (auto e = out_.frame(t, ff); !e) return fail(4, e.error());
        return {};
    }

private:
    const FaceMesh& fm_;
    const FvRaster& raster_;
    const FvConvRun& r_;
    RunOutput& out_;
    std::vector<float> ch0_, ch1_, ch2_;
};

void printRow(const FvConvRun& r, double t, double dt, const Sample& s)
{
    std::printf("t=%.2f/%.2f Nu=%.3f/%.3f (flux %.3f) urms/uff=%.3f dt=%.2e\n", t, r.tEnd, s.nuBot, s.nuTop, s.nuFlux,
                r.uFf > 0 ? s.urms / r.uFf : 0.0, dt);
    std::fflush(stdout);
}

} // namespace

std::expected<RunStats, RunError> runFvConvComp(const FaceMesh& fm, const dns::FvState& ic, const FvCompConfig& cfg,
                                                const FvConvRun& r, const FvRaster& raster, RunOutput& out)
{
    auto bins = binsOf(fm, r.bins, r.H);
    if (!bins) return fail(3, bins.error());
    auto made = DeviceFvComp::make(fm, cfg);
    if (!made) return fail(made.error().starts_with("FV device") ? 6 : 3, made.error());
    DeviceFvComp& dev = *made;
    if (auto u = dev.upload(ic); !u) return fail(6, u.error());
    if (!r.quiet) {
        std::printf("# %s\n# FV path: %d cells, %d internal faces, %d boundary faces, %.1f MB on the device\n",
                    r.label.c_str(), fm.nC, fm.nI, fm.nB, double(dev.deviceBytes()) / 1048576.0);
        std::fflush(stdout);
    }
    Sink sink(fm, raster, r, out);
    if (auto e = sink.declare({"machMax", "mass", "energy"}); !e) return std::unexpected(e.error());
    const double aBot = areaOf(fm, r.plates.floor), aTop = areaOf(fm, r.plates.lid);
    const double condF = r.kapFace * (r.Tbot - r.Ttop) / r.H;
    const bool sw = fm.dim == 2;
    std::array<std::vector<double>, 7> prim;
    std::array<std::vector<double>, 12> grad;
    auto sample = [&]() -> std::expected<Sample, RunError> {
        if (auto c = dev.cellFields(prim, grad); !c) return fail(6, c.error());
        return measure(fm, *bins, prim[0].data(), prim[1].data(), prim[2].data(), prim[3].data(), prim[6].data(), r);
    };
    auto writeFrame = [&](double t) -> std::expected<void, RunError> {
        if (auto c = dev.cellFields(prim, grad); !c) return fail(6, c.error());
        // grad rows [3 field + axis]; fields u, v, w.
        auto vort = vorticityOf(grad[3].data(), grad[1].data(), grad[7].data(), grad[5].data(), grad[2].data(),
                                grad[6].data(), prim[0].size(), sw);
        std::vector<double> sp(prim[0].size());
        for (std::size_t c = 0; c < sp.size(); ++c) sp[c] = std::hypot(prim[1][c], prim[2][c], prim[3][c]);
        return sink.frame(t, vort, sp, prim[6]);
    };

    RunStats st;
    double t = 0.0, dtStab = 0.0;
    int sinceDt = 1 << 30;
    internal::StatsCadence cad{r.statsEvery};
    internal::ProfileSchedule prof(r.profileDt);
    const auto wall0 = std::chrono::steady_clock::now();
    while (t < r.tEnd) {
        if (stopRequested()) break;
        std::string diverged;
        // 1e-9: runner_conv's collapse threshold, the same unit box.
        if (auto e = internal::recomputeDt(sinceDt, dtStab, r.dtRecompute, r.dtMargin, 1e-9, [&] {
                auto d = dev.stableDt({.cfl = r.cfl});
                if (!d) diverged = d.error();
                return d ? *d : -1.0;
            });
            !e)
            return fail(7, diverged.empty() ? e.error() : diverged);
        const double dt = std::min(r.tEnd - t, dtStab);
        dev.step(dt);
        t += dt;
        ++st.steps;
        const bool statsDue = cad.due(st.steps, t, r.tEnd), profDue = prof.due(t, r.tEnd);
        if (statsDue || profDue) {
            auto s = sample();
            if (!s) return std::unexpected(s.error());
            if (statsDue) {
                auto q = dev.surface();
                if (!q) return fail(7, q.error());
                auto tot = dev.totals();
                if (!tot) return fail(6, tot.error());
                double qb = 0, qt = 0;
                for (int p : r.plates.floor) qb += q->heatPatch[std::size_t(p)];
                for (int p : r.plates.lid) qt += q->heatPatch[std::size_t(p)];
                const double row[] = {t,      dt,    s->nuBot, s->nuTop,        s->nuFlux,     s->urms, s->ke,
                                      -qb / (aBot * condF), qt / (aTop * condF), tot->machMax, tot->mass, tot->energy};
                out.row("stats", row);
                if (!r.quiet && cad.printDue(st.steps, t, r.tEnd)) {
                    printRow(r, t, dt, *s);
                    cad.printed(st.steps);
                }
                cad.logged(st.steps);
            }
            if (profDue) {
                sink.profiles(t, *s, *bins);
                prof.done(t);
            }
        }
        if (out.wantsAnyFrame(t))
            if (auto e = writeFrame(t); !e) return std::unexpected(e.error());
    }
    st.tStop = t;
    finishTiming(st, wall0, out);
    return st;
}

std::expected<RunStats, RunError> runFvConvRb(const FaceMesh& fm, const FvIncompBcs& b, const FvIncompControls& ctl,
                                              const std::vector<double>& U0, const std::vector<double>& T0,
                                              const FvConvRun& r, const FvRaster& raster, RunOutput& out)
{
    auto bins = binsOf(fm, r.bins, r.H);
    if (!bins) return fail(3, bins.error());
    auto setup = buildFvIncompSetup(fm, b);
    if (!setup) return fail(3, setup.error());
    const FvIncompSetup& s = *setup;
    auto ds = DeviceFvIncomp::make(s, la::graphOf(fm), ctl, U0, T0);
    if (!ds) return fail(6, ds.error());
    if (!r.quiet) {
        std::printf("# %s\n# FV path: %d cells, %d internal faces, %d boundary faces, max non-orthogonality %.2f deg\n",
                    r.label.c_str(), s.nC, s.nI, s.nB, s.nonOrthMaxDeg);
        std::fflush(stdout);
    }
    Sink sink(fm, raster, r, out);
    if (auto e = sink.declare({"maxDiv", "pcgIters"}); !e) return std::unexpected(e.error());
    const std::size_t nC = std::size_t(s.nC), nB = std::size_t(s.nB);
    // Boundary face -> plate: +1 floor, -1 lid.
    std::vector<int> plate(nB, 0);
    auto mark = [&](const std::vector<int>& ps, int v) {
        for (int p : ps) {
            const MeshPatch& m = fm.patches[std::size_t(p)];
            for (std::size_t i = 0; i < nB; ++i)
                if (s.bFace[i] >= m.first && s.bFace[i] < m.first + m.count) plate[i] = v;
        }
    };
    mark(r.plates.floor, 1);
    mark(r.plates.lid, -1);
    const double aBot = areaOf(fm, r.plates.floor), aTop = areaOf(fm, r.plates.lid);
    const double condF = r.kapFace * (r.Tbot - r.Ttop) / r.H;
    const double hmin = 1.0 / std::sqrt(*std::ranges::max_element(s.id2));
    // The free-fall bound: what limits the first steps of a layer at rest (runner_rb).
    const double gMag = std::hypot(ctl.betaG[0], ctl.betaG[1], ctl.betaG[2]);
    const double dtBuoy = std::sqrt(hmin / std::max(gMag * (r.Tbot - r.Ttop), 1e-30));
    const bool sw = fm.dim == 2;

    auto writeFrame = [&](double t) -> std::expected<void, RunError> {
        auto h = ds->state();
        if (!h) return fail(7, h.error());
        auto g = ds->gradU();
        if (!g) return fail(7, g.error());
        // (3 i + q) nC: d_i u_q.
        const double* G = g->data();
        auto at = [&](int i, int q) { return G + std::size_t(3 * i + q) * nC; };
        auto vort = vorticityOf(at(0, 1), at(1, 0), at(1, 2), at(2, 1), at(2, 0), at(0, 2), nC, sw);
        std::vector<double> sp(nC);
        for (std::size_t c = 0; c < nC; ++c) sp[c] = std::hypot(h->U[c], h->U[nC + c], h->U[2 * nC + c]);
        return sink.frame(t, vort, sp, h->T);
    };

    RunStats st;
    double t = 0.0;
    long unconverged = 0, solves = 0;
    internal::StatsCadence cad{r.statsEvery};
    internal::ProfileSchedule prof(r.profileDt);
    const auto wall0 = std::chrono::steady_clock::now();
    while (t < r.tEnd) {
        if (stopRequested()) break;
        auto dtS = ds->stableDt(r.cfl);
        if (!dtS) return fail(7, dtS.error());
        if (!(*dtS > 0) || !std::isfinite(*dtS))
            return fail(7, "stable dt is not finite and positive (the state diverged)");
        // Equal steps to tEnd, never one short step: DeviceFvIncomp::step's precondition.
        const double rem = r.tEnd - t, lim = std::min(*dtS, dtBuoy);
        if (rem <= 1e-9 * lim) break;
        const double dt = rem / std::ceil(rem / lim);
        auto lg = ds->step(dt);
        if (!lg) return fail(7, std::format("t = {:g}: {}", t, lg.error()));
        t += dt;
        ++st.steps;
        unconverged += lg->unconverged;
        solves += lg->solves;
        const bool statsDue = cad.due(st.steps, t, r.tEnd), profDue = prof.due(t, r.tEnd);
        if (statsDue || profDue) {
            auto h = ds->state();
            if (!h) return fail(7, h.error());
            const Sample smp = measure(fm, *bins, nullptr, h->U.data(), h->U.data() + nC, h->U.data() + 2 * nC,
                                       h->T.data(), r);
            if (statsDue) {
                auto d = ds->diagnostics();
                if (!d) return fail(7, d.error());
                if (!std::isfinite(d->K)) return fail(7, "the kinetic energy is not finite (the state diverged)");
                auto q = ds->wallHeat();
                if (!q) return fail(7, q.error());
                double qb = 0, qt = 0;
                for (std::size_t i = 0; i < nB; ++i) {
                    if (plate[i] > 0) qb += (*q)[i];
                    if (plate[i] < 0) qt += (*q)[i];
                }
                const double row[] = {t,       dt,      smp.nuBot,           smp.nuTop,          smp.nuFlux, smp.urms,
                                      smp.ke,  -qb / (aBot * condF), qt / (aTop * condF), d->divMax, double(lg->iterations)};
                out.row("stats", row);
                if (!r.quiet && cad.printDue(st.steps, t, r.tEnd)) {
                    printRow(r, t, dt, smp);
                    cad.printed(st.steps);
                }
                cad.logged(st.steps);
                if (unconverged > 0) {
                    std::fprintf(stderr, "dns: warning t=%.4g: %ld of %ld pressure solves stopped short of --tol\n", t,
                                 unconverged, solves);
                    unconverged = solves = 0;
                }
            }
            if (profDue) {
                sink.profiles(t, smp, *bins);
                prof.done(t);
            }
        }
        if (out.wantsAnyFrame(t))
            if (auto e = writeFrame(t); !e) return std::unexpected(e.error());
    }
    st.tStop = t;
    finishTiming(st, wall0, out);
    return st;
}

} // namespace dns::gpu
