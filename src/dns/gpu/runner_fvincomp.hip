#include "dns/gpu/runner_fvincomp.hpp"

#include "dns/diagnostics.hpp"
#include "dns/foamio.hpp"
#include "dns/gpu/fv_incomp.hpp"
#include "dns/gpu/hip_runtime.hpp"
#include "dns/interrupt.hpp"
#include "dns/la/graph.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <format>
#include <fstream>
#include <map>

namespace dns::gpu {
namespace fs = std::filesystem;

namespace {

std::unexpected<RunError> fail(int code, const std::string& m) { return std::unexpected(RunError{code, m}); }

// t within this fraction of a step of tEnd: arrived, the rest is t's own rounding.
constexpr double kArrived = 1e-9;

// U and p in polyMesh cell and patch-face order under dir. Cyclic and `empty`
// patches take their constraint type; p on a patch face is its cell's.
std::expected<void, std::string> writeFields(const fs::path& dir, const PolyMesh& pm, const FaceMesh& fm,
                                             const FvIncompSetup& s, const FvIncompState& st)
{
    const std::size_t nC = std::size_t(s.nC), nB = std::size_t(s.nB), nI = std::size_t(s.nI);
    std::vector<double> U(3 * nC), p(nC);
    for (std::size_t c = 0; c < nC; ++c) {
        const std::size_t src = std::size_t(fm.srcCell[c]);
        for (std::size_t q = 0; q < 3; ++q) U[3 * src + q] = st.U[q * nC + c];
        p[src] = st.p[c];
    }
    // boundary face b -> its slot in polyMesh patch-local order
    std::map<std::string, std::pair<std::vector<double>, std::vector<double>>> vals;
    std::map<std::string, std::int64_t> start;
    std::int64_t at = pm.man.nInternalFaces;
    for (const auto& pa : pm.man.patches) {
        start[pa.name] = at;
        at += pa.nFaces;
        vals[pa.name] = {std::vector<double>(3 * std::size_t(pa.nFaces)), std::vector<double>(std::size_t(pa.nFaces))};
    }
    for (std::size_t b = 0; b < nB; ++b) {
        const std::size_t fb = std::size_t(s.bFace[b]);
        const auto pit = std::ranges::find_if(fm.patches, [&](const MeshPatch& m) {
            return std::size_t(m.first) <= fb && fb < std::size_t(m.first + m.count);
        });
        auto& v = vals[pit->name];
        const std::size_t local = std::size_t(fm.srcFace[nI + fb] - start[pit->name]);
        for (std::size_t q = 0; q < 3; ++q) v.first[3 * local + q] = st.Ub[q * nB + b];
        v.second[local] = st.p[std::size_t(s.bOwn[b])];
    }
    std::vector<FoamPatchOut> pu, pp;
    for (const auto& pa : pm.man.patches) {
        FoamPatchOut a{pa.name, {}, {}}, c{pa.name, {}, {}};
        if (pa.type == FoamPatchType::Empty || pa.type == FoamPatchType::Cyclic) {
            a.constraint = c.constraint = pa.type == FoamPatchType::Empty ? "empty" : "cyclic";
        } else {
            a.values = std::move(vals[pa.name].first);
            c.values = std::move(vals[pa.name].second);
        }
        pu.push_back(std::move(a));
        pp.push_back(std::move(c));
    }
    if (auto r = writeFoamField(dir, "U", 3, false, "[0 1 -1 0 0 0 0]", U, pu, 12); !r) return r;
    return writeFoamField(dir, "p", 1, false, "[0 2 -2 0 0 0 0]", p, pp, 12);
}

} // namespace

std::expected<RunStats, RunError> runFvIncomp(const FvIncompRun& r, const FaceMesh& fm, const FvIncompBcs& b,
                                              const std::vector<double>& U0, FvIncompState* final)
{
    auto setup = buildFvIncompSetup(fm, b);
    if (!setup) return fail(3, setup.error());
    const FvIncompSetup& s = *setup;
    const auto g = la::graphOf(fm);
    const std::string label = fvIncompLabel(r.ctl, r.meshName);
    const bool cdOn = b.area > 0;
    const double qRef = 0.5 * b.uRef * b.uRef * b.area;

    // --- phase 3c: outputs before the first device allocation ---
    std::ofstream fStats;
    if (r.stats) {
        const std::string path = r.outPrefix + "_stats.csv";
        std::error_code ec;
        if (const fs::path dir = fs::path(path).parent_path(); !dir.empty()) fs::create_directories(dir, ec);
        fStats.open(path, std::ios::trunc);
        if (!fStats) return fail(4, path + ": cannot create");
        fStats << "# " << label << "\n";
        fStats << "t,dt,K,cd,cl,cdf,cfl,maxDiv,pcgIters,resid,epsVisc,rmsDiv\n";
    }

    auto ds = DeviceFvIncomp::make(s, g, r.ctl, U0);
    if (!ds) return fail(6, ds.error());
    if (!r.quiet) {
        std::printf("# %s\n# FV path: %d cells, %d internal faces, %d boundary faces, max non-orthogonality %.2f deg, "
                    "%s pressure\n",
                    label.c_str(), s.nC, s.nI, s.nB, s.nonOrthMaxDeg, s.pureNeumann ? "pure Neumann" : "fixed-value");
        std::fflush(stdout);
    }

    RunStats st;
    double t = 0, dt = 0, nextSeries = 0;
    long unconverged = 0, solves = 0;
    int lastIters = 0;
    TgvSeries series;
    if (r.seriesDt > 0) TgvSeries::header();
    auto probeSeries = [&]() -> std::expected<void, RunError> {
        auto d = ds->diagnostics();
        if (!d) return fail(7, d.error());
        series.row(t, d->K / r.seriesVolume, d->eps / r.seriesVolume);
        return {};
    };
    const auto wall0 = std::chrono::steady_clock::now();
    while (t < r.tEnd) {
        if (stopRequested()) {
            std::fprintf(stderr, "dns: %s at t = %g\n", stopReason(), t);
            break;
        }
        if (r.seriesDt > 0 && t >= nextSeries) {
            if (auto e = probeSeries(); !e) return std::unexpected(e.error());
            nextSeries += r.seriesDt;
        }
        auto dtS = ds->stableDt(r.cfl);
        if (!dtS) return fail(7, dtS.error());
        if (!(*dtS > 0) || !std::isfinite(*dtS))
            return fail(7, "stable dt is not finite and positive (the state diverged)");
        // Equal steps to tEnd, never one short step: DeviceFvIncomp::step's precondition.
        const double rem = r.tEnd - t;
        if (rem <= kArrived * *dtS) break;
        dt = rem / std::ceil(rem / *dtS);
        auto lg = ds->step(dt);
        if (!lg) return fail(7, std::format("t = {:g}: {}", t, lg.error()));
        t += dt;
        ++st.steps;
        lastIters = lg->iterations;
        unconverged += lg->unconverged;
        solves += lg->solves;
        if (st.steps % r.statsEvery == 0) {
            auto d = ds->diagnostics();
            if (!d) return fail(7, d.error());
            if (!std::isfinite(d->K)) return fail(7, "the kinetic energy is not finite (the state diverged)");
            if (cdOn) {
                auto f = ds->forces();
                if (!f) return fail(7, f.error());
                st.cd = (f->fp[0] + f->fv[0]) / qRef;
                st.cl = (f->fp[1] + f->fv[1]) / qRef;
                st.cdf = f->fv[0] / qRef;
            }
            if (r.stats)
                fStats << std::format("{:.8g},{:.6g},{:.9e},{:.8e},{:.8e},{:.8e},{:.4g},{:.4e},{},{:.4e},{:.9e},{:.4e}\n", t,
                                      dt, d->K, st.cd, st.cl, st.cdf, r.cfl, d->divMax, lastIters, lg->divRatio, d->eps,
                                      d->divRms);
            if (!r.quiet && st.steps % (10 * r.statsEvery) == 0) {
                std::printf("t=%.3f/%.3f dt=%.3e cd=%.4f (cdf=%.4f) cl=%+.4f K=%.5g maxDiv=%.2e pcg=%d\n", t, r.tEnd, dt,
                            st.cd, st.cdf, st.cl, d->K, d->divMax, lastIters);
                std::fflush(stdout);
            }
            if (unconverged > 0) {
                std::fprintf(stderr, "dns: warning t=%.4g: %ld of %ld pressure solves stopped short of --tol\n", t,
                             unconverged, solves);
                unconverged = solves = 0;
            }
        }
    }
    if (r.seriesDt > 0)
        if (auto e = probeSeries(); !e) return std::unexpected(e.error());
    st.tStop = t;
    const double wall = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - wall0).count();
    st.msPerStep = st.steps ? wall / double(st.steps) : 0;
    if (r.stats) {
        fStats.close();
        if (!fStats) return fail(4, r.outPrefix + "_stats.csv: write failed");
    }
    if (r.pm || final) {
        auto hs = ds->state();
        if (!hs) return fail(7, hs.error());
        if (r.pm)
            if (auto w = writeFields(fs::path(r.outPrefix) / std::format("{:g}", t), *r.pm, fm, s, *hs); !w)
                return fail(4, w.error());
        if (final) *final = std::move(*hs);
    }
    return st;
}

} // namespace dns::gpu
