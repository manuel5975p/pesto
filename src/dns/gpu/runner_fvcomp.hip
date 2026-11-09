#include "dns/gpu/runner_fvcomp.hpp"

#include "dns/foamio.hpp"
#include "dns/fvturb.hpp"

#include "dns/gpu/chem.hpp"
#include "dns/gpu/hip_runtime.hpp"
#include "dns/gpu/internal/checkpoint.hpp"
#include "dns/gpu/internal/frame_scratch.hpp"
#include "dns/gpu/internal/run_loop.hpp"
#include "dns/interrupt.hpp"
#include "dns/lowmach.hpp"
#include "dns/steady_stop.hpp"

#include <algorithm>
#include <cassert>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <format>
#include <numeric>
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

// One field's patches of a --foam-write time: B's values, each patch in source face order;
// `empty` and `wedge` by their constraint. read: a boundary state's value(s), nc per face.
template <class Read>
std::vector<FoamPatchOut> foamPatches(const FaceMesh& fm, const std::vector<FvPrim<double>>& bnd, int nc, Read&& read)
{
    std::vector<FoamPatchOut> o;
    for (const MeshPatch& m : fm.patches) {
        FoamPatchOut q{.name = m.name, .constraint = {}, .values = {}};
        if (m.type == FoamPatchType::Empty || m.type == FoamPatchType::Wedge) {
            q.constraint = m.type == FoamPatchType::Empty ? "empty" : "wedge";
            o.push_back(std::move(q));
            continue;
        }
        std::vector<int> order(std::size_t(m.count));
        std::iota(order.begin(), order.end(), m.first);
        std::ranges::sort(order, {}, [&](int b) { return fm.srcFace[std::size_t(fm.nI + b)]; });
        q.values.resize(order.size() * std::size_t(nc));
        for (std::size_t i = 0; i < order.size(); ++i) read(bnd[std::size_t(order[i])], q.values.data() + i * std::size_t(nc));
        o.push_back(std::move(q));
    }
    return o;
}

// --foam-write: DIR/<t>/{p,T,U,rho} of the state s, source cell order, 17 digits, boundary
// values from B (bnd: per FaceMesh boundary face). tools/fv/foamdiff.py reads the cells.
std::expected<void, std::string> writeFoamTime(const FaceMesh& fm, const dns::FvState& s, const std::vector<FvPrim<double>>& bnd, double gamma,
                                               double Rgas, const std::string& prefix, double t)
{
    const std::size_t n = std::size_t(fm.nC);
    std::vector<double> p(n), T(n), rho(n), U(3 * n);
    for (std::size_t c = 0; c < n; ++c) {
        const std::size_t o = std::size_t(fm.srcCell[c]);
        const FvPrim<double> q = fvPrimOf(s.var(0)[c], s.var(1)[c], s.var(2)[c], s.var(3)[c], s.var(4)[c], gamma, Rgas);
        p[o] = q.p;
        T[o] = q.temp;
        rho[o] = q.rho;
        for (std::size_t k = 0; k < 3; ++k) U[3 * o + k] = q.u[k];
    }
    const std::filesystem::path dir = std::filesystem::path(prefix + "_foam") / std::format("{:.12g}", t);
    std::error_code ec;
    std::filesystem::create_directories(dir, ec);
    if (ec) return std::unexpected(std::format("{}: {}", dir.string(), ec.message()));
    const auto scalar = [&](auto&& f) { return foamPatches(fm, bnd, 1, [&](const FvPrim<double>& b, double* o) { o[0] = f(b); }); };
    if (auto r = writeFoamField(dir, "p", 1, false, "[1 -1 -2 0 0 0 0]", p, scalar([](const FvPrim<double>& b) { return b.p; }), 17); !r) return r;
    if (auto r = writeFoamField(dir, "T", 1, false, "[0 0 0 1 0 0 0]", T, scalar([](const FvPrim<double>& b) { return b.temp; }), 17); !r) return r;
    if (auto r = writeFoamField(dir, "rho", 1, false, "[1 -3 0 0 0 0 0]", rho, scalar([](const FvPrim<double>& b) { return b.rho; }), 17); !r) return r;
    const auto uB = foamPatches(fm, bnd, 3, [](const FvPrim<double>& b, double* o) { std::ranges::copy(b.u, o); });
    return writeFoamField(dir, "U", 3, false, "[0 1 -1 0 0 0 0]", U, uB, 17);
}

// runFvComp's loop state: EVERY host variable the time loop carries from one
// step to the next, so the checkpoint's "loop.*" records (fields();
// gpu/internal/checkpoint.hpp). A loop variable outside it breaks the restart
// byte rule silently (gpu_restart_fv is the proof).
struct FvCompLoop
{
    double t = 0.0, dtStab = 0.0, uCur = 0.0, dtFirst = 0.0;
    int sinceDt = 1 << 30;
    long step = 0;
    internal::StatsCadence cad;
    internal::ProfileSchedule prof;
    double dtChem = HUGE_VAL, xfPrev = NAN, tPrev = 0, lastHits = 0, lastMin = 0, lastMax = 1;
    bool bindChem = false;
    double sweep = 0.0; // --turb: X = sum U dt, the box's travel past the inflow (FP64)
    // [thrust] (combscale S1): P0 the IC's six sums; the previous _thrust.csv row's role sums and t
    // (fvThrustWindow's start). Records only with [thrust], so an off checkpoint keeps its records.
    FvRow6 P0{};
    FvRoleSums ledPrev{};
    double tLedPrev = 0.0;
    // --shock hll's sticky dt cadence (internal::shockDtEvery); a record only under the auto cadence.
    bool shockHot = false;
    bool thrustOn, shockOn;

    explicit FvCompLoop(const FvCompRun& r)
        : uCur(r.ramp > 0 ? 0.0 : r.fs.U), cad{r.statsEvery}, prof(r.profileDt), thrustOn(r.thrust.has_value()),
          shockOn(r.shockDtAuto)
    {
    }

    template <class F>
    void fields(F&& f)
    {
        f("t", t);
        f("dtStab", dtStab);
        f("uCur", uCur);
        f("dtFirst", dtFirst);
        f("sinceDt", sinceDt);
        f("steps", step);
        f("cad.lastLog", cad.lastLog);
        f("cad.lastPrint", cad.lastPrint);
        f("prof.next", prof.next);
        f("dtChem", dtChem);
        f("xfPrev", xfPrev);
        f("tPrev", tPrev);
        f("lastHits", lastHits);
        f("lastMin", lastMin);
        f("lastMax", lastMax);
        f("bindChem", bindChem);
        f("sweep", sweep);
        if (thrustOn) {
            f("thrust.P0", P0);
            f("thrust.prevIn", ledPrev.I[0]); // FvRole order In, Exit, Wall, Side (fvnozzle.hpp)
            f("thrust.prevExit", ledPrev.I[1]);
            f("thrust.prevWall", ledPrev.I[2]);
            f("thrust.prevSide", ledPrev.I[3]);
            f("thrust.prevOmega", ledPrev.omega);
            f("thrust.tPrev", tLedPrev);
        }
        if (shockOn) f("shockHot", shockHot);
    }
};

} // namespace

std::expected<RunStats, RunError> runFvComp(const FaceMesh& fm, const dns::FvState& ic, const FvCompRun& r,
                                            const FvRaster& raster, RunOutput& out)
{
    const CheckpointPolicy& cp = r.ckpt;
    assert(!cp.active() || (r.ramp <= 0 && r.steadyTol <= 0));
    auto made = DeviceFvComp::make(fm, r.solver);
    if (!made) return fail(made.error().starts_with("FV device") ? 6 : 3, made.error());
    DeviceFvComp& dev = *made;
    if (auto u = dev.upload(ic); !u) return fail(6, u.error());
    // The solver validated the converged table; a ramp starts from frac 0.
    auto retarget = [&](double frac) -> std::expected<void, RunError> {
        for (std::size_t p = 0; p < fm.patches.size(); ++p) {
            if (fm.patches[p].type == FoamPatchType::Empty) continue;
            if (fm.patches[p].type == FoamPatchType::Wedge) continue;
            const auto it = std::ranges::find(r.spec.patches, fm.patches[p].name, &FvPatchSpec::name);
            if (it->kind != FvPatchKind::Far && it->kind != FvPatchKind::Inflow && it->kind != FvPatchKind::Total) continue;
            if (auto e = dev.setBc(int(p), fvBcOf(*it, r.fs, frac)); !e) return fail(6, e.error());
        }
        return {};
    };
    if (r.ramp > 0)
        if (auto e = retarget(0.0); !e) return std::unexpected(e.error());
    RunStats st;
    FvCompLoop L(r);
    // [thrust]: the ledger from the IC on; P0 the IC's sums (a restart's come from its records).
    std::vector<FvRole> roles;
    FvThrustCase tk{};
    std::vector<std::uint8_t> pastThroat;
    FvRoleSums& ledPrev = L.ledPrev;
    FvRow6& P0 = L.P0;
    double& tLedPrev = L.tLedPrev;
    const int fuelRow = r.solver.chem.nY > 0 ? kFvMeanVars + r.solver.chem.fuel : -1;
    if (r.thrust) {
        roles = fvRoles(r.solver.bc, fm);
        tk = *r.thrust;
        tk.q = r.solver.chem.nY > 0 ? r.solver.chem.q : 0.0;
        tk.exitArea = fvExitArea(fm, roles, tk.axis);
        // The station's storage columns (masterplan_combclose 2.2): Mth's cells, the IC's throat rule.
        if (std::isfinite(tk.station)) {
            assert(r.flame && r.flame->spec().chamber);
            pastThroat = fvPastThroat(fm, *r.flame);
        }
        if (auto e = dev.setLedger(true); !e) return fail(e.error().starts_with("FV device") ? 6 : 3, e.error());
        if (!cp.restart) {
            dns::FvState s0;
            if (auto e = dev.download(s0); !e) return fail(6, e.error());
            P0 = fvTotals6(fm, s0, fuelRow);
        }
    }
    // --restart: the checkpoint's words replace the IC (and the ledger setLedger zeroed), bit for bit.
    if (cp.restart)
        if (auto e = dev.load(*cp.restart); !e) return std::unexpected(e.error());
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
    // The candle: the structured candle's two columns (max |u|, the outflow cells' min u.dir).
    const bool candle = r.jet && r.jet->spec().candle;
    if (candle) cols.insert(cols.end(), {"uMax", "uOutMin"});
    if (rans) {
        cols.emplace_back("mutMax");
        if (ransSst(r.solver.rans.model)) cols.emplace_back("floorHits");
    }
    // TFLES: the structured runner's Omega-weighted mean E and max F E (V-weighted here).
    const bool thick = r.flame && r.solver.tfles.has_value();
    if (thick) cols.insert(cols.end(), {"Ebar", "FEmax"});
    const bool zCols = r.flame && r.premix.has_value();
    if (zCols) cols.insert(cols.end(), {"zMin", "zMax"});
    // --restart: every output back to its checkpointed length (the series then
    // append, without a header).
    if (cp.restart) {
        auto at = outputResume(*cp.restart);
        if (!at) return fail(3, "--restart: " + at.error());
        if (auto e = out.resumeAt(*at); !e) return fail(3, e.error());
    }
    // --turb: the structured 3D flame's surface ratio area = sum V |grad Y_F| / (A dY_F)
    // (dY_F: the inflow Y_F minus the burnt one; the least-squares gradient, central on the
    // uniform box) and I0 = SL/area.
    const bool areaCols = r.flame && r.turbSweep;
    if (areaCols) cols.insert(cols.end(), {"area", "I0"});
    const bool shockCols = r.solver.shock.mode != ShockMode::None;
    if (shockCols) cols.insert(cols.end(), {"shockCells", "byteMax"});
    if (r.nozzle) cols.insert(cols.end(), {"mdotIn", "mdotOut", "Fwall", "Fin", "Fexit", "Cd", "Cf"});
    if (auto e = out.declareSeries("stats", cols, r.label); !e) return fail(4, e.error());
    if (r.thrust)
        if (auto e = out.declareSeries("thrust", fvThrustColumns(std::isfinite(r.thrust->station)), r.label, 17); !e) return fail(4, e.error());
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

    double &t = L.t, &dtStab = L.dtStab, &uCur = L.uCur, &dtFirst = L.dtFirst;
    int& sinceDt = L.sinceDt;
    long& step = L.step;
    internal::StatsCadence& cad = L.cad;
    internal::ProfileSchedule& prof = L.prof;
    double &dtChem = L.dtChem, &xfPrev = L.xfPrev, &tPrev = L.tPrev, &lastHits = L.lastHits, &lastMin = L.lastMin,
           &lastMax = L.lastMax;
    bool& bindChem = L.bindChem;
    // --restart: the loop where the checkpoint left it.
    long steps0 = 0;
    if (cp.restart) {
        if (auto e = internal::loadLoop(*cp.restart, L); !e) return fail(3, "--restart: " + e.error());
        steps0 = step;
    }
    const auto wall0 = std::chrono::steady_clock::now();
    internal::CheckpointSchedule ckpt(cp, t, wall0);
    auto writeCheckpoint = [&]() -> std::expected<void, RunError> {
        Checkpoint ck;
        internal::putHeader(ck, cp, kFvFlameCompSolverName, {fm.nC, fm.nI, fm.nB, r.solver.chem.nY}, t, step);
        internal::saveLoop(ck, L);
        auto at = out.resumePoint();
        if (!at) return fail(4, "checkpoint: " + at.error());
        putOutputResume(ck, *at);
        if (auto e = dev.save(ck); !e) return std::unexpected(e.error());
        if (auto e = ck.write(checkpointPath(cp.prefix)); !e) return fail(4, "checkpoint: " + e.error());
        return {};
    };
    bool stopped = false; // a checkpoint stop: the run is incomplete
    long lastCells = 0;
    // The trailing columns of every stats row: shockCells byteMax (--shock), the nozzle's (F: the
    // tallied wall force). dns flow and dns flame alike.
    auto shockNozzleCols = [&](std::vector<double>& row, const std::array<double, 3>& F) -> std::expected<void, RunError> {
        if (shockCols) {
            auto n = dev.shockCells();
            if (!n) return fail(6, n.error());
            auto m = dev.shockByteMax();
            if (!m) return fail(6, m.error());
            lastCells = *n;
            row.push_back(double(*n));
            row.push_back(double(*m));
        }
        if (r.nozzle) {
            auto mass = dev.patchFlux(0);
            if (!mass) return fail(6, mass.error());
            std::array<std::vector<double>, 3> mom;
            for (int k = 0; k < 3; ++k) {
                auto v = dev.patchFlux(1 + k);
                if (!v) return fail(6, v.error());
                mom[std::size_t(k)] = std::move(*v);
            }
            const FvNozzleRow z = fvNozzleRow(r.solver.bc, *mass, mom, F, *r.nozzle, r.solver.tr.gamma, r.solver.tr.Rgas);
            row.insert(row.end(), {z.mdotIn, z.mdotOut, z.Fwall, z.Fin, z.Fexit, z.Cd, z.Cf});
        }
        return {};
    };
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
            std::vector<double> v = {t,      dt,          dtChem, m.SL,   m.uRef - dxf, m.xf,   m.dth,    m.Tmax, m.mass, m.energy,
                                     m.omegaEdge, m.hits, m.uRef, m.pRef, m.TRef,       m.yMin, m.yMax, m.burn, m.fuel};
            if (thick) {
                auto ts = dev.tflesStats();
                if (!ts) return fail(6, ts.error());
                v.push_back(ts->Ebar);
                v.push_back(ts->FEmax);
            }
            if (zCols) {
                double zMin = HUGE_VAL, zMax = -HUGE_VAL;
                const double *ry = flameState.var(kFvMeanVars), *ro = flameState.var(kFvMeanVars + 1), *rho = flameState.var(0);
                for (std::size_t c = 0; c < std::size_t(flameState.nC); ++c) {
                    const double z = r.premix->mixtureFraction(ry[c] / rho[c], ro[c] / rho[c]);
                    zMin = std::min(zMin, z);
                    zMax = std::max(zMax, z);
                }
                v.insert(v.end(), {zMin, zMax});
            }
            if (areaCols) {
                FvEval ev;
                dns::FvState rr;
                if (auto e = dev.evaluate(ev, rr); !e) return fail(6, e.error());
                const std::size_t nC = std::size_t(fm.nC), k0 = std::size_t(3 * r.flame->spec().fuel);
                double a = 0.0;
                for (std::size_t c = 0; c < nC; ++c)
                    a += fm.V[c] * std::sqrt(ev.gradY[k0][c] * ev.gradY[k0][c] + ev.gradY[k0 + 1][c] * ev.gradY[k0 + 1][c] +
                                             ev.gradY[k0 + 2][c] * ev.gradY[k0 + 2][c]);
                // Normalised by the fuel's jump (C2b: Y_F,rem is burnt; C0: 1).
                const double dY = r.turbPatch->y[std::size_t(r.flame->spec().fuel)] - r.flame->spec().yBurnt;
                const double area = a / (r.flame->area() * dY);
                v.insert(v.end(), {area, m.SL / area});
            }
            std::array<double, 3> F{};
            if (r.nozzle) {
                auto s = dev.surface();
                if (!s) return fail(7, s.error());
                F = {s->fp[0] + s->fv[0], s->fp[1] + s->fv[1], s->fp[2] + s->fv[2]};
            }
            if (auto e = shockNozzleCols(v, F); !e) return e;
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
            // The candle's Far top is its outflow.
            const bool out = k == FvBcKind::Outflow || (candle && k == FvBcKind::Far);
            if (!out && (k == FvBcKind::Far || k == FvBcKind::Inflow)) fuelIn -= (*pf)[p];
            if (out) fuelOut += (*pf)[p];
        }
        lastHits = m.hits;
        lastMin = std::min(m.yfMin, m.yoMin);
        lastMax = m.zMax;
        const FvJetSpec& js = r.jet->spec();
        std::vector<double> v = {t,     dt,      dtChem, m.xTip,  m.xBase, m.half,  m.Tmax,   m.TmaxIn, m.burn,
                                 fuelIn, fuelOut, fuelIn > 0 ? m.burn / fuelIn : 0.0, m.mass, m.energy, m.pMean,
                                 m.zMin, m.zMax,  m.yfMin, m.yoMin, m.hits,  1.0};
        double uOutMin = HUGE_VAL;
        if (candle) {
            // Owners of Outflow / Far faces, ascending face order: u.dir (< 0 = air drawn down).
            for (std::size_t b = 0; b < std::size_t(fm.nB); ++b) {
                const int row = r.solver.bcRow.empty() ? -1 : r.solver.bcRow[b];
                if (row < 0) continue;
                const FvBcKind k = r.solver.bc[std::size_t(row)].kind;
                if (k != FvBcKind::Outflow && k != FvBcKind::Far) continue;
                const std::size_t c = std::size_t(fm.own[std::size_t(fm.nI) + b]);
                const double rho = flameState.var(0)[c];
                double un = 0.0;
                for (int k = 0; k < 3; ++k) un += flameState.var(1 + k)[c] / rho * js.dir[std::size_t(k)];
                uOutMin = std::min(uOutMin, un);
            }
            v.insert(v.end(), {m.uMax, uOutMin});
        }
        out.row("stats", v);
        cad.logged(step);
        if (!r.quiet && cad.printDue(step, t, r.tEnd)) {
            std::printf("t=%.3f/%.3f xTip=%.3f half=%.3f TmaxIn/Tad=%.4f burn/in=%.4f p/p0=%.4f hits=%g dt=%.2e%s", t, r.tEnd,
                        m.xTip, m.half, m.TmaxIn / (js.model.sigma * js.T0), fuelIn > 0 ? m.burn / fuelIn : 0.0, m.pMean,
                        m.hits, dt, bindChem ? " (chem dt)" : "");
            if (candle) std::printf(" |u|max=%.4g uOutMin=%.4g", m.uMax, uOutMin);
            std::printf("\n");
            std::fflush(stdout);
            cad.printed(step);
        }
        const double Tmid = 0.5 * (1.0 + js.model.sigma) * js.T0;
        std::string why;
        if (js.quiescent) return {}; // no fuel: nothing to extinguish
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
        if (auto e = shockNozzleCols(row, F); !e) return e;
        out.row("stats", row);
        return {};
    };

    // [thrust]: one _thrust.csv row, the ledger window since the previous row.
    auto thrustRow = [&](double dt) -> std::expected<void, RunError> {
        FvEval ev;
        dns::FvState rr, s;
        if (auto e = dev.evaluate(ev, rr); !e) return fail(6, e.error());
        if (auto e = dev.download(s); !e) return fail(6, e.error());
        auto led = dev.ledger();
        if (!led) return fail(6, led.error());
        const FvRoleSums now = fvRoleSums(*led, roles, fuelRow);
        const FvExitSample x = fvExitSample(fm, roles, ev, rr, tk);
        const FvThrustWindow w = fvThrustWindow(ledPrev, tLedPrev, now, t, tk);
        // The reacting nozzle's burnt station (masterplan_combnozzle 2.3): host, off the downloaded state.
        const bool station = std::isfinite(tk.station);
        const FvStation b = station ? fvStation(fm, s, tk.axis, tk.station, tk.gamma, tk.Rgas) : FvStation{};
        const FvStorage m = station ? fvStorage(fm, s, tk.axis, tk.station, pastThroat) : FvStorage{};
        out.row("thrust",
                fvThrustRow(t, step, dt, fvTotals6(fm, s, fuelRow), P0, now, w, x, tk, station ? &b : nullptr, station ? &m : nullptr));
        ledPrev = now;
        tLedPrev = t;
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

    double& sweep = L.sweep;
    std::array<std::vector<double>, 3> turbU;
    std::vector<FvBc> turbRows(r.turbSweep ? r.turbSweep->points() : 0);
    // --shock hll: default cadence until a cell is sensed, then every step. Sticky across a
    // restart: FvCompLoop's "loop.shockHot" (under the auto cadence).
    bool& shockHot = L.shockHot;
    double nextFoam = r.foamWrite;
    auto writeFoam = [&]() -> std::expected<void, RunError> {
        dns::FvState s, rr;
        FvEval ev;
        if (auto e = dev.evaluate(ev, rr); !e) return fail(6, e.error());
        if (auto e = dev.download(s); !e) return fail(6, e.error());
        if (auto e = writeFoamTime(fm, s, ev.bnd, r.solver.tr.gamma, r.solver.tr.Rgas, r.foamPrefix, t); !e) return fail(4, e.error());
        return {};
    };
    while (t < r.tEnd) {
        if (stopRequested()) {
            // SIGINT / SIGTERM with checkpoints on: keep this step boundary.
            if (ckpt.stopWorth(t)) {
                if (auto e = writeCheckpoint(); !e) return std::unexpected(e.error());
                (void)ckpt.written(t);
                std::printf("checkpoint: t=%.9g steps=%ld (%s; resume with --restart %s)\n", t, step, stopReason(),
                            cp.prefix.c_str());
                stopped = true;
            }
            break;
        }
        // Collapse is relative: a real mesh's dt has no absolute scale.
        std::string diverged;
        const int dtEvery = internal::shockDtEvery(r.shockDtAuto, r.dtRecompute, sinceDt, shockHot, t, [&] {
            auto n = dev.shockCells();
            return n ? *n : 0L;
        });
        if (auto e = internal::recomputeDt(sinceDt, dtStab, dtEvery, r.dtMargin, 1e-4 * dtFirst, [&] {
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
        // --turb: the box column the inflow faces see at the step's start, held over its
        // stages; the box then sweeps past at U.
        if (r.turbSweep) {
            r.turbSweep->sample(sweep, turbU);
            for (std::size_t i = 0; i < turbRows.size(); ++i)
                turbRows[i] = fvFarTurbRow(*r.turbPatch, r.fs, {turbU[0][i], turbU[1][i], turbU[2][i]});
            if (auto e = dev.setBcRows(r.turbFirst, turbRows); !e) return fail(6, e.error());
        }
        dev.step(dt);
        sweep += r.fs.U * dt;
        if (t < r.ramp) {
            // Quiet start, as the structured O-grid: an exact Galilean boost,
            // then the open patches retarget.
            const double frac = rampFrac(t + dt, r.ramp), uNew = r.fs.U * frac;
            const double du = uNew - uCur;
            if (!r.nozzle) dev.boost({du * dir[0], du * dir[1], du * dir[2]});
            uCur = uNew;
            if (auto e = retarget(frac); !e) return std::unexpected(e.error());
        }
        t += dt;
        ++step;
        if (r.flame) {
            const bool row = cad.due(step, t, r.tEnd), profile = prof.due(t, r.tEnd);
            if (row || profile)
                if (auto e = flameRow(dt, row, profile); !e) return std::unexpected(e.error());
            if (row && r.thrust)
                if (auto e = thrustRow(dt); !e) return std::unexpected(e.error());
        } else if (r.jet) {
            if (cad.due(step, t, r.tEnd))
                if (auto e = jetRow(dt); !e) return std::unexpected(e.error());
        } else if (cad.due(step, t, r.tEnd)) {
            if (auto e = statsRow(dt); !e) return std::unexpected(e.error());
            if (r.thrust)
                if (auto e = thrustRow(dt); !e) return std::unexpected(e.error());
            cad.logged(step);
        }
        if (out.wantsAnyFrame(t))
            if (auto e = writeFrame(t); !e) return std::unexpected(e.error());
        if (r.foamWrite > 0 && (t >= nextFoam - 1e-12 * r.tEnd || t >= r.tEnd)) {
            if (auto e = writeFoam(); !e) return std::unexpected(e.error());
            nextFoam += r.foamWrite;
        }
        if (!burning && !r.quiet && cad.printDue(step, t, r.tEnd)) {
            std::printf("step %ld t %.5g dt %.4g cd %.6f cl %.6f\n", step, t, dt, st.cd, st.cl);
            std::fflush(stdout);
            cad.printed(step);
        }
        // --checkpoint-dt: at the end of the step (its rows and frame written).
        if (ckpt.due(t, r.tEnd)) {
            if (auto e = writeCheckpoint(); !e) return std::unexpected(e.error());
            if (ckpt.written(t)) {
                std::printf("checkpoint: t=%.9g steps=%ld (incomplete; resume with --restart %s)\n", t, step,
                            cp.prefix.c_str());
                stopped = true;
                break;
            }
            std::printf("checkpoint: t=%.9g steps=%ld -> %s\n", t, step, checkpointPath(cp.prefix).c_str());
        }
    }
    // A checkpointed run's .json says whether it reached --tt.
    if (cp.active()) out.setCompletion(!stopped && t >= r.tEnd, cp.restarts);
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
    // ms/step over THIS job's steps (a restart's steps0 came before it).
    finishTiming(st, wall0, out, steps0);
    return st;
}

} // namespace dns::gpu
