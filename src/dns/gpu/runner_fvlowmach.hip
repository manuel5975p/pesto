#include "dns/gpu/runner_fvlowmach.hpp"

#include "dns/gpu/fv_incomp.hpp"
#include "dns/gpu/hip_runtime.hpp"
#include "dns/gpu/internal/checkpoint.hpp"
#include "dns/gpu/internal/frame_scratch.hpp"
#include "dns/gpu/internal/run_loop.hpp"
#include "dns/interrupt.hpp"
#include "dns/la/graph.hpp"
#include "dns/lowmach.hpp"

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

namespace {

// runFvLowMach's loop state: EVERY host variable the time loop carries from one
// step to the next, so the checkpoint's "loop.*" records (fields();
// gpu/internal/checkpoint.hpp). A loop variable outside it breaks the restart
// byte rule silently (gpu_restart_fv is the proof).
struct FvLowMachLoop
{
    double t = 0, dtStab = 0, dtFirst = 0, xfPrev = NAN, tPrev = 0, lastHits = 0, lastMin = 0, lastMax = 1;
    FvLowMachDt bound;
    int sinceDt = 1 << 30;
    long step = 0, unconverged = 0, solves = 0;
    int lastIters = 0;
    bool bindChem = false;
    internal::StatsCadence cad;
    internal::ProfileSchedule prof;

    explicit FvLowMachLoop(const FvLowMachRun& r) : cad{r.statsEvery}, prof(r.profileDt) {}

    template <class F>
    void fields(F&& f)
    {
        f("t", t);
        f("dtStab", dtStab);
        f("dtFirst", dtFirst);
        f("xfPrev", xfPrev);
        f("tPrev", tPrev);
        f("lastHits", lastHits);
        f("lastMin", lastMin);
        f("lastMax", lastMax);
        f("bound.conv", bound.conv);
        f("bound.diff", bound.diff);
        f("bound.chem", bound.chem);
        f("bound.pos", bound.pos);
        f("sinceDt", sinceDt);
        f("steps", step);
        f("unconverged", unconverged);
        f("solves", solves);
        f("lastIters", lastIters);
        f("bindChem", bindChem);
        f("cad.lastLog", cad.lastLog);
        f("cad.lastPrint", cad.lastPrint);
        f("prof.next", prof.next);
    }
};

} // namespace

std::expected<RunStats, RunError> runFvLowMach(const FaceMesh& fm, const FvLowMachIC& ic, FvLowMachRun r,
                                               const FvRaster& raster, RunOutput& out)
{
    if (auto b = bindFvIncompBcs(r.bcs.flow, fm); !b) return fail(3, b.error());
    auto setup = buildFvIncompSetup(fm, r.bcs.flow);
    if (!setup) return fail(3, setup.error());
    FvLowMachTables tab = buildFvLowMachTables(fm, r.bcs, *setup, r.model.prGas);
    const FvLowMachRun::Burner& bu = r.burner;
    const bool open = bu.on && !tab.bOpen.empty(); // gpu/runner_lowmach's open columns
    if (bu.on) {
        const std::size_t nI = std::size_t(fm.nI);
        auto n = fvLowMachRows(fm, bu.patch, *setup, tab, r.model.prGas, [&](std::size_t q) {
            const std::size_t f = nI + std::size_t(setup->bFace[q]);
            return bu.row({fm.Cf[0][f], fm.Cf[1][f], fm.Cf[2][f]}, fm.srcCell[std::size_t(setup->bOwn[q])]);
        });
        if (!n) return fail(3, n.error());
    }
    const FvIncompSetup& s = *setup;
    const auto graph = la::graphOf(fm);
    // --turb: the inflow patch's faces in the setup's boundary numbering q.
    std::vector<int> turbQ;
    std::optional<TurbSweep> turbSweep;
    if (r.turb) {
        const auto it = std::ranges::find(fm.patches, r.turbPatch, &MeshPatch::name);
        if (it == fm.patches.end()) return fail(3, "--turb: no inflow patch " + r.turbPatch);
        std::vector<std::array<double, 3>> pts;
        for (int q = 0; q < s.nB; ++q) {
            const int b = s.bFace[std::size_t(q)];
            if (b < it->first || b >= it->first + it->count) continue;
            const std::size_t f = std::size_t(fm.nI + b);
            turbQ.push_back(q);
            pts.push_back({fm.Cf[0][f], fm.Cf[1][f], fm.Cf[2][f]});
        }
        turbSweep = TurbSweep::make(r.turb->modes, r.turb->spec, pts);
    }

    std::vector<std::string> cols = {"t",      "dt",      "dtChem",   "dtDiff", "SL",     "SLdisp", "xf",    "dth",
                                     "Tmax",   "mass",    "massTally", "omegaEdge", "yHits", "uFresh", "pFresh", "TFresh",
                                     "yMin",   "yMax",    "pJump",    "szDefect", "divErr", "eosErr", "pcgIt",
                                     "burn",   "fuel",    "eDefect"};
    if (r.premix2) cols.insert(cols.end(), {"dzMin", "dzMax"});
    if (bu.on) {
        // gpu/runner_lowmach's burner columns, same names (2D slot, closed laterals).
        cols = {"t",     "dt",      "dtChem", "dtDiff",   "xTip",    "xBase",  "tanA",   "tanL",      "tanR",
                "rms",   "sinRatio", "SLflank", "mFlank",  "burn",    "fuelIn", "fuelNom", "burnRatio", "Tmax",
                "mass",  "massErr", "yHits",  "yMin",     "yMax",    "divErr", "eosErr", "pcgIt",     "uMax"};
        if (r.premix2)
            cols.insert(cols.end(), {"xTipOut", "fuelOut", "burnPre", "burnDif", "preFrac", "zMin", "zMax"});
        if (open) {
            cols.insert(cols.end(), {"mdotIn", "mdotOut", "mdotLat", "uLatMin", "uLatMax"});
            if (!r.premix2) cols.push_back("fuelOut");
        }
        if (bu.rim) cols.insert(cols.end(), {"xRimBase", "rBase", "qRim"});
    }
    if (r.posK > 0) cols.push_back("dtPos");
    // M4: the eddy fields (DeviceFvLowMach::eddyStats), gpu/runner_lowmach's columns.
    const bool lesOn = r.model.cs > 0, thickOn = r.model.tfles.has_value();
    if (lesOn) cols.push_back("mutMax");
    if (thickOn) cols.insert(cols.end(), {"FEmax", "Ebar"});
    if (thickOn)
        if (auto w = fvLowMachTfles(fm, tab); !w) return fail(3, w.error());
    const CheckpointPolicy& ckp = r.ckpt;
    // --restart: every output back to its checkpointed length (the series then
    // append, without a header).
    if (ckp.restart) {
        auto at = outputResume(*ckp.restart);
        if (!at) return fail(3, "--restart: " + at.error());
        if (auto e = out.resumeAt(*at); !e) return fail(3, e.error());
    }
    if (auto e = out.declareSeries("stats", cols, r.label); !e) return fail(4, e.error());
    const std::string pcols[] = {"t", "x", "T", "rho", "u", "p", "Y", "omega"};
    if (!bu.on)
        if (auto e = out.declareSeries("profiles", pcols); !e) return fail(4, e.error());

    auto made = DeviceFvLowMach::make(s, tab, graph, r.model, r.ctl, ic);
    if (!made) return fail(6, made.error());
    DeviceFvLowMach& dev = *made;
    // --restart: the checkpoint's words (state, p, phi, tallies) replace the IC's.
    if (ckp.restart)
        if (auto e = dev.load(*ckp.restart); !e) return std::unexpected(e.error());
    auto eddyCols = [&](std::vector<double>& row) -> std::expected<void, RunError> {
        if (!lesOn && !thickOn) return {};
        auto e = dev.eddyStats();
        if (!e) return fail(6, e.error());
        if (lesOn) row.push_back(e->mutMax);
        if (thickOn) row.insert(row.end(), {e->feMax, e->eBar});
        return {};
    };

    if (!r.quiet) {
        std::printf("# %s\n# FV path: %d cells, %d internal faces, %d boundary faces, max non-orthogonality %.2f deg\n",
                    r.label.c_str(), s.nC, s.nI, s.nB, s.nonOrthMaxDeg);
        std::fflush(stdout);
    }

    const FvFlameProbe& probe = *r.flame;
    const FvFlameSpec& fl = probe.spec();
    const double Hin = fl.Tu + r.model.qcp * r.yFresh;
    const Grid& rg = raster.grid();
    std::vector<float> chan0(rg.size()), chan1(rg.size()), chan2(rg.size());

    RunStats st;
    FvLowMachLoop L(r);
    double &t = L.t, &dtStab = L.dtStab, &dtFirst = L.dtFirst, &xfPrev = L.xfPrev, &tPrev = L.tPrev,
           &lastHits = L.lastHits, &lastMin = L.lastMin, &lastMax = L.lastMax;
    FvLowMachDt& bound = L.bound;
    int& sinceDt = L.sinceDt;
    long &step = L.step, &unconverged = L.unconverged, &solves = L.solves;
    int& lastIters = L.lastIters;
    bool& bindChem = L.bindChem;
    internal::StatsCadence& cad = L.cad;
    internal::ProfileSchedule& prof = L.prof;
    // --restart: the loop where the checkpoint left it.
    long steps0 = 0;
    if (ckp.restart) {
        if (auto e = internal::loadLoop(*ckp.restart, L); !e) return fail(3, "--restart: " + e.error());
        steps0 = step;
    }
    const auto wall0 = std::chrono::steady_clock::now();
    internal::CheckpointSchedule ckpt(ckp, t, wall0);
    auto writeCheckpoint = [&]() -> std::expected<void, RunError> {
        Checkpoint ck;
        internal::putHeader(ck, ckp, kFvLowMachSolverName, {fm.nC, fm.nI, fm.nB, r.model.nY}, t, step);
        internal::saveLoop(ck, L);
        auto at = out.resumePoint();
        if (!at) return fail(4, "checkpoint: " + at.error());
        putOutputResume(ck, *at);
        if (auto e = dev.save(ck); !e) return std::unexpected(e.error());
        if (auto e = ck.write(checkpointPath(ckp.prefix)); !e) return fail(4, "checkpoint: " + e.error());
        return {};
    };
    bool stopped = false; // a checkpoint stop: the run is incomplete
    double sweep = 0;
    std::array<std::vector<double>, 3> turbU;
    std::vector<double> turbBU, turbRU;

    auto measured = [&](FvLowMachCells& c, std::vector<double>& ry) {
        ry = c.ry[0];
        return probe.measureLowMach({.rho = &c.rho, .ry = &ry, .ru = &c.ru, .T = &c.T, .p = &c.p, .omega = &c.omega});
    };
    auto writeFrame = [&]() -> std::expected<void, RunError> {
        auto c = dev.cells();
        if (!c) return fail(6, c.error());
        std::vector<double> dT(c->T), Y(c->T.size());
        std::ranges::for_each(dT, [&](double& v) { v -= fl.Tu; });
        for (std::size_t q = 0; q < Y.size(); ++q) Y[q] = c->ry[0][q] / c->rho[q];
        raster.sample(dT, chan0);
        raster.sample(Y, chan1);
        raster.sample(c->omega, chan2);
        const FrameField ff[3] = {{"temp", chan0.data(), 0, float((fl.sigma - 1.0) * fl.Tu), false},
                                  {"fuel", chan1.data(), 0, 1.0f, false},
                                  {"omega", chan2.data(), 0, float(1.0 / fl.dth), false}};
        if (auto e = out.frame(t, ff); !e) return fail(4, e.error());
        return {};
    };
    auto row = [&](double dt, bool stats, bool profile) -> std::expected<void, RunError> {
        auto c = dev.cells();
        if (!c) return fail(6, c.error());
        std::vector<double> ry;
        const FvFlameMeasure m = measured(*c, ry);
        if (!std::isfinite(m.mass) || !std::isfinite(m.Tmax) || !std::isfinite(m.xf))
            return fail(7, "flame state is not finite or has no mid-temperature crossing (diverged)");
        lastHits = m.hits;
        lastMin = m.yMin;
        lastMax = m.yMax;
        if (stats) {
            auto to = dev.totals();
            if (!to) return fail(6, to.error());
            double sz = 0;
            for (std::size_t q = 0; q < c->T.size(); ++q)
                sz = std::max(sz, std::abs(c->T[q] + r.model.qcp * c->ry[0][q] / c->rho[q] - Hin));
            const double dxf = std::isfinite(xfPrev) && t > tPrev ? (m.xf - xfPrev) / (t - tPrev) : 0.0;
            std::vector<double> v = {t,        dt,          bound.chem,      bound.diff, m.SL,     m.uRef - dxf, m.xf,
                                     m.dth,    m.Tmax,      m.mass,          to->massTally, m.omegaEdge, m.hits, m.uRef,
                                     m.pRef,   m.TRef,      m.yMin,          m.yMax,     m.pRef - m.slabs.back().p, sz,
                                     to->divErr, to->eosErr, double(lastIters), m.burn,  m.fuel,   to->enthalpy - to->enthalpyTally};
            if (r.premix2) {
                double zMin = HUGE_VAL, zMax = -HUGE_VAL;
                for (std::size_t q = 0; q < c->T.size(); ++q) {
                    const double z = (r.zS * (c->ry[0][q] / c->rho[q]) - c->ry[1][q] / c->rho[q] + r.zO2) / r.zDen;
                    zMin = std::min(zMin, z);
                    zMax = std::max(zMax, z);
                }
                v.insert(v.end(), {zMin - r.zIn, zMax - r.zIn});
            }
            if (r.posK > 0) v.push_back(bound.pos);
            if (auto e = eddyCols(v); !e) return std::unexpected(e.error());
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
            std::printf("t=%.3f/%.3f SL=%.6f xf=%.4f dth=%.4f Tmax/Tb=%.5f yHits=%g dt=%.2e pcg=%d%s\n", t, r.tEnd, m.SL,
                        m.xf, m.dth, m.Tmax / (fl.sigma * fl.Tu), m.hits, dt, lastIters, bindChem ? " (chem dt)" : "");
            std::fflush(stdout);
            cad.printed(step);
        }
        const double guard = 2.0 * fl.dth;
        if (m.xf < guard || m.xf > probe.length() - guard)
            return fail(8, std::format("flame not stabilized: drifted to the {} end at t={:.4g} (x_f={:.4g}; adjust "
                                       "--inflow-speed)",
                                       m.xf < guard ? "inflow" : "outflow", t, m.xf));
        return {};
    };

    // --- --mode bunsen: the cone on the measurement plane (pixel = cell on the box) ---
    std::optional<FvRaster> plane;
    int pnx = 0, pny = 0;
    double floorXi = 0;
    if (bu.on) {
        const double hiX = *std::ranges::max_element(fm.C[0]) + 0.5 * bu.h;
        const double hiY = *std::ranges::max_element(fm.C[1]) + 0.5 * bu.h;
        pnx = int(std::lround((hiX - bu.lo[0]) / bu.h));
        pny = int(std::lround((hiY - bu.lo[1]) / bu.h));
        if (pnx < 4 || pny < 4) return fail(3, "--mode bunsen: the mesh does not cover the burner's measurement plane");
        auto p = FvRaster::make(fm, {.longSide = std::max(pnx, pny),
                                     .box = {bu.lo[0], bu.lo[0] + pnx * bu.h, bu.lo[1], bu.lo[1] + pny * bu.h}});
        if (!p) return fail(3, p.error());
        if (p->grid().nx != pnx || p->grid().ny != pny) return fail(3, "--mode bunsen: measurement plane size");
        plane = std::move(*p);
        floorXi = std::ldexp(1.0, -24) / bu.h;
    }
    const double Tmid = 0.5 * (1 + fl.sigma) * fl.Tu;
    // The fuel the premixed flame burns per unit fresh mass: C0 1, C2 Y_F1 - Y_F,rem.
    const double dYPre = r.premix2 ? bu.premix.yF1() - bu.premix.yFrem() : 1.0;
    std::vector<std::uint8_t> inPatch(std::size_t(s.nB), 0), outPatch(std::size_t(s.nB), 0),
        rimPatch(std::size_t(s.nB), 0);
    for (std::size_t q = 0; q < std::size_t(s.nB); ++q) {
        const std::size_t fb = std::size_t(s.bFace[q]);
        for (const MeshPatch& mp : fm.patches)
            if (std::size_t(mp.first) <= fb && fb < std::size_t(mp.first + mp.count)) {
                inPatch[q] = bu.on && mp.name == bu.patch;
                rimPatch[q] = bu.rim && mp.name == "rim";
            }
        outPatch[q] = s.bCode[q] == std::uint8_t(FvIncompFaceBc::Outflow);
    }
    const double ambRho = r.model.prGas / tab.ambT, cp = r.model.tr.cp();
    auto burnerRow = [&](double dt) -> std::expected<void, RunError> {
        auto c = dev.cells();
        if (!c) return fail(6, c.error());
        auto to = dev.totals();
        if (!to) return fail(6, to.error());
        auto fx = dev.faces();
        if (!fx) return fail(6, fx.error());
        const std::size_t nC = std::size_t(fm.nC);
        std::vector<double> xi;
        if (r.premix2) {
            auto x = dev.flameIndex(floorXi);
            if (!x) return fail(6, x.error());
            xi = std::move(*x);
        }
        const PartialPremixModel& pm = bu.premix;
        double Tmax = 0, mass = 0, burn = 0, burnPre = 0, burnDif = 0, hits = 0, uMax = 0;
        double yMin = HUGE_VAL, yMax = -HUGE_VAL, zMin = HUGE_VAL, zMax = -HUGE_VAL;
        std::vector<double> wc(nC), u(nC), v(nC), z(r.premix2 ? nC : 0);
        for (std::size_t q = 0; q < nC; ++q) {
            const double rho = c->rho[q], V = fm.V[q], om = c->omega[q], yf = c->ry[0][q] / rho;
            Tmax = std::max(Tmax, c->T[q]);
            mass += V * rho;
            burn += V * om;
            yMin = std::min(yMin, yf);
            yMax = std::max(yMax, yf);
            hits += (yf < -1e-6 || yf > 1 + 1e-6) ? 1.0 : 0.0;
            u[q] = c->ru[q] / rho;
            v[q] = c->ru[nC + q] / rho;
            const double w = c->ru[2 * nC + q] / rho;
            uMax = std::max(uMax, std::sqrt(u[q] * u[q] + v[q] * v[q] + w * w));
            wc[q] = om;
            if (r.premix2) {
                const double yo = c->ry[1][q] / rho;
                hits += (yo < -1e-6 || yo > 1 + 1e-6) ? 1.0 : 0.0;
                z[q] = pm.mixtureFraction(yf, yo);
                zMin = std::min(zMin, z[q]);
                zMax = std::max(zMax, z[q]);
                const double wp = lmPremixedWeight(xi[q]);
                wc[q] = om * wp;
                burnPre += V * om * wp;
                burnDif += V * om * (1.0 - wp);
            }
        }
        if (!std::isfinite(mass) || !std::isfinite(Tmax)) return fail(7, "flame state is not finite (diverged)");
        // Boundary fluxes are outward: the inflow's fuel comes in as -F.
        double fuelIn = 0, fuelNom = 0, fuelOut = 0;
        // Open faces (gpu/runner_lowmach's LmBoundarySums): in through the inflows,
        // out through the outflows, net IN through the open patches, and their
        // outward normal velocity m_n / rho_upwind (the ambient's drawing in).
        double mIn = 0, mOut = 0, mLat = 0, uLatMin = HUGE_VAL, uLatMax = -HUGE_VAL, qRim = 0;
        for (std::size_t q = 0; q < std::size_t(s.nB); ++q) {
            if (inPatch[q]) {
                fuelIn -= fx->fy[0][q];
                fuelNom -= fx->phiB[q] * tab.bY[0][q];
            }
            if ((r.premix2 || open) && outPatch[q]) fuelOut += fx->fy[0][q];
            if (rimPatch[q]) qRim -= cp * fx->heat[q];
            if (!open) continue;
            const double ph = fx->phiB[q];
            if (s.bCode[q] == std::uint8_t(FvIncompFaceBc::Inflow)) mIn -= ph;
            if (tab.bOpen[q] == 1) mOut += ph;
            if (tab.bOpen[q] == 2) {
                mLat -= ph;
                const double un = ph / s.bMagS[q] / (ph < 0 ? ambRho : c->rho[std::size_t(s.bOwn[q])]);
                uLatMin = std::min(uLatMin, un);
                uLatMax = std::max(uLatMax, un);
            }
        }
        const double span = bu.depth;
        mass /= span;
        burn /= span;
        burnPre /= span;
        burnDif /= span;
        fuelIn /= span;
        fuelNom /= span;
        fuelOut /= span;
        mIn /= span;
        mOut /= span;
        mLat /= span;
        qRim /= span;
        const double tally = to->massTally / span;
        // The plane.
        const std::size_t np = std::size_t(pnx) * std::size_t(pny);
        std::vector<float> Tp(np), Wp(np), Rp(np), Up(np), Vp(np);
        plane->sample(c->T, Tp);
        plane->sample(wc, Wp);
        plane->sample(c->rho, Rp);
        plane->sample(u, Up);
        plane->sample(v, Vp);
        const double h = bu.h, yc = bu.geom.yc;
        const ConeMeasure cn = measureCone(Tp, pnx, pny, h, Tmid, yc, Wp, int(std::ceil(0.5 * fl.dth / h)));
        const double mFlank = flankMassFlux(cn, Rp, Up, Vp, pnx, h);
        double SLflank = NAN;
        if (cn.ok) {
            double w = 0;
            for (int j = 0; j < pny; ++j)
                for (int i = cn.iLo; i <= cn.iHi; ++i) w += Wp[std::size_t(j) * std::size_t(pnx) + std::size_t(i)];
            const double cosA = 1.0 / std::hypot(1.0, cn.tanA);
            const double len = 2.0 * (cn.iHi - cn.iLo + 1) * h / cosA;
            SLflank = w * h * h / (len * dYPre);
        }
        double xTipOut = NAN;
        if (r.premix2) {
            std::vector<float> Zp(np);
            plane->sample(z, Zp);
            const int jl = int(std::floor(yc / h - 0.5)), jr = jl + 1;
            auto ax = [&](int i) {
                return 0.5 * (double(Zp[std::size_t(jl) * std::size_t(pnx) + std::size_t(i)]) +
                              double(Zp[std::size_t(jr) * std::size_t(pnx) + std::size_t(i)]));
            };
            const double zst = pm.zst();
            if (ax(0) >= zst)
                for (int i = 0; i + 1 < pnx; ++i)
                    if (ax(i) >= zst && ax(i + 1) < zst) {
                        xTipOut = (i + 0.5 + (ax(i) - zst) / (ax(i) - ax(i + 1))) * h;
                        break;
                    }
        }
        double rBase = NAN; // --rim: gpu/runner_lowmach's base-row walk on the cone plane
        if (bu.rim && std::isfinite(cn.xBase)) {
            const int iB = int(std::floor(cn.xBase / h));
            const int jl = int(std::floor(yc / h - 0.5)), jr = jl + 1;
            auto at = [&](int j) { return double(Tp[std::size_t(j) * std::size_t(pnx) + std::size_t(iB)]); };
            double yR = NAN, yL = NAN;
            for (int j = jr; j + 1 < pny; ++j)
                if (at(j) < Tmid && at(j + 1) >= Tmid) {
                    yR = (j + 0.5 + (Tmid - at(j)) / (at(j + 1) - at(j))) * h;
                    break;
                }
            for (int j = jl; j - 1 >= 0; --j)
                if (at(j) < Tmid && at(j - 1) >= Tmid) {
                    yL = (j + 0.5 - (Tmid - at(j)) / (at(j - 1) - at(j))) * h;
                    break;
                }
            rBase = 0.5 * (yR - yL);
        }
        const double sinA = cn.ok ? cn.tanA / std::hypot(1.0, cn.tanA) : NAN;
        const double net = r.premix2 || open ? fuelIn - fuelOut : fuelIn;
        lastHits = hits;
        lastMin = yMin;
        lastMax = yMax;
        std::vector<double> row = {t,
                                   dt,
                                   bound.chem,
                                   bound.diff,
                                   cn.xTip,
                                   cn.xBase,
                                   cn.ok ? cn.tanA : NAN,
                                   cn.ok ? cn.tanL : NAN,
                                   cn.ok ? cn.tanR : NAN,
                                   cn.ok ? cn.rms : NAN,
                                   sinA * bu.geom.uJet,
                                   SLflank,
                                   mFlank,
                                   burn,
                                   fuelIn,
                                   fuelNom,
                                   burn / net,
                                   Tmax,
                                   mass,
                                   (mass - tally) / tally,
                                   hits,
                                   yMin,
                                   yMax,
                                   to->divErr,
                                   to->eosErr,
                                   double(lastIters),
                                   uMax};
        if (r.premix2) row.insert(row.end(), {xTipOut, fuelOut, burnPre, burnDif, burnPre / burn, zMin, zMax});
        if (open) {
            row.insert(row.end(), {mIn, mOut, mLat, uLatMin, uLatMax});
            if (!r.premix2) row.push_back(fuelOut);
        }
        const double x0 = bu.rim ? bu.geom.xRim : 0.0;
        if (bu.rim) row.insert(row.end(), {cn.xBase - x0, rBase, qRim});
        if (r.posK > 0) row.push_back(bound.pos);
        if (auto e = eddyCols(row); !e) return std::unexpected(e.error());
        out.row("stats", row);
        cad.logged(step);
        if (!r.quiet && cad.printDue(step, t, r.tEnd)) {
            std::printf("t=%.3f/%.3f tip=%.3f base=%.3f sin(a)U/SL=%.5f SLflank=%.5f mFlank=%.5f burn/in=%.4f "
                        "Tmax/Tb=%.5f |u|max=%.3f dt=%.2e pcg=%d%s",
                        t, r.tEnd, cn.xTip, cn.xBase, sinA * bu.geom.uJet, SLflank, mFlank, burn / net,
                        Tmax / (fl.sigma * fl.Tu), uMax, dt, lastIters, bindChem ? " (chem dt)" : "");
            if (r.premix2) std::printf(" tipOut=%.3f preFrac=%.4f", xTipOut, burnPre / burn);
            std::printf("\n");
            std::fflush(stdout);
            cad.printed(step);
        }
        // gpu/runner_lowmach's anchoring guards; --rim: heights above the rim.
        const double guard = 2.0 * fl.dth, Lx = pnx * h;
        std::string why;
        if (cn.axisHot)
            why = "the slot axis is hot at the inflow (flash-back)";
        else if (bu.rim && std::isfinite(cn.xBase) && cn.xBase < x0)
            why = std::format("the base x={:.4g} sits below the rim x={:.4g} (flash-back into the tube)", cn.xBase, x0);
        else if (!std::isfinite(cn.xTip))
            why = "the slot axis never reaches Tmid (blow-through)";
        else if (cn.xTip > Lx - guard)
            why = std::format("the tip x={:.4g} reached the outflow (blow-through)", cn.xTip);
        else if (!std::isfinite(cn.xBase) || cn.xBase - x0 > 0.5 * (cn.xTip - x0))
            why = std::format("the base x={:.4g} lifted past half the tip height {:.4g}{} (blow-off)", cn.xBase, cn.xTip,
                              bu.rim ? " above the rim" : "");
        else if (!cn.ok)
            why = std::format("fewer than 4 burning flank rows between base {:.4g} and tip {:.4g} (the cone opened: "
                              "blow-off)",
                              cn.xBase, cn.xTip);
        if (!why.empty())
            return fail(8, std::format("flame not anchored at t={:.4g}: {} (adjust --jet-speed, {})", t, why,
                                       bu.rim ? "--rim, --rim-temp, --coflow-speed" : "--pilot-*, --coflow-temp"));
        return {};
    };

    while (t < r.tEnd) {
        if (stopRequested()) {
            // SIGINT / SIGTERM with checkpoints on: keep this step boundary.
            if (ckpt.stopWorth(t)) {
                if (auto e = writeCheckpoint(); !e) return std::unexpected(e.error());
                (void)ckpt.written(t);
                std::printf("checkpoint: t=%.9g steps=%ld (%s; resume with --restart %s)\n", t, step, stopReason(),
                            ckp.prefix.c_str());
                stopped = true;
            }
            break;
        }
        std::string diverged;
        if (auto e = internal::recomputeDt(sinceDt, dtStab, r.dtRecompute, r.dtMargin, 1e-4 * dtFirst, [&] {
                auto d = dev.stableDt(r.cfl, r.chemK, r.posK);
                if (!d) {
                    diverged = d.error();
                    return -1.0;
                }
                bound = *d;
                const double a = std::min(std::min(d->conv, d->diff), d->pos);
                bindChem = d->chem < a;
                return std::min(a, d->chem);
            });
            !e)
            return fail(7, diverged.empty() ? e.error() : diverged);
        if (dtFirst == 0.0) dtFirst = dtStab;
        // Equal steps to tEnd, never a sliver: the projection's memory of c.
        const double rem = r.tEnd - t;
        if (rem <= 1e-9 * dtStab) break;
        const double dt = rem / std::ceil(rem / dtStab);
        // --turb: the box column the inflow faces see at the step's start, held over its
        // stages; the box then sweeps past at turbU.
        if (turbSweep) {
            turbSweep->sample(sweep, turbU);
            fvLowMachTurbRows(turbQ, turbU, s.bU, tab.rUb, s.bT, r.model.prGas, turbBU, turbRU);
            if (auto e = dev.setInflow(turbBU, turbRU); !e) return fail(6, e.error());
        }
        auto lg = dev.step(dt);
        sweep += r.turbU * dt;
        if (!lg) return fail(7, std::format("t = {:g}: {}", t, lg.error()));
        lastIters = lg->iterations;
        unconverged += lg->unconverged;
        solves += lg->solves;
        t += dt;
        ++step;
        const bool stats = cad.due(step, t, r.tEnd), profile = !bu.on && prof.due(t, r.tEnd);
        if (bu.on) {
            if (stats)
                if (auto e = burnerRow(dt); !e) return std::unexpected(e.error());
        } else if (stats || profile) {
            if (auto e = row(dt, stats, profile); !e) return std::unexpected(e.error());
        }
        if (out.wantsAnyFrame(t))
            if (auto e = writeFrame(); !e) return std::unexpected(e.error());
        // --checkpoint-dt: at the end of the step (its rows and frame written); never
        // at the end (the loop's own end test: no step left).
        const bool atEnd = r.tEnd - t <= 1e-9 * dtStab;
        if (!atEnd && ckpt.due(t, r.tEnd)) {
            if (auto e = writeCheckpoint(); !e) return std::unexpected(e.error());
            if (ckpt.written(t)) {
                std::printf("checkpoint: t=%.9g steps=%ld (incomplete; resume with --restart %s)\n", t, step,
                            ckp.prefix.c_str());
                stopped = true;
                break;
            }
            std::printf("checkpoint: t=%.9g steps=%ld -> %s\n", t, step, checkpointPath(ckp.prefix).c_str());
        }
    }
    // A checkpointed run's .json says whether it reached --tt.
    if (ckp.active()) out.setCompletion(!stopped && r.tEnd - t <= 1e-9 * dtStab, ckp.restarts);
    if (unconverged > 0)
        std::fprintf(stderr, "dns: warning: %ld of %ld pressure solves stopped short of --tol\n", unconverged, solves);
    if (lastHits > 0)
        std::fprintf(stderr,
                     "dns: warning: %g cells with Y outside [-1e-6, 1+1e-6] at the last stats row (counted, never "
                     "clamped): Y in [%.3g, 1%+.3g]\n",
                     lastHits, lastMin, lastMax - 1);
    std::printf("label: %s\n", r.label.c_str());
    st.steps = step;
    st.tStop = t;
    // ms/step over THIS job's steps (a restart's steps0 came before it).
    finishTiming(st, wall0, out, steps0);
    return st;
}

} // namespace dns::gpu
