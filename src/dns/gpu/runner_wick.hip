#include "dns/gpu/runner_wick.hpp"

#include "dns/interrupt.hpp"

#include "dns/chem.hpp"
#include "dns/gpu/chem.hpp"
#include "dns/gpu/comp_conv.hpp"
#include "dns/gpu/comp_ghost.hpp"
#include "dns/gpu/comp_nscbc.hpp"
#include "dns/gpu/comp_sink.hpp"
#include "dns/gpu/device_util.hpp"
#include "dns/gpu/internal/frame_scratch.hpp"
#include "dns/gpu/internal/reduce.hpp"
#include "dns/gpu/internal/run_loop.hpp"
#include "dns/gpu/reductions.hpp"

#include <hip/hip_runtime.h>

#include <algorithm>
#include <cassert>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <format>
#include <optional>
#include <vector>

namespace dns::gpu {

using internal::fail;
using internal::finishTiming;

namespace {

// The fuel source: rho += cr S, E += cr S h_v, rho Y_F += cr S yF on each
// band cell (plain stage write; never compensated). Held for the whole step:
// an SSP-RK3 step adds exactly S dt (the weights sum to 1).
__global__ void kWickInject(const int* cells, int m, Real crS, Real crSh,
                            Real crSy, Real* rho, Real* E, Real* ry)
{
    const int q = blockIdx.x * blockDim.x + threadIdx.x;
    if (q >= m) return;
    const int c = cells[q];
    rho[c] += crS;
    E[c] += crSh;
    ry[c] += crSy;
}

// The soot sources' stiffness, FP64 from the conserved state: the depletion
// eigenvalues k_f (fuel -> soot) + k_ox rho (Y_O + s Y_s) (soot and
// oxidizer), k = A e^(-Ta/T). The runner takes kChemDt / max J.
struct SootRate
{
    const Real *rho, *mx, *my, *mz, *E, *rs, *ro;
    double gamma, Rgas, lnAf, Taf, lnAox, Taox, s;
    __host__ __device__ double operator()(int c) const
    {
        const double r = rho[c];
        const double ke = 0.5 * (double(mx[c]) * mx[c] + double(my[c]) * my[c] +
                                 double(mz[c]) * mz[c]) / r;
        const double p = (gamma - 1.0) * (double(E[c]) - ke);
        const double T = p / (r * Rgas);
        const double ys = fmax(rs[c] / r, 0.0), yo = fmax(ro[c] / r, 0.0);
        const double J = exp(lnAf - Taf / T) +
                         exp(lnAox - Taox / T) * r * (yo + s * ys);
        return isfinite(J) ? J : HUGE_VAL;
    }
};

} // namespace

std::expected<RunStats, RunError> runWick(const Grid& g, const CompState& ic,
                                          std::span<const std::vector<double>> rhoY,
                                          const WickRun& r, RunOutput& out)
{
    assert(g.nz == 1 && r.regions.size() == g.size() &&
           int(rhoY.size()) == r.sp.nY && r.sp.ox == 1 && r.sp.fuel == 0);
    const GpuGrid gg = GpuGrid::from(g);
    const std::size_t n = g.size();
    const int nx = g.nx, ny = g.ny;
    const double h = g.hx, dV = g.cellVolume(), hy = g.hy * g.hz;
    const double T0 = r.T0, gam = r.tr.gamma, cp = r.tr.cp();
    const WickSpec spec = r.spec;
    const bool soot = r.sp.soot >= 0;

    // Masks: solid, the source band.
    std::vector<std::uint8_t> solidHost(n);
    std::vector<int> src;
    for (std::size_t c = 0; c < n; ++c) {
        solidHost[c] = r.regions[c] == std::uint8_t(WickRegion::Solid);
        if (r.regions[c] == std::uint8_t(WickRegion::Source))
            src.push_back(int(c));
    }
    if (src.empty()) return fail(3, "wick: the fuel source band has no cells");
    auto solidDev = DeviceBuffer<std::uint8_t>::make(n);
    auto srcDev = DeviceBuffer<int>::make(src.size());
    if (!solidDev || !srcDev) return fail(6, "GPU allocation failed");
    if (solidDev->upload(solidHost.data()) != hipSuccess ||
        srcDev->upload(src.data()) != hipSuccess)
        return fail(6, "wick mask upload failed");

    auto dev = DeviceComp::make(g, r.tr);
    auto red = ReduceScratch::make();
    if (!dev || !red) return fail(6, "GPU allocation failed");
    SpeciesSetup sp = r.sp;
    sp.solid = solidDev->data();
    if (!dev->enableSpecies(sp)) return fail(6, "species allocation failed");
    auto ds = DeviceCompState::make(n, 0, false, sp.nY);
    if (!ds) return fail(6, "GPU allocation failed");

    // The body: one isothermal ghost wall, the wick at T_wick and the body
    // at T0 (TwallAt, sampled at each ghost's wall point); its pressure image
    // carries the hydrostatic gradient (a resting wall whose "acceleration"
    // is +g x: grad p |wall = -rho g, the conv plates' image). Deep cells:
    // the ambient at the body's mid-height.
    const double xMid = 0.5 * (spec.xBase + spec.xTop);
    const double pDeep = candleAmbientP(T0, r.gravity, T0, xMid);
    const DeviceCompGhost::Sdf sdf = [spec](double x, double y, double) {
        return wickSdf(spec, x, y);
    };
    const double bbox[6] = {spec.xBase, spec.yc - 0.5 * spec.wCandle, -1e30,
                            spec.xTip(), spec.yc + 0.5 * spec.wCandle, 1e30};
    GhostWallThermal wall;
    wall.mode = GhostWallThermal::Mode::Isothermal;
    wall.Twall = T0 * spec.tCandle;
    wall.Rgas = r.tr.Rgas;
    wall.TwallAt = [spec, T0](double x, double y, double) {
        return T0 * wickWallT(spec, x, y);
    };
    const double gx = r.gravity;
    const GhostWallMotion hydro = [gx](double, double, double) {
        return GhostWallKinematics{{0.0, 0.0, 0.0}, {gx, 0.0, 0.0}};
    };
    auto gm = DeviceCompGhost::make(g, sdf, gam, pDeep / (r.tr.Rgas * T0),
                                    pDeep, bbox, wall, &hydro);
    if (!gm) return fail(6, "ghost wall init failed");
    DeviceCompGhost ghost = std::move(*gm);
    {
        std::vector<std::uint8_t> s2(n, 0);
        for (int c : ghost.solidCells()) s2[std::size_t(c)] = 1;
        if (s2 != solidHost)
            return fail(3, "wick: the ghost wall's solid cells disagree with "
                           "the region map");
    }
    // Interface faces (fluid c, solid nb), split wick / body: the conduction
    // tally below. Each pair once, in cell order (deterministic).
    struct WallFace
    {
        int f, s;
        bool wick;
    };
    std::vector<WallFace> faces;
    for (int j = 0; j < ny; ++j)
        for (int i = 0; i < nx; ++i) {
            const std::size_t c = std::size_t(j) * nx + i;
            if (solidHost[c]) continue;
            const int nb[4] = {i > 0 ? int(c) - 1 : -1,
                               i + 1 < nx ? int(c) + 1 : -1,
                               int(std::size_t((j + ny - 1) % ny) * nx + i),
                               int(std::size_t((j + 1) % ny) * nx + i)};
            for (int s : nb)
                if (s >= 0 && solidHost[std::size_t(s)])
                    faces.push_back({int(c), s,
                                     ((s % nx) + 0.5) * h > spec.xTop});
        }
    std::printf("wick candle: %d ghost cells, %d deep, %d fallback, %d "
                "two-sided faces (%d skipped); %zu wall faces; source band "
                "%zu cells\n",
                int(ghost.solidCells().size() - ghost.deepCells().size()),
                int(ghost.deepCells().size()), ghost.nFallback(),
                ghost.nSide(), ghost.nSideSkipped(), faces.size(), src.size());

    DeviceCompNscbc::Config nc;
    nc.uInf = r.uCo;
    nc.pInf = r.pOut;
    nc.gamma = gam;
    nc.etaIn = r.etaIn;
    nc.Rgas = r.tr.Rgas;
    nc.thermalInflow = r.thermalInflow;
    nc.TInf = T0;
    nc.sigma = kDiffOutflowSigma;
    nc.inflowU = r.rowU;
    nc.inflowJet = r.rowJet;
    nc.inflowRho = r.rowTarget;
    nc.hydroLo = r.hydroLo;
    nc.hydroHi = r.hydroHi;
    // The top's backflow relaxes to the ambient density there.
    nc.rhoInf = r.pOut / (r.tr.Rgas * T0);
    auto nscm = DeviceCompNscbc::make(g, nc);
    if (!nscm) return fail(6, "NSCBC init failed");
    DeviceCompNscbc nsc = std::move(*nscm);
    nsc.setFreestream(r.uCo);
    std::optional<DeviceCompSink> snk;
    if (r.sinkNb > 0) {
        auto m = DeviceCompSink::make(
            g, {.nb = r.sinkNb, .alpha = r.sinkAmp,
                .cInf = std::sqrt(gam * r.tr.Rgas * T0)});
        if (!m) return fail(6, "sink init failed");
        snk = std::move(*m);
    }

    // The fuel vapour: Q per unit depth and time, uniform over the band, at
    // rest with the enthalpy cp T_wick.
    const double S = spec.Q / (double(src.size()) * dV);
    const double hV = cp * T0 * spec.tWick;
    dev->setStageHook([&](DeviceCompState& s) { ghost.apply(s); });
    dev->setRhsHook([&](const DeviceCompState& s, DeviceCompState& o,
                        const StageCombine& cmb) {
        nsc.apply(*dev, s, o, cmb);
        addGravity(gg, 0, r.gravity, cmb, false, s, o);
        dev->speciesOpenConsistency(s, o, cmb);
        dev->speciesInflowPin(o);
        if (snk) {
            snk->apply(s, o, cmb);
            snk->applySpecies(s, o, cmb);
        }
        ghost.applyRhs(*dev, s, o, cmb);
        const Real cr = cmb.crAt(0); // uniform step (no lts)
        kWickInject<<<launchBlocks(src.size()), blockSize>>>(
            srcDev->data(), int(src.size()), Real(cr * S), Real(cr * S * hV),
            Real(cr * S * spec.yF), o.rho.data(), o.E.data(),
            o.y[sp.fuel].data());
        checkLastKernel("kWickInject");
    });

    if (ds->upload(ic) != hipSuccess) return fail(6, "IC upload failed");
    for (int k = 0; k < sp.nY; ++k)
        if (ds->uploadSpecies(k, rhoY[std::size_t(k)]) != hipSuccess)
            return fail(6, "IC upload failed");
    ghost.apply(*ds);
    if (auto e = internal::checkDevice("the initial condition"); !e)
        return std::unexpected(e.error());

    const std::vector<std::string> cols = {
        "t",       "dt",       "dtChem",  "dtSoot",  "xTip",    "height",
        "half",    "xBase",    "xSoot",   "Tmax",    "burn",    "fuelInj",
        "fuelInjCum", "fuel",  "sootForm", "sootOx", "soot",    "fuelOut",
        "sootOut", "fuelBal",  "sootBal", "qWick",   "qBody",   "qWickRatio",
        "Tband",   "mass",     "energy",  "pMean",   "sootMax", "fvMax",
        "hits",    "uMax",     "uOutMin"};
    if (auto e = out.declareSeries("stats", cols); !e) return fail(4, e.error());

    std::vector<float> Th, Rh, Uh, Vh, Ph, Wh, YFh, YOh, YSh, WFh, WXh;
    struct Meas
    {
        WickMeasure wm;
        double Tmax = 0, burn = 0, fuel = 0, sootForm = 0, sootOx = 0,
               soot = 0, fuelOut = 0, sootOut = 0, qWick = 0, qBody = 0,
               Tband = 0, mass = 0, energy = 0, pMean = 0, fvMax = 0,
               uMax = 0, uOutMin = HUGE_VAL;
    };
    const double qHeat = r.sp.q;
    auto measure = [&]() -> std::expected<Meas, RunError> {
        ghost.apply(*ds); // idempotent: the step's output left ghosts stale
        dev->updateChem(*ds);
        for (auto* v : {&Th, &Rh, &Uh, &Vh, &Ph, &Wh, &YFh, &YOh}) v->resize(n);
        if (dev->T().download(Th.data()) != hipSuccess ||
            ds->rho.download(Rh.data()) != hipSuccess ||
            dev->u().download(Uh.data()) != hipSuccess ||
            dev->v().download(Vh.data()) != hipSuccess ||
            dev->p().download(Ph.data()) != hipSuccess ||
            dev->omega().download(Wh.data()) != hipSuccess ||
            dev->speciesY(0).download(YFh.data()) != hipSuccess ||
            dev->speciesY(1).download(YOh.data()) != hipSuccess)
            return fail(6, "field download failed");
        if (soot) {
            for (auto* v : {&YSh, &WFh, &WXh}) v->resize(n);
            if (dev->speciesY(sp.soot).download(YSh.data()) != hipSuccess ||
                dev->sootForm().download(WFh.data()) != hipSuccess ||
                dev->sootOx().download(WXh.data()) != hipSuccess)
                return fail(6, "field download failed");
        }
        Meas m;
        m.wm = measureWick(r.diff, spec, YFh, YOh,
                           soot ? std::span<const float>(YSh)
                                : std::span<const float>(),
                           Wh, solidHost, nx, ny, h);
        double pSum = 0, nFluid = 0, tb = 0;
        for (int j = 0; j < ny; ++j)
            for (int i = 0; i < nx; ++i) {
                const std::size_t c = std::size_t(j) * nx + i;
                if (solidHost[c]) continue;
                const double rho = Rh[c], yf = YFh[c];
                const double ys = soot ? double(YSh[c]) : 0.0;
                m.Tmax = std::max(m.Tmax, double(Th[c]));
                m.burn += Wh[c] * dV;
                m.fuel += rho * yf * dV;
                m.mass += rho * dV;
                m.energy += (double(Ph[c]) / (gam - 1) +
                             0.5 * rho * (double(Uh[c]) * Uh[c] +
                                          double(Vh[c]) * Vh[c]) +
                             qHeat * rho * (yf + ys)) *
                            dV;
                if (soot) {
                    m.sootForm += WFh[c] * dV;
                    m.sootOx += WXh[c] * dV;
                    m.soot += rho * ys * dV;
                    m.fvMax = std::max(m.fvMax, rho * ys / kSootDensityBox);
                }
                pSum += Ph[c] / candleAmbientP(T0, r.gravity, T0, (i + 0.5) * h);
                nFluid += 1;
                m.uMax = std::max(m.uMax, std::hypot(double(Uh[c]),
                                                     double(Vh[c])));
                if (r.regions[c] == std::uint8_t(WickRegion::Source))
                    tb += Th[c];
            }
        m.pMean = pSum / nFluid;
        m.Tband = tb / double(src.size());
        // Net species out of the domain: x-hi minus x-lo, rho u Y of the
        // boundary rows (the open faces carry no diffusion).
        for (int j = 0; j < ny; ++j) {
            const std::size_t c0 = std::size_t(j) * nx, cn = c0 + nx - 1;
            const double mo = double(Rh[cn]) * Uh[cn], mi = double(Rh[c0]) * Uh[c0];
            m.fuelOut += (mo * YFh[cn] - mi * YFh[c0]) * hy;
            if (soot) m.sootOut += (mo * YSh[cn] - mi * YSh[c0]) * hy;
            m.uOutMin = std::min(m.uOutMin, double(Uh[cn]));
        }
        // Wall conduction fluid -> body, the Fourier face flux the stage
        // uses: kappa(T_face) (T_fluid - T_ghost)/h per unit area.
        for (const WallFace& f : faces) {
            const double Tf = Th[std::size_t(f.f)], Tg = Th[std::size_t(f.s)];
            const double kap = r.tr.kappa(0.5 * (Tf + Tg));
            const double q = kap * (Tf - Tg) / h * hy; // face area h x 1
            (f.wick ? m.qWick : m.qBody) += q;
        }
        return m;
    };

    // Frames: temp (T - T0), fuel, oxid, omega, and with soot rho Y_s and
    // omega_ox. The mask (<prefix>_mask.bin) carries the body.
    const int nch = soot ? 6 : 4;
    const float tempScale = float((r.diff.sigma - 1) * T0);
    internal::FrameScratch fs;
    auto writeFrame = [&](double t) {
        if (!fs.ensure(n, nch, false)) return;
        ghost.apply(*ds);
        dev->updateChem(*ds);
        fs.take(0, dev->T(), T0);
        fs.take(1, dev->speciesY(0));
        fs.take(2, dev->speciesY(1));
        fs.take(3, dev->omega());
        if (soot) {
            fs.take(4, ds->y[sp.soot]);
            fs.take(5, dev->sootOx());
        }
        const FrameField ff[6]{{"temp", fs.channel(0), 0, tempScale, false},
                               {"fuel", fs.channel(1), 0, 1.0f, false},
                               {"oxid", fs.channel(2), 0, 1.0f, false},
                               {"omega", fs.channel(3), 0, 1.0f, false},
                               {"soot", fs.channel(4), 0, 0.01f, false},
                               {"sootox", fs.channel(5), 0, 1.0f, false}};
        fs.emit(out, t, std::span(ff, std::size_t(nch)));
    };

    RunStats st;
    double t = 0.0, injCum = 0.0;
    internal::StatsCadence cad{r.statsEvery};
    double dtStab = 0.0, dtChem = 0.0, dtSoot = HUGE_VAL;
    int sinceDt = 1 << 30;
    bool bindChem = false;
    const double Tmid = 0.5 * (1 + r.diff.sigma) * T0;
    double lastHits = 0;
    const auto wall0 = std::chrono::steady_clock::now();
    auto sootDt = [&]() -> double {
        if (!soot) return HUGE_VAL;
        const SootRate f{ds->rho.data(), ds->mx.data(), ds->my.data(),
                         ds->mz.data(),  ds->E.data(),  ds->y[sp.soot].data(),
                         ds->y[sp.ox].data(), gam, r.tr.Rgas, sp.sootLnAf,
                         sp.sootTaf, sp.sootLnAox, sp.sootTaox, sp.stoich};
        const double J = internal::reduce(*red, n, f,
                                          rocprim::maximum<double>(), 0.0);
        return J > 0 ? kChemDt / J : HUGE_VAL;
    };
    auto emitRow = [&](double tt, double dt, const Meas& m) {
        const double fuelBal = (m.burn + m.sootForm + m.fuelOut) / spec.Q;
        const double sootBal =
            m.sootForm > 0 ? (m.sootOx + m.sootOut) / m.sootForm : NAN;
        const double row[] = {tt,          dt,           dtChem,
                              dtSoot,      m.wm.xTip,    m.wm.height,
                              m.wm.half,   m.wm.xBase,   m.wm.xSoot,
                              m.Tmax,      m.burn,       spec.Q,
                              injCum,      m.fuel,       m.sootForm,
                              m.sootOx,    m.soot,       m.fuelOut,
                              m.sootOut,   fuelBal,      sootBal,
                              m.qWick,     m.qBody,      m.qWick / (spec.Q * qHeat),
                              m.Tband,     m.mass,       m.energy,
                              m.pMean,     m.wm.sootMax, m.fvMax * 1e6,
                              m.wm.hits,   m.uMax,       m.uOutMin};
        out.row("stats", row);
    };
    {
        auto m0 = measure();
        if (!m0) return std::unexpected(m0.error());
        emitRow(0.0, 0.0, *m0);
    }
    if (out.wantsAnyFrame(0.0)) writeFrame(0.0);

    while (t < r.tEnd) {
        if (stopRequested()) break;
        if (auto e = internal::recomputeDt(
                sinceDt, dtStab, r.dtRecompute, r.dtMargin, 1e-9,
                [&] {
                    const double a = dev->stableDt(*ds, *red, r.cfl);
                    dtChem = chemStableDt(*dev, *ds, *red);
                    dtSoot = sootDt();
                    bindChem = std::min(dtChem, dtSoot) < a;
                    return std::min({a, dtChem, dtSoot});
                });
            !e)
            return fail(7, e.error());
        const double dt = std::min(r.tEnd - t, dtStab);
        dev->step(*ds, Real(dt));
        injCum += spec.Q * dt;
        t += dt;
        ++st.steps;

        if (cad.due(st.steps, t, r.tEnd)) {
            auto m = measure();
            if (!m) return std::unexpected(m.error());
            if (auto e = internal::checkDevice("a stats row", &*red); !e)
                return std::unexpected(e.error());
            if (!std::isfinite(m->mass) || !std::isfinite(m->energy) ||
                !std::isfinite(m->Tmax))
                return fail(7, std::format("wick candle state is not finite at "
                                           "t={:.4g} (diverged)", t));
            emitRow(t, dt, *m);
            lastHits = m->wm.hits;
            if (cad.printDue(st.steps, t, r.tEnd)) {
                std::printf(
                    "t=%.3f/%.3f H=%.4g half=%.4g base=%.4g Tmax/Tad=%.4f "
                    "fuelBal=%.4f burn/Q=%.4f",
                    t, r.tEnd, m->wm.height, m->wm.half,
                    m->wm.xBase - spec.xTop, m->Tmax / (r.diff.sigma * T0),
                    (m->burn + m->sootForm + m->fuelOut) / spec.Q,
                    m->burn / spec.Q);
                if (soot)
                    std::printf(" soot: form/Q=%.4f ox/form=%.4f out/form=%.3g "
                                "top=%.4g fv=%.3gppm",
                                m->sootForm / spec.Q,
                                m->sootForm > 0 ? m->sootOx / m->sootForm : 0.0,
                                m->sootForm > 0 ? m->sootOut / m->sootForm : 0.0,
                                m->wm.xSoot - spec.xTop, m->fvMax * 1e6);
                std::printf(" qWick/(Qq)=%.4f |u|max=%.4g hits=%g dt=%.2e%s\n",
                            m->qWick / (spec.Q * qHeat), m->uMax, m->wm.hits,
                            dt, bindChem ? " (chem dt)" : "");
                std::fflush(stdout);
                cad.printed(st.steps);
            }
            cad.logged(st.steps);
            if (t > 2 && m->Tmax < Tmid)
                return fail(8, std::format("wick candle went out at t={:.4g}: "
                                           "Tmax = {:.4g} T0 below the mid "
                                           "temperature {:.4g} T0 (raise "
                                           "--damkohler or --fuel-rate)",
                                           t, m->Tmax / T0, Tmid / T0));
        }
        if (out.wantsAnyFrame(t)) writeFrame(t);
    }
    if (auto e = internal::checkDevice("the end of the run", &*red); !e)
        return std::unexpected(e.error());
    if (lastHits > 0)
        std::fprintf(stderr,
                     "dns: warning: %g fluid cells with Z, Y_F, Y_O outside "
                     "[-1e-6, 1+1e-6] or Y_s < -1e-6 at the last stats row "
                     "(counted, never clamped)\n",
                     lastHits);
    std::printf("label: %s\n", r.label.c_str());
    finishTiming(st, wall0, out);
    st.tStop = t;
    return st;
}

} // namespace dns::gpu
