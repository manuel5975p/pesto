#include "dns/gpu/runner_chamber.hpp"

#include "dns/interrupt.hpp"

#include "dns/gpu/chem.hpp"
#include "dns/gpu/comp_ghost.hpp"
#include "dns/gpu/comp_nscbc.hpp"
#include "dns/gpu/comp_sink.hpp"
#include "dns/gpu/device_util.hpp"
#include "dns/gpu/internal/frame_scratch.hpp"
#include "dns/gpu/internal/plane_sums.hpp"
#include "dns/gpu/internal/run_loop.hpp"
#include "dns/gpu/reductions.hpp"

#include <hip/hip_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <format>
#include <numbers>
#include <optional>
#include <vector>

namespace dns::gpu {

using internal::fail;
using internal::finishTiming;

namespace {

// The injector: rho += cr S, E += cr S hIn, rhoY_fuel += cr S on each band
// cell (plain stage write; the chamber never runs compensated).
__global__ void kInject(const int* cells, int m, Real crS, Real crSh,
                        Real* rho, Real* E, Real* ry)
{
    const int q = blockIdx.x * blockDim.x + threadIdx.x;
    if (q >= m) return;
    const int c = cells[q];
    rho[c] += crS;
    E[c] += crSh;
    ry[c] += crS;
}

// Per column i and region r (kChamberRegions), FP64 sums over the column's
// fluid cells, in j order (deterministic): see Sum.
enum Sum : int
{
    sN = 0, sRho, sRY, sOmega, sEnergy, sP, sP0, sMx, sMxY, sH0, sMxM, sHits,
    sTmax, kSums
};

__global__ void kChamberColumns(GpuGrid g, const std::uint8_t* region,
                                const Real* rho, const Real* E, const Real* ry,
                                const Real* T, const Real* omega,
                                const Real* u, const Real* v, const Real* p,
                                double q, double cp, double gamma, double Rg,
                                double* out)
{
    const int i = int(std::size_t(blockIdx.x) * blockDim.x + threadIdx.x);
    if (i >= g.nx) return;
    double s[kChamberRegions][kSums] = {};
    for (int j = 0; j < g.ny; ++j) {
        const std::size_t c = gidx(g, i, j, 0);
        const int r = region[c];
        if (r == 0) continue;
        double* a = s[r];
        const double rr = rho[c], y = ry[c], uu = u[c], vv = v[c];
        const double TT = T[c], pp = p[c];
        const double k2 = uu * uu + vv * vv;
        const double mx = rr * uu;
        const double Y = y / rr;
        a[sN] += 1;
        a[sRho] += rr;
        a[sRY] += y;
        a[sOmega] += omega[c];
        a[sEnergy] += double(E[c]) + q * y;
        a[sP] += pp;
        a[sP0] += pp + 0.5 * rr * k2;
        a[sMx] += mx;
        a[sMxY] += mx * Y;
        a[sH0] += mx * (cp * TT + 0.5 * k2);
        a[sMxM] += mx * sqrt(k2 / (gamma * Rg * TT));
        a[sHits] += (Y < -1e-6 || Y > 1 + 1e-6) ? 1.0 : 0.0;
        a[sTmax] = fmax(a[sTmax], TT);
    }
    for (int r = 0; r < kChamberRegions; ++r)
        for (int v2 = 0; v2 < kSums; ++v2)
            out[(std::size_t(i) * kChamberRegions + r) * kSums + v2] = s[r][v2];
}

} // namespace

double chamberRamp(double t, double ramp)
{
    if (!(ramp > 0) || t >= ramp) return 1.0;
    const double s = std::sin(0.5 * std::numbers::pi * t / ramp);
    return s * s;
}

std::expected<RunStats, RunError> runChamber(const Grid& g, const CompState& ic,
                                             const std::vector<double>& rhoY,
                                             const ChamberRun& r,
                                             RunOutput& out)
{
    assert(g.nz == 1 && r.regions.size() == g.size() && r.sp.nY == 1);
    const GpuGrid gg = GpuGrid::from(g);
    const std::size_t n = g.size();
    const double dV = g.cellVolume();
    const double p0 = r.Tu; // rho_u = R = 1
    const double gam = r.tr.gamma, cp = r.tr.cp();

    // Region + solid masks on the device; the injector cell list.
    std::vector<std::uint8_t> solidHost(n);
    std::vector<int> inj;
    for (std::size_t c = 0; c < n; ++c) {
        solidHost[c] = r.regions[c] == std::uint8_t(ChamberRegion::Solid);
        if (r.regions[c] == std::uint8_t(ChamberRegion::Injector))
            inj.push_back(int(c));
    }
    if (inj.empty()) return fail(3, "chamber: the injector band has no fluid cells");
    auto solidDev = DeviceBuffer<std::uint8_t>::make(n);
    auto regionDev = DeviceBuffer<std::uint8_t>::make(n);
    auto injDev = DeviceBuffer<int>::make(inj.size());
    if (!solidDev || !regionDev || !injDev) return fail(6, "GPU allocation failed");
    if (solidDev->upload(solidHost.data()) != hipSuccess ||
        regionDev->upload(r.regions.data()) != hipSuccess ||
        injDev->upload(inj.data()) != hipSuccess)
        return fail(6, "chamber mask upload failed");

    auto dev = DeviceComp::make(g, r.tr);
    auto red = ReduceScratch::make();
    if (!dev || !red) return fail(6, "GPU allocation failed");
    SpeciesSetup sp = r.sp;
    sp.solid = solidDev->data();
    if (!dev->enableSpecies(sp)) return fail(6, "species allocation failed");
    auto ds = DeviceCompState::make(n, 0, false, sp.nY);
    if (!ds) return fail(6, "GPU allocation failed");

    // The chamber wall: one immersed body, adiabatic, at rest. Deep cells
    // hold the ambient (rho 1, p0).
    const ChamberSpec spec = r.spec;
    const DeviceCompGhost::Sdf sdf = [spec](double x, double y, double) {
        return chamberSdf(spec, x, y);
    };
    const double bbox[6] = {spec.x0, spec.yc - 0.5 * spec.Hc - spec.tw, -1e30,
                            spec.xe(), spec.yc + 0.5 * spec.Hc + spec.tw, 1e30};
    auto gm = DeviceCompGhost::make(g, sdf, gam, 1.0, p0, bbox);
    if (!gm) return fail(6, "ghost wall init failed");
    DeviceCompGhost ghost = std::move(*gm);
    {
        std::vector<std::uint8_t> s2(n, 0);
        for (int c : ghost.solidCells()) s2[std::size_t(c)] = 1;
        if (s2 != solidHost)
            return fail(3, "chamber: the ghost wall's solid cells disagree "
                           "with the region map (sdf evaluated off-centre?)");
    }
    std::printf("chamber wall: %d ghost cells, %d deep, %d fallback, %d "
                "two-sided faces (%d skipped); injector %zu cells\n",
                int(ghost.solidCells().size() - ghost.deepCells().size()),
                int(ghost.deepCells().size()), ghost.nFallback(), ghost.nSide(),
                ghost.nSideSkipped(), inj.size());

    DeviceCompNscbc::Config nc;
    nc.rhoInf = 1.0;
    nc.uInf = r.uCo;
    nc.pInf = p0;
    nc.gamma = gam;
    nc.etaIn = r.etaIn;
    nc.Rgas = r.tr.Rgas;
    nc.thermalInflow = r.thermalInflow;
    nc.TInf = r.Tu;
    auto nscm = DeviceCompNscbc::make(g, nc);
    if (!nscm) return fail(6, "NSCBC init failed");
    DeviceCompNscbc nsc = std::move(*nscm);
    nsc.setFreestream(r.uCo);
    std::optional<DeviceCompSink> snk;
    if (r.sinkNb > 0) {
        auto m = DeviceCompSink::make(
            g, {.nb = r.sinkNb, .alpha = r.sinkAmp,
                .cInf = std::sqrt(gam * r.tr.Rgas * r.Tu)});
        if (!m) return fail(6, "sink init failed");
        snk = std::move(*m);
    }

    // The injection rate of the current step (mass per unit time and depth).
    double rateNow = 0;
    const double perCell = 1.0 / (double(inj.size()) * dV);
    const double hIn = cp * r.Tu; // enthalpy of the injected gas at rest
    dev->setStageHook([&](DeviceCompState& s) { ghost.apply(s); });
    dev->setRhsHook([&](const DeviceCompState& s, DeviceCompState& o,
                        const StageCombine& cmb) {
        nsc.apply(*dev, s, o, cmb);
        dev->speciesOpenConsistency(s, o, cmb);
        if (snk) {
            snk->apply(s, o, cmb);
            snk->applySpecies(s, o, cmb);
        }
        ghost.applyRhs(*dev, s, o, cmb);
        if (rateNow > 0) {
            const double S = rateNow * perCell;
            const Real cr = cmb.crAt(0); // uniform step (no lts)
            kInject<<<launchBlocks(inj.size()), blockSize>>>(
                injDev->data(), int(inj.size()), Real(cr * S),
                Real(cr * S * hIn), o.rho.data(), o.E.data(),
                o.y[sp.fuel].data());
            checkLastKernel("kInject");
        }
    });

    if (ds->upload(ic) != hipSuccess || ds->uploadSpecies(0, rhoY) != hipSuccess)
        return fail(6, "initial condition upload failed");
    ghost.apply(*ds);
    if (auto h = internal::checkDevice("the initial condition"); !h)
        return std::unexpected(h.error());

    const std::string cols[] = {
        "t",     "dt",      "inj",     "injCum",  "fuel",   "burn",
        "burnCav", "outNoz", "outDom", "mdotIn",  "mdotNoz", "Me",
        "pCh",   "p0Noz",   "pAmb",    "h0Noz",   "Tmax",   "mass",
        "energy", "yHits",  "omegaInj", "fx"};
    if (auto e = out.declareSeries("stats", cols); !e) return fail(4, e.error());

    DeviceBuffer<double> colScratch;
    std::vector<double> ps;
    struct Meas
    {
        double fuel = 0, burn = 0, burnCav = 0, outNoz = 0, outDom = 0,
               mdotNoz = 0, Me = 0, pCh = 0, p0Noz = 0, pAmb = 0, h0Noz = 0,
               Tmax = 0, mass = 0, energy = 0, yHits = 0, omegaInj = 0, fx = 0;
    };
    const double hy = g.hy * g.hz; // face area of an x face
    auto measure = [&]() -> std::expected<Meas, RunError> {
        ghost.apply(*ds); // idempotent: the step's output left ghosts stale
        const auto F = ghost.force(*dev, *ds);
        dev->updateChem(*ds);
        const std::size_t m = std::size_t(g.nx) * kChamberRegions * kSums;
        if (internal::ensurePlaneScratch(colScratch, m) != hipSuccess)
            return fail(6, "chamber stats scratch failed");
        kChamberColumns<<<launchBlocks(std::size_t(g.nx)), blockSize>>>(
            gg, regionDev->data(), ds->rho.data(), ds->E.data(),
            ds->y[0].data(), dev->T().data(), dev->omega().data(),
            dev->u().data(), dev->v().data(), dev->p().data(), sp.q, cp, gam,
            r.tr.Rgas, colScratch.data());
        checkLastKernel("kChamberColumns");
        if (internal::fetchPlaneScratch(colScratch, m, ps) != hipSuccess)
            return fail(6, "chamber stats download failed");
        auto at = [&](int i, int reg, int q) {
            return ps[(std::size_t(i) * kChamberRegions + reg) * kSums + q];
        };
        Meas M;
        double reg[kChamberRegions][kSums] = {};
        for (int i = 0; i < g.nx; ++i)
            for (int k = 1; k < kChamberRegions; ++k)
                for (int q = 0; q < kSums; ++q) {
                    if (q == sTmax) reg[k][q] = std::max(reg[k][q], at(i, k, q));
                    else reg[k][q] += at(i, k, q);
                }
        double wInj = 0;
        for (int k = 1; k < kChamberRegions; ++k) {
            M.fuel += reg[k][sRY] * dV;
            M.burn += reg[k][sOmega] * dV;
            M.mass += reg[k][sRho] * dV;
            M.energy += reg[k][sEnergy] * dV;
            M.yHits += reg[k][sHits];
            M.Tmax = std::max(M.Tmax, reg[k][sTmax]);
            if (inCavity(std::uint8_t(k))) M.burnCav += reg[k][sOmega] * dV;
        }
        wInj = reg[int(ChamberRegion::Injector)][sOmega] * dV;
        const auto& ex = reg[int(ChamberRegion::Exit)];
        M.mdotNoz = ex[sMx] * hy;
        M.outNoz = ex[sMxY] * hy;
        M.h0Noz = ex[sH0] * hy;
        M.Me = ex[sMx] != 0 ? ex[sMxM] / ex[sMx] : NAN;
        auto mean = [&](ChamberRegion k, int q) {
            const auto& a = reg[int(k)];
            return a[sN] > 0 ? a[q] / a[sN] : NAN;
        };
        M.pCh = mean(ChamberRegion::FreshProbe, sP);
        M.p0Noz = mean(ChamberRegion::NozzleProbe, sP0);
        M.pAmb = mean(ChamberRegion::AmbientProbe, sP);
        double xhi = 0, xlo = 0;
        for (int k = 1; k < kChamberRegions; ++k) {
            xhi += at(g.nx - 1, k, sMxY);
            xlo += at(0, k, sMxY);
        }
        M.outDom = (xhi - xlo) * hy;
        M.omegaInj = M.burn > 0 ? wInj / M.burn : 0;
        M.fx = F.fx;
        return M;
    };

    const float tempScale = float((r.sigma - 1) * r.Tu);
    internal::FrameScratch fs;
    auto writeFrame = [&](double t) {
        if (!fs.ensure(n, 3, false)) return;
        dev->updateChem(*ds);
        fs.take(0, dev->T(), r.Tu);
        fs.take(1, dev->speciesY(0));
        fs.take(2, dev->omega());
        const FrameField ff[3]{{"temp", fs.channel(0), 0, tempScale, false},
                               {"fuel", fs.channel(1), 0, 1.0f, false},
                               {"omega", fs.channel(2), 0, 1.0f, false}};
        fs.emit(out, t, ff);
    };

    RunStats st;
    double t = 0.0, injCum = 0.0;
    internal::StatsCadence cad{r.statsEvery};
    double dtStab = 0.0, dtChem = 0.0;
    int sinceDt = 1 << 30;
    bool bindChem = false;
    const double Tb = r.sigma * r.Tu;
    const double uInj = r.mdot / spec.Hc; // fresh bulk speed (rho_u = 1)
    const double tFlash = r.ramp + 2.0, tBlow = r.ramp + 2.0 * spec.Lc / uInj;
    double lastHits = 0;
    const auto wall0 = std::chrono::steady_clock::now();

    auto emitRow = [&](double tt, double dt, const Meas& m) {
        const double inj = r.mdot * chamberRamp(tt, r.ramp);
        const double row[] = {tt,        dt,         inj,       injCum,
                              m.fuel,    m.burn,     m.burnCav, m.outNoz,
                              m.outDom,  inj,        m.mdotNoz, m.Me,
                              m.pCh,     m.p0Noz,    m.pAmb,    m.h0Noz,
                              m.Tmax,    m.mass,     m.energy,  m.yHits,
                              m.omegaInj, m.fx};
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
                    bindChem = dtChem < a;
                    return std::min(a, dtChem);
                });
            !e)
            return fail(7, e.error());
        const double dt = std::min(r.tEnd - t, dtStab);
        rateNow = r.mdot * chamberRamp(t, r.ramp);
        dev->step(*ds, Real(dt));
        injCum += rateNow * dt; // Y_in = 1: fuel mass = injected mass
        t += dt;
        ++st.steps;

        if (cad.due(st.steps, t, r.tEnd)) {
            auto m = measure();
            if (!m) return std::unexpected(m.error());
            if (auto h = internal::checkDevice("a stats row", &*red); !h)
                return std::unexpected(h.error());
            if (!std::isfinite(m->mass) || !std::isfinite(m->energy) ||
                !std::isfinite(m->fuel) || !std::isfinite(m->pCh))
                return fail(7, std::format("chamber state is not finite at "
                                           "t={:.4g} (diverged)", t));
            emitRow(t, dt, *m);
            lastHits = m->yHits;
            const double inj = r.mdot * chamberRamp(t, r.ramp);
            if (cad.printDue(st.steps, t, r.tEnd)) {
                std::printf("t=%.3f/%.3f burn/inj=%.4f outNoz/inj=%.4f "
                            "pCh/pAmb=%.5f Me=%.4f Tmax/Tb=%.4f omegaInj=%.2e "
                            "yHits=%g dt=%.2e%s\n",
                            t, r.tEnd, inj > 0 ? m->burn / inj : 0.0,
                            inj > 0 ? m->outNoz / inj : 0.0, m->pCh / m->pAmb,
                            m->Me, m->Tmax / Tb, m->omegaInj, m->yHits, dt,
                            bindChem ? " (chem dt)" : "");
                std::fflush(stdout);
                cad.printed(st.steps);
            }
            cad.logged(st.steps);
            if (r.reaction && t > tFlash && m->omegaInj > 1e-3)
                return fail(8, std::format(
                                   "flash-back: {:.3g} of the burn sits in "
                                   "the injector band at t={:.4g} (raise "
                                   "--inject-speed)",
                                   m->omegaInj, t));
            if (r.reaction && t > tBlow && m->outNoz > 0.5 * inj)
                return fail(8, std::format(
                                   "blow-off: {:.3g} of the injected fuel "
                                   "leaves through the nozzle at t={:.4g} "
                                   "(lower --inject-speed or raise "
                                   "--holder-height)",
                                   m->outNoz / inj, t));
        }
        if (out.wantsAnyFrame(t)) writeFrame(t);
    }
    if (auto h = internal::checkDevice("the end of the run", &*red); !h)
        return std::unexpected(h.error());
    if (lastHits > 0)
        std::fprintf(stderr,
                     "dns: warning: %g cells with Y outside [-1e-6, 1+1e-6] "
                     "at the last stats row (counted, never clamped)\n",
                     lastHits);
    std::printf("label: %s\n", r.label.c_str());
    finishTiming(st, wall0, out);
    st.tStop = t;
    return st;
}

} // namespace dns::gpu
