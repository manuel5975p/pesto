#include "dns/gpu/runner_flame.hpp"

#include "dns/interrupt.hpp"

#include "dns/gpu/chem.hpp"
#include "dns/gpu/comp_conv.hpp"
#include "dns/gpu/comp_nscbc.hpp"
#include "dns/gpu/comp_sink.hpp"
#include "dns/gpu/derived.hpp"
#include "dns/gpu/device_util.hpp"
#include "dns/gpu/flame_turb.hpp"
#include "dns/gpu/internal/frame_scratch.hpp"
#include "dns/gpu/internal/run_loop.hpp"
#include "dns/gpu/reductions.hpp"
#include "dns/gpu/tfles.hpp"

#include <hip/hip_runtime.h>

#include <algorithm>
#include <cassert>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <format>
#include <numbers>
#include <optional>
#include <utility>
#include <vector>

namespace dns::gpu {

using internal::fail;
using internal::finishTiming;

std::expected<RunStats, RunError> runFlame(
    const Grid& g, const CompState& ic,
    std::span<const std::vector<double>> rhoY, const FlameRun& r,
    RunOutput& out)
{
    assert(!(r.bunsen && r.diffusion) && int(rhoY.size()) == r.sp.nY);
    // The 2D burners share the per-row inflow and the boundary heat release.
    const bool rows = r.bunsen || r.diffusion;
    assert(r.csLes == 0 || (g.nz > 1 && !r.diffusion));
    assert(!r.tflesOn || !r.diffusion);
    // csLes == 0: DeviceComp::make(g, tr), the DNS solver bit for bit.
    auto dev = DeviceComp::make(g, r.tr, r.csLes, r.sgs);
    auto red = ReduceScratch::make();
    if (!dev || !red) return fail(6, "GPU allocation failed");
    const GpuGrid gg = GpuGrid::from(g);
    // Flame LES: mu_t on the open x rows from their neighbours (the SGS
    // gradient there differences inflow against outflow across the wrap).
    if (r.csLes > 0) dev->setMutHook(mutCopyOpenXHook(gg));
    // TFLES: F E, E/F, Omega on the device, read by the species kernels.
    std::optional<DeviceTfles> tf;
    SpeciesSetup sp = r.sp;
    if (r.tflesOn) {
        auto m = DeviceTfles::make(g, r.tfles);
        if (!m) return fail(6, "TFLES allocation failed");
        tf = std::move(*m);
        sp.tflesFE = tf->fe();
        sp.tflesEoF = tf->eof();
        sp.tflesOm = tf->om();
    }
    if (!dev->enableSpecies(sp)) return fail(6, "species allocation failed");
    auto ds = DeviceCompState::make(g.size(), 0, false, r.sp.nY);
    if (!ds) return fail(6, "GPU allocation failed");

    DeviceCompNscbc::Config nc;
    nc.rhoInf = 1.0;
    nc.uInf = r.uIn;
    nc.pInf = r.pB; // the IC's burnt pressure: the IC is the fixed point
    nc.gamma = r.tr.gamma;
    nc.etaIn = r.etaIn;
    nc.Rgas = r.tr.Rgas;
    nc.thermalInflow = r.thermalInflow;
    nc.TInf = r.Tu;
    nc.sigma = r.outSigma;
    if (rows) {
        nc.inflowU = r.rowU;
        nc.inflowJet = r.rowJet;
        nc.inflowRho = r.rowTarget;
    }
    assert(r.gravity >= 0 && (r.gravity == 0 || r.candle));
    nc.hydroLo = r.hydroLo;
    nc.hydroHi = r.hydroHi;
    // Candle: the top is the ambient column's far field; air drawn back in
    // through it relaxes to the ambient density there (rho_a = p_a/(R T0)).
    if (r.gravity > 0) nc.rhoInf = r.pB / (r.tr.Rgas * r.Tu);
    auto nscm = DeviceCompNscbc::make(g, nc);
    if (!nscm) return fail(6, "NSCBC init failed");
    DeviceCompNscbc nsc = std::move(*nscm);
    nsc.setFreestream(r.uIn);
    // --turb: the frozen box on the device, its rows on the inflow face.
    const bool turb = !r.turbBox[0].empty();
    assert(!turb || (!r.diffusion && g.nz > 1));
    std::optional<DeviceTurbInflow> turbIn;
    if (turb) {
        auto m = DeviceTurbInflow::make(r.turbNbx, g.ny, g.nz, g.hx, r.turbBox);
        if (!m) return fail(6, "turbulence table allocation failed");
        turbIn = std::move(*m);
        nsc.setTurbulentInflow(turbIn->du(), turbIn->dv(), turbIn->dw());
    }
    std::optional<DeviceCompSink> snk;
    if (r.sinkNb > 0) {
        auto m = DeviceCompSink::make(
            g, {.nb = r.sinkNb, .alpha = r.sinkAmp,
                .cInf = std::sqrt(r.tr.gamma * r.tr.Rgas * r.Tu)});
        if (!m) return fail(6, "sink init failed");
        snk = std::move(*m);
    }
    dev->setRhsHook([&](const DeviceCompState& s, DeviceCompState& o,
                        const StageCombine& cmb) {
        nsc.apply(*dev, s, o, cmb);
        // Candle: THE gravity source on every cell, after the overwrite (the
        // NSCBC rows' LODI carries the hydrostatic waves that balance it).
        // It writes momentum and E only, so the species re-carry below still
        // sees the NSCBC density.
        if (r.gravity > 0) addGravity(gg, 0, r.gravity, cmb, false, s, o);
        // The cone burns in the inflow rows: the overwrite dropped their
        // q omega (issues.md §35); the 1D flame keeps its bits. NOT the
        // diffusion jet: its pinned inflow sheet has omega 0, and at the
        // OUTFLOW row the re-added heat ran away (an open tip burning in
        // the LODI row: T 4.3 T_ad at i = nx - 1, p/p0 swinging 0.95-1.12,
        // backflow; coflow 40 l, s 2). There the dropped q omega leaves
        // with the gas (issues.md §35, §48).
        if (r.bunsen) dev->speciesBoundaryHeat(o, cmb);
        dev->speciesOpenConsistency(s, o, cmb);
        // Diffusion: the inflow row holds its Burke-Schumann composition
        // (consistent with the held T; DeviceComp::speciesInflowPin).
        if (r.diffusion) dev->speciesInflowPin(o);
        if (snk) {
            snk->apply(s, o, cmb);
            snk->applySpecies(s, o, cmb);
        }
    });

    if (ds->upload(ic) != hipSuccess)
        return fail(6, "initial condition upload failed");
    for (int k = 0; k < r.sp.nY; ++k)
        if (ds->uploadSpecies(k, rhoY[std::size_t(k)]) != hipSuccess)
            return fail(6, "initial condition upload failed");
    if (auto h = internal::checkDevice("the initial condition"); !h)
        return std::unexpected(h.error());

    const double hx = g.hx, dV = g.cellVolume();
    const int nb = r.sinkNb;
    const double plane = double(g.ny) * g.nz;
    const double Tmid = 0.5 * (1 + r.sigma) * r.Tu;
    const double dTflame = (r.sigma - 1) * r.Tu;

    std::vector<std::string> cols1d = {
        "t",      "dt",     "dtChem",    "SL",    "SLdisp", "xf",
        "dth",    "Tmax",   "mass",      "energy", "omegaEdge", "yHits",
        "uFresh", "pFresh", "TFresh",    "yMin",  "yMax"};
    // 3D (nz > 1): the flame-surface ratio A_T/A (co-area integral of |grad
    // c| over the cross-section), I0 = SL/area (Damkoehler's first
    // hypothesis: 1 at Le = 1), u' over a slab ahead of the brush, the inflow
    // target, the fresh-mass thickness x_I = integral of rho Y dx / (rho_u
    // Y_u) (dx_I/dt = u_in - S_c exactly) and the brush's leading edge (the
    // first plane-mean c >= kLeadC from the inflow). dth is then the
    // plane-mean brush thickness.
    const bool d3 = g.nz > 1 && !rows;
    if (d3)
        cols1d.insert(cols1d.end(), {"area", "I0", "uPrime", "uInflow",
                                     "xFresh", "xLead"});
    // Bunsen: the cone (measureCone) -- tip, base, flank tan(alpha) from the
    // half-width and each side, the fit residual, the kinematic check
    // sin(alpha) U_jet / S_L (1 = the ideal cone), the flank consumption
    // speed (integral of omega over the fit rows / the two flanks' length),
    // the flank's kinematic mass flux rho u.n at Tmid (flankMassFlux); burn =
    // integral of omega, fuelIn = the inflow face's fuel flux (the boundary
    // cells' rho u times the row's Y, as the species face carries it),
    // fuelNom = the profile's nominal rho u Y.
    const std::vector<std::string> colsB = {
        "t",      "dt",      "dtChem",  "xTip",     "xBase",   "tanA",
        "tanL",   "tanR",    "rms",     "sinRatio", "SLflank", "mFlank",
        "burn",   "fuelIn",  "fuelNom", "burnRatio", "Tmax",   "mass",
        "energy", "yHits",   "yMin",    "yMax",     "uJet"};
    // Diffusion: the jet (measureJet) -- the tip where the axis Z drops below
    // Z_st, the anchor height, the Z_st half-width at mid tip; burn =
    // integral of omega, fuelIn / fuelOut = the inflow / outflow faces' rho
    // u Y_F (an open tip leaks fuel: burn = fuelIn - fuelOut when steady);
    // the mixture-fraction and reactant bounds, the out-of-bounds count.
    const std::vector<std::string> colsD = {
        "t",      "dt",      "dtChem",   "xTip",    "xBase", "half",
        "Tmax",   "TmaxIn",  "burn",     "fuelIn",  "fuelOut", "burnRatio",
        "mass",   "energy",  "pMean",    "zMin",    "zMax",  "yfMin",
        "yoMin",  "hits",    "uJet"};
    // Candle: the jet's columns + the box's max |u| and the outflow row's min
    // u (< 0 = backflow through the top).
    std::vector<std::string> colsC = colsD;
    colsC.insert(colsC.end(), {"uMax", "uOutMin"});
    std::vector<std::string> cols = r.candle      ? colsC
                                    : r.diffusion ? colsD
                                    : r.bunsen    ? colsB
                                                  : cols1d;
    // 3D bunsen (nz > 1): the cone is measured on the SPAN-MEAN (z) fields;
    // appended ST = burn / (rho_u Y_u A_cone) (the consumption speed over
    // the span-mean cone's area, A_cone = 2 hypot(xTip - xBase, W/2) L_z),
    // area = integral |grad c| dV / A_cone (the flame-surface ratio) and I0 =
    // ST/area (Damkoehler's first hypothesis). SLflank is then per unit
    // span too.
    const bool b3 = g.nz > 1 && r.bunsen;
    if (b3) cols.insert(cols.end(), {"ST", "area", "I0"});
    // TFLES: the Omega-weighted mean efficiency and the largest F E.
    const std::size_t nBase = cols.size();
    if (tf) cols.insert(cols.end(), {"Ebar", "FEmax"});
    // pMean's reference: p0, or the candle's ambient column averaged over
    // the rows (p_a(x) at the cell centres).
    double pRefMean = r.pB;
    if (r.candle && r.gravity > 0) {
        pRefMean = 0;
        for (int i = 0; i < g.nx; ++i)
            pRefMean += candleAmbientP(r.Tu, r.gravity, r.Tu, (i + 0.5) * g.hx);
        pRefMean /= g.nx;
    }
    // The fresh reference column: 2 delta_th inside the inflow band edge,
    // ~10 delta_th upstream of the flame. The consumption speed is normalised
    // by ITS rho Y (the fresh state the flame actually burns), and the
    // displacement speed uses its velocity.
    const int iRef = std::min(g.nx - 1, nb + int(std::lround(2 * r.dth / hx)));
    if (auto e = out.declareSeries("stats", cols); !e) return fail(4, e.error());
    if (!rows) {
        const std::string pcols[] = {"t", "x", "T", "rho", "u", "p", "Y", "omega"};
        if (auto e = out.declareSeries("profiles", pcols); !e)
            return fail(4, e.error());
    }

    DeviceBuffer<double> colScratch;
    std::vector<double> ps;
    std::vector<float> Th, Rh, Uh, Vh, Wh, YFh, YOh;
    struct Meas
    {
        double SL = 0, xf = 0, dth = 0, Tmax = 0, mass = 0, energy = 0,
               omegaEdge = 0, yHits = 0, uRef = 0, pRef = 0, TRef = 0;
        double yMin = HUGE_VAL, yMax = -HUGE_VAL, burn = 0;
        ConeMeasure cone;
        double SLflank = NAN, mFlank = NAN, fuelIn = 0;
        JetMeasure jet;
        double fuelOut = 0, TmaxIn = 0, pMean = 0;
        double uMax = 0, uOutMin = HUGE_VAL;
        double area = NAN, uPrime = NAN, xFresh = NAN, xLead = NAN;
        double Ebar = NAN, FEmax = NAN; // TFLES
        double ST = NAN;                // 3D bunsen (area: the cone's)
    };
    // 3D bunsen: the span (z) means of the downloaded fields, k ascending in
    // FP64, onto the nx x ny plane measureCone reads.
    std::vector<float> Tm, Rm, Um, Vm, Wm;
    auto spanMean = [&](const std::vector<float>& f, std::vector<float>& m) {
        const std::size_t plane2 = std::size_t(g.nx) * g.ny;
        m.resize(plane2);
        for (std::size_t q = 0; q < plane2; ++q) {
            double s = 0;
            for (int k = 0; k < g.nz; ++k) s += f[q + plane2 * std::size_t(k)];
            m[q] = float(s / g.nz);
        }
    };
    // Round nozzle: the plane through the jet axis z = zc (a cell face, nz
    // even): the mean of the k = nz/2 - 1 and nz/2 planes, in FP64.
    auto axialPlane = [&](const std::vector<float>& f, std::vector<float>& m) {
        const std::size_t plane2 = std::size_t(g.nx) * g.ny;
        const std::size_t k0 = plane2 * std::size_t(g.nz / 2 - 1), k1 = k0 + plane2;
        m.resize(plane2);
        for (std::size_t q = 0; q < plane2; ++q)
            m[q] = float(0.5 * (double(f[q + k0]) + double(f[q + k1])));
    };
    DeviceBuffer<double> tflesScratch;
    std::vector<double> tfs;
    // A stats row: the mode's base columns, then (TFLES) Ebar, FEmax.
    std::vector<double> rowBuf, rowBuf3;
    auto emitRow = [&](std::span<const double> base, double Ebar,
                       double FEmax) {
        if (!tf) {
            out.row("stats", base.first(nBase));
            return;
        }
        rowBuf.assign(base.begin(), base.begin() + std::ptrdiff_t(nBase));
        rowBuf.push_back(Ebar);
        rowBuf.push_back(FEmax);
        out.row("stats", rowBuf);
    };
    DeviceBuffer<double> turbScratch;
    std::vector<double> ts;
    // The uPrime slab ahead of the brush: x_f - 3 l .. x_f - 1.5 l, l = l_t
    // (delta_th without --turb).
    const double lSlab = turb ? r.lt : r.dth;
    auto col = [&](int i, int q) { return ps[std::size_t(i) * kFlameSums + q]; };
    auto measure = [&]() -> std::expected<Meas, RunError> {
        dev->updateChem(*ds);
        if (flameColumnStats(*dev, *ds, colScratch, ps) != hipSuccess)
            return fail(6, "flame column stats download failed");
        Meas m;
        if (tf) {
            if (tf->columnStats(tflesScratch, tfs) != hipSuccess)
                return fail(6, "TFLES column stats download failed");
            double so = 0, soe = 0, fe = 0;
            for (int i = 0; i < g.nx; ++i) {
                const std::size_t b = std::size_t(i) * DeviceTfles::kTflesSums;
                so += tfs[b];
                soe += tfs[b + 1];
                fe = std::max(fe, tfs[b + 2]);
            }
            m.Ebar = so > 0 ? soe / so : NAN;
            m.FEmax = fe;
        }
        double wmax = 0, wedge = 0, gmax = -1;
        int kmax = -1;
        bool found = false;
        for (int i = 0; i < g.nx; ++i) {
            const double w = col(i, 4) / plane;
            m.SL += w * hx;
            m.mass += col(i, 0) * dV;
            m.energy += col(i, 5) * dV;
            m.yHits += col(i, 8);
            m.yMin = std::min(m.yMin, col(i, 9));
            m.yMax = std::max(m.yMax, col(i, 10));
            m.burn += col(i, 4) * dV;
            m.Tmax = std::max(m.Tmax, col(i, 2) / plane);
            wmax = std::max(wmax, w);
            const bool band = nb > 0 ? (i < nb || i >= g.nx - nb)
                                     : (i == 0 || i == g.nx - 1);
            if (band) wedge = std::max(wedge, w);
            if (i + 1 < g.nx) {
                const double Ta = col(i, 2) / plane, Tb = col(i + 1, 2) / plane;
                if (!found && Ta < Tmid && Tb >= Tmid) {
                    m.xf = (i + 0.5 + (Tmid - Ta) / (Tb - Ta)) * hx;
                    found = true;
                }
                const double s = (Tb - Ta) / hx;
                if (s > gmax) {
                    gmax = s;
                    kmax = i;
                }
            }
        }
        if (r.diffusion) {
            for (auto* v : {&Th, &Rh, &Uh, &Wh, &YFh, &YOh}) v->resize(g.size());
            if (dev->T().download(Th.data()) != hipSuccess ||
                ds->rho.download(Rh.data()) != hipSuccess ||
                dev->u().download(Uh.data()) != hipSuccess ||
                dev->omega().download(Wh.data()) != hipSuccess ||
                dev->speciesY(0).download(YFh.data()) != hipSuccess ||
                dev->speciesY(1).download(YOh.data()) != hipSuccess)
                return fail(6, "field download failed");
            m.Tmax = *std::max_element(Th.begin(), Th.end());
            // Past the inflow band: the inflow row holds the Burke-Schumann
            // sheet's T whether or not the flame burns downstream.
            const int ib = std::max(nb, 4);
            for (int j = 0; j < g.ny; ++j)
                for (int i = ib; i < g.nx - ib; ++i)
                    m.TmaxIn = std::max(
                        m.TmaxIn, double(Th[std::size_t(j) * g.nx + i]));
            for (int i = 0; i < g.nx; ++i) m.pMean += col(i, 7);
            m.pMean /= double(g.nx) * plane * pRefMean;
            if (r.candle) {
                Vh.resize(g.size());
                if (dev->v().download(Vh.data()) != hipSuccess)
                    return fail(6, "field download failed");
                for (std::size_t c = 0; c < g.size(); ++c)
                    m.uMax = std::max(
                        m.uMax, std::hypot(double(Uh[c]), double(Vh[c])));
                for (int j = 0; j < g.ny; ++j)
                    m.uOutMin = std::min(
                        m.uOutMin,
                        double(Uh[std::size_t(j) * g.nx + g.nx - 1]));
            }
            m.jet = measureJet(r.diff, YFh, YOh, Wh, g.nx, g.ny, hx, r.geom.yc);
            // fuelIn = the species flux through the first interior face
            // (0|1) as kSpeciesStage2 forms it there: the face mdot times
            // the central Y (the boundary face's theta 1) minus rho D dY/dx
            // -- the flame base sits at the exit plane, so x-diffusion
            // carries ~3% of the fuel in (the inflow row's own rho u Y
            // missed it: burn/in read 1.032 steady). Outflow: rho u Y_F of
            // the last row (the open face carries no diffusion).
            for (int j = 0; j < g.ny; ++j) {
                const std::size_t c0 = std::size_t(j) * g.nx;
                const std::size_t c1 = c0 + 1, cn = c0 + g.nx - 1;
                const double md = 0.25 * (double(Rh[c0]) + Rh[c1]) *
                                  (double(Uh[c0]) + Uh[c1]);
                const double rd =
                    r.tr.muRef *
                    r.tr.ratio(0.5 * (double(Th[c0]) + Th[c1])) * r.sp.invPr;
                m.fuelIn += (md * 0.5 * (double(YFh[c0]) + YFh[c1]) -
                             rd * (double(YFh[c1]) - YFh[c0]) / hx) *
                            g.hy * g.hz;
                m.fuelOut += double(Rh[cn]) * Uh[cn] * YFh[cn] * g.hy * g.hz;
            }
            return m;
        }
        if (b3) {
            for (auto* v : {&Th, &Rh, &Uh, &Vh, &Wh}) v->resize(g.size());
            if (dev->T().download(Th.data()) != hipSuccess ||
                ds->rho.download(Rh.data()) != hipSuccess ||
                dev->u().download(Uh.data()) != hipSuccess ||
                dev->v().download(Vh.data()) != hipSuccess ||
                dev->omega().download(Wh.data()) != hipSuccess)
                return fail(6, "field download failed");
            // Pointwise peak (the 3D maximum, not the span mean's).
            m.Tmax = *std::max_element(Th.begin(), Th.end());
            spanMean(Th, Tm);
            spanMean(Rh, Rm);
            spanMean(Uh, Um);
            spanMean(Vh, Vm);
            spanMean(Wh, Wm);
            // The burning window: half a thermal thickness in 2D; with TFLES
            // the reaction zone of the F-thickened flame sits up to ~F - 1/2
            // thicknesses behind the Tmid crossing (measured: at F 3, 2.5
            // cells/delta_th the 2D window of 1 cell dropped every flank
            // row of the span-mean cone). F = 1 is the 2D window.
            const double Fw = r.tflesOn ? r.tfles.F : 1.0;
            // Round nozzle: the cone of the AXIAL plane (the mean of the
            // two k planes either side of zc, a cell face), not the span
            // mean -- a z mean of a round flame smears jet into coflow.
            if (r.geom.round)
                for (auto [f, mm] : {std::pair{&Th, &Tm}, {&Rh, &Rm}, {&Uh, &Um},
                                     {&Vh, &Vm}, {&Wh, &Wm}})
                    axialPlane(*f, *mm);
            m.cone = measureCone(Tm, g.nx, g.ny, hx, Tmid, r.geom.yc, Wm,
                                 int(std::ceil(0.5 * (2 * Fw - 1) * r.dth / hx)));
            m.mFlank = flankMassFlux(m.cone, Rm, Um, Vm, g.nx, hx);
            for (int k = 0; k < g.nz; ++k)
                for (int j = 0; j < g.ny; ++j) {
                    const std::size_t row = std::size_t(j) + std::size_t(g.ny) * k;
                    const std::size_t c = row * g.nx; // i = 0
                    m.fuelIn += double(Rh[c]) * Uh[c] * r.rowY[row] * g.hy * g.hz;
                }
            const double Lz = g.nz * g.hz;
            // SLflank is per unit SPAN of a slot flank: undefined for the
            // round cone (its ST below is the flank-area mean).
            if (m.cone.ok && !r.geom.round) {
                double w = 0;
                for (int i = m.cone.iLo; i <= m.cone.iHi; ++i) w += col(i, 4);
                const double cosA = 1.0 / std::hypot(1.0, m.cone.tanA);
                const double len =
                    2.0 * (m.cone.iHi - m.cone.iLo + 1) * hx / cosA;
                m.SLflank = w * dV / (len * Lz);
            }
            if (std::isfinite(m.cone.xTip) && std::isfinite(m.cone.xBase)) {
                if (turbColumnStats(*dev, turbScratch, ts) != hipSuccess)
                    return fail(6, "turbulence column stats download failed");
                double a = 0;
                for (int i = 0; i < g.nx; ++i)
                    a += ts[std::size_t(i) * kTurbSums + 6] * dV;
                // Slot: two flanks of span Lz; round: the circular cone's
                // lateral area pi R hypot(H, R), R = W/2.
                const double aCone =
                    r.geom.round
                        ? std::numbers::pi * 0.5 * r.geom.W *
                              std::hypot(m.cone.xTip - m.cone.xBase, 0.5 * r.geom.W)
                        : 2.0 * std::hypot(m.cone.xTip - m.cone.xBase,
                                           0.5 * r.geom.W) * Lz;
                m.ST = m.burn / aCone; // rho_u Y_u = 1
                m.area = a / aCone;
            }
            return m;
        }
        if (r.bunsen) {
            for (auto* v : {&Th, &Rh, &Uh, &Vh, &Wh}) v->resize(g.size());
            if (dev->T().download(Th.data()) != hipSuccess ||
                ds->rho.download(Rh.data()) != hipSuccess ||
                dev->u().download(Uh.data()) != hipSuccess ||
                dev->v().download(Vh.data()) != hipSuccess ||
                dev->omega().download(Wh.data()) != hipSuccess)
                return fail(6, "field download failed");
            m.Tmax = *std::max_element(Th.begin(), Th.end());
            m.cone = measureCone(Th, g.nx, g.ny, hx, Tmid, r.geom.yc, Wh,
                                 int(std::ceil(0.5 * r.dth / hx)));
            m.mFlank = flankMassFlux(m.cone, Rh, Uh, Vh, g.nx, hx);
            for (int j = 0; j < g.ny; ++j) {
                const std::size_t c = std::size_t(j) * g.nx; // i = 0
                m.fuelIn += double(Rh[c]) * Uh[c] * r.rowY[j] * g.hy * g.hz;
            }
            if (m.cone.ok) {
                // Flank consumption speed: omega over the fit rows (both
                // flanks), per unit flank length, rho_u Y_u = 1.
                double w = 0;
                for (int i = m.cone.iLo; i <= m.cone.iHi; ++i) w += col(i, 4);
                const double cosA = 1.0 / std::hypot(1.0, m.cone.tanA);
                const double len =
                    2.0 * (m.cone.iHi - m.cone.iLo + 1) * hx / cosA;
                m.SLflank = w * dV / len;
            }
            return m;
        }
        // Parabolic peak of the face slopes around the discrete maximum.
        double peak = gmax;
        if (kmax > 0 && kmax + 2 < g.nx) {
            auto sl = [&](int i) {
                return (col(i + 1, 2) - col(i, 2)) / plane / hx;
            };
            const double a = sl(kmax - 1), b = sl(kmax), c = sl(kmax + 1);
            const double den = a - 2 * b + c;
            if (den < 0) peak = b - (c - a) * (c - a) / (8 * den);
        }
        m.dth = peak > 0 ? dTflame / peak : 0;
        m.omegaEdge = wmax > 0 ? wedge / wmax : 0;
        // Consumption speed: integral of omega over the fresh rho Y.
        const double ryRef = col(iRef, 3) / plane;
        m.SL = ryRef > 0 ? m.SL / ryRef : NAN;
        m.uRef = col(iRef, 6) / plane;
        m.pRef = col(iRef, 7) / plane;
        m.TRef = col(iRef, 2) / plane;
        if (!found) m.xf = NAN;
        if (d3) {
            if (turbColumnStats(*dev, turbScratch, ts) != hipSuccess)
                return fail(6, "turbulence column stats download failed");
            auto tc = [&](int i, int q) {
                return ts[std::size_t(i) * kTurbSums + q];
            };
            double a = 0, var = 0;
            int nSlab = 0;
            for (int i = 0; i < g.nx; ++i) {
                a += tc(i, 6) * hx;
                const double x = (i + 0.5) * hx;
                if (!(x >= m.xf - 3 * lSlab && x <= m.xf - 1.5 * lSlab))
                    continue;
                for (int q = 0; q < 3; ++q) {
                    const double mu = tc(i, q) / plane;
                    var += tc(i, 3 + q) / plane - mu * mu;
                }
                ++nSlab;
            }
            m.area = a / plane;
            // max(0, .): a laminar plane's E[u^2] - E[u]^2 rounds to -ulp.
            m.uPrime =
                nSlab > 0 ? std::sqrt(std::max(0.0, var) / (3.0 * nSlab)) : NAN;
            double xi = 0;
            for (int i = 0; i < g.nx; ++i) {
                xi += col(i, 3) / plane * hx; // rho_u Y_u = 1
                const double c = (col(i, 2) / plane - r.Tu) / dTflame;
                if (std::isnan(m.xLead) && c >= kLeadC) m.xLead = (i + 0.5) * hx;
            }
            m.xFresh = xi;
        }
        return m;
    };

    auto emitProfiles = [&](double t) {
        for (int i = 0; i < g.nx; ++i) {
            auto cl = [&](int q) { return col(i, q); };
            const double row[] = {t,
                                  (i + 0.5) * hx,
                                  cl(2) / plane,
                                  cl(0) / plane,
                                  cl(6) / plane,
                                  cl(7) / plane,
                                  cl(3) / cl(0),
                                  cl(4) / plane};
            out.row("profiles", row);
        }
    };

    const float tempScale = float(dTflame);
    const float omegaScale = float(1.0 / r.dth); // ~ peak omega of S_L = 1
    internal::FrameScratch fs;
    auto writeFrame = [&](double t) {
        if (r.diffusion) {
            // + the oxidizer: a fourth channel only in this mode.
            if (!fs.ensure(g.size(), 4, false)) return;
            dev->updateChem(*ds);
            fs.take(0, dev->T(), r.Tu);
            fs.take(1, dev->speciesY(0));
            fs.take(2, dev->speciesY(1));
            fs.take(3, dev->omega());
            const FrameField ff[4]{
                {"temp", fs.channel(0), 0, tempScale, false},
                {"fuel", fs.channel(1), 0, 1.0f, false},
                {"oxid", fs.channel(2), 0, 1.0f, false},
                {"omega", fs.channel(3), 0, omegaScale, false}};
            fs.emit(out, t, ff);
            return;
        }
        if (d3 || b3) {
            // + |curl u| and |u| (cell-centred, central): the turbulence.
            if (!fs.ensure(g.size(), 5, false)) return;
            dev->updateChem(*ds);
            curlMagCentred(gg, dev->u(), dev->v(), dev->w(), fs.vort());
            speedCentred(gg, dev->u(), dev->v(), dev->w(), fs.speed());
            fs.take(0, dev->T(), r.Tu);
            fs.take(1, dev->speciesY(0));
            fs.take(2, dev->omega());
            fs.take(3, fs.vort());
            fs.take(4, fs.speed());
            const double us = std::max(1.0, r.uPrime);
            const FrameField ff[5]{
                {"temp", fs.channel(0), 0, tempScale, false},
                {"fuel", fs.channel(1), 0, 1.0f, false},
                {"omega", fs.channel(2), 0, omegaScale, false},
                {"vort", fs.channel(3), 0, float(3 * us / lSlab), false},
                {"speed", fs.channel(4), 0,
                 r.bunsen ? float(r.sigma * r.geom.uJet + 3 * r.uPrime)
                          : float(r.sigma * r.uIn + 3 * r.uPrime),
                 false}};
            fs.emit(out, t, ff);
            return;
        }
        if (!fs.ensure(g.size(), 3, false)) return;
        dev->updateChem(*ds);
        fs.take(0, dev->T(), r.Tu);
        fs.take(1, dev->speciesY(0));
        fs.take(2, dev->omega());
        const FrameField ff[3]{{"temp", fs.channel(0), 0, tempScale, false},
                               {"fuel", fs.channel(1), 0, 1.0f, false},
                               {"omega", fs.channel(2), 0, omegaScale, false}};
        fs.emit(out, t, ff);
    };

    // Bunsen: the jet speed now (the pulse) and the nominal fuel inflow
    // sum_j rho_row u_row(t) Y_row hy.
    const double dy = g.hy * g.hz;
    auto jetNow = [&](double t) {
        return r.geom.pulse > 0
                   ? r.geom.uJet * r.geom.pulse *
                         std::sin(2 * std::numbers::pi * r.geom.pulseFreq * t)
                   : 0.0;
    };
    auto fuelIn = [&](double t) {
        const double du = jetNow(t);
        double f = 0;
        for (std::size_t q = 0; q < r.rowU.size(); ++q)
            f += r.rowRho[q] * (r.rowU[q] + du * r.rowJet[q]) * r.rowY[q] * dy;
        return f;
    };

    RunStats st;
    double t = 0.0;
    internal::StatsCadence cad{r.statsEvery};
    double dtStab = 0.0, dtChem = 0.0;
    int sinceDt = 1 << 30;
    internal::ProfileSchedule prof(r.profileDt);
    double xfPrev = NAN, tPrev = 0;
    const double guard = nb * hx + 2 * r.dth;
    const double Lx = g.nx * hx;
    double lastHits = 0, lastMin = 0, lastMax = 1;
    const auto wall0 = std::chrono::steady_clock::now();
    bool bindChem = false;
    // --turb: the swept distance (FP64 host) and the current inflow target
    // (held over a step; --hold-flame moves it at stats rows).
    double sweep = 0, uNow = r.uIn, sEma = r.uIn, tCtl = 0, xI0 = NAN;

    while (t < r.tEnd) {
        if (stopRequested()) break;
        // TFLES: F E, E/F, Omega of this step's input state, held over its
        // three stages (and read by the dt bound below).
        if (tf) {
            dev->primitives(*ds);
            tf->update(*ds, dev->u(), dev->v(), dev->w(), dev->T());
        }
        if (auto e = internal::recomputeDt(
                sinceDt, dtStab, r.dtRecompute, r.dtMargin, 1e-9,
                [&] {
                    double a = dev->stableDt(*ds, *red, r.cfl);
                    if (tf)
                        a = std::min(a, tflesStableDt(*dev, *ds,
                                                      tf->feField(), *red));
                    dtChem = chemStableDt(*dev, *ds, *red);
                    bindChem = dtChem < a;
                    return std::min(a, dtChem);
                });
            !e)
            return fail(7, e.error());
        const double dt = std::min(r.tEnd - t, dtStab);
        // The pulse is held over the step (the stages see its start value).
        if (rows && r.geom.pulse > 0) nsc.setInflowPulse(jetNow(t));
        // The box column the inflow sees now, held over the step (as the
        // pulse); the face then advects it in at u_in.
        if (turbIn) turbIn->sample(sweep);
        // The box sweeps past the face at the inflow speed: u_in, or the
        // bunsen jet's (pulse included, held over the step as the pulse).
        const double uSweep = r.bunsen ? r.geom.uJet + jetNow(t) : uNow;
        dev->step(*ds, Real(dt));
        t += dt;
        sweep += uSweep * dt;
        ++st.steps;

        const bool statsDue = cad.due(st.steps, t, r.tEnd);
        const bool profDue = !rows && prof.due(t, r.tEnd);
        if (statsDue || profDue) {
            auto m = measure();
            if (!m) return std::unexpected(m.error());
            if (auto h = internal::checkDevice("a stats row", &*red); !h)
                return std::unexpected(h.error());
            if (!std::isfinite(m->mass) || !std::isfinite(m->Tmax) ||
                (!rows && !std::isfinite(m->xf)))
                return fail(7, "flame state is not finite (diverged)");
            lastHits = m->yHits;
            lastMin = m->yMin;
            lastMax = m->yMax;
            if (r.diffusion) {
                lastHits = m->jet.hits;
                lastMin = std::min(m->jet.yfMin, m->jet.yoMin);
                lastMax = m->jet.zMax;
            }
            if (statsDue && r.diffusion) {
                const JetMeasure& jm = m->jet;
                const double row[] = {t,          dt,         dtChem,
                                      jm.xTip,    jm.xBase,   jm.half,
                                      m->Tmax,    m->TmaxIn,  m->burn,
                                      m->fuelIn,
                                      m->fuelOut, m->burn / m->fuelIn,
                                      m->mass,    m->energy,  m->pMean,
                                      jm.zMin,
                                      jm.zMax,    jm.yfMin,   jm.yoMin,
                                      jm.hits,    1.0 + jetNow(t),
                                      m->uMax,    m->uOutMin};
                emitRow(row, m->Ebar, m->FEmax);
                if (cad.printDue(st.steps, t, r.tEnd)) {
                    std::printf(
                        "t=%.3f/%.3f tip=%.4g base=%.4g half=%.4g "
                        "Tmax/Tad=%.5f (past band %.5f) burn/in=%.4f "
                        "out/in=%.4f p/p0=%.4f "
                        "Z=[%.3g,1%+.2e] hits=%g dt=%.2e%s",
                        t, r.tEnd, jm.xTip, jm.xBase, jm.half,
                        m->Tmax / (r.sigma * r.Tu),
                        m->TmaxIn / (r.sigma * r.Tu), m->burn / m->fuelIn,
                        m->fuelOut / m->fuelIn, m->pMean, jm.zMin,
                        jm.zMax - 1, jm.hits,
                        dt, bindChem ? " (chem dt)" : "");
                    if (r.candle)
                        std::printf(" |u|max=%.4g uOutMin=%.4g", m->uMax,
                                    m->uOutMin);
                    std::printf("\n");
                    std::fflush(stdout);
                    cad.printed(st.steps);
                }
                cad.logged(st.steps);
            } else if (statsDue && r.bunsen) {
                const ConeMeasure& c = m->cone;
                const double sinA = c.ok ? c.tanA / std::hypot(1.0, c.tanA)
                                         : NAN;
                const double uj = r.geom.uJet + jetNow(t);
                const double row[] = {t,       dt,        dtChem,
                                      c.xTip,  c.xBase,   c.ok ? c.tanA : NAN,
                                      c.ok ? c.tanL : NAN,
                                      c.ok ? c.tanR : NAN,
                                      c.ok ? c.rms : NAN,
                                      sinA * r.geom.uJet, m->SLflank,
                                      m->mFlank, m->burn, m->fuelIn,
                                      fuelIn(t), m->burn / m->fuelIn,
                                      m->Tmax, m->mass,   m->energy,
                                      m->yHits, m->yMin,  m->yMax,
                                      uj};
                if (b3) {
                    rowBuf3.assign(std::begin(row), std::end(row));
                    rowBuf3.insert(rowBuf3.end(),
                                   {m->ST, m->area, m->ST / m->area});
                    emitRow(rowBuf3, m->Ebar, m->FEmax);
                } else {
                    emitRow(row, m->Ebar, m->FEmax);
                }
                if (cad.printDue(st.steps, t, r.tEnd)) {
                    std::printf(
                        "t=%.3f/%.3f tip=%.3f base=%.3f alpha=%.3f deg "
                        "sin(a)U/SL=%.5f SLflank=%.5f mFlank=%.5f "
                        "burn/in=%.4f Tmax/Tb=%.5f yHits=%g Y=[%.3g,1%+.2e] "
                        "dt=%.2e%s",
                        t, r.tEnd, c.xTip, c.xBase,
                        std::atan(c.tanA) * 180 / std::numbers::pi,
                        sinA * r.geom.uJet, m->SLflank, m->mFlank,
                        m->burn / m->fuelIn,
                        m->Tmax / (r.sigma * r.Tu), m->yHits, m->yMin,
                        m->yMax - 1, dt, bindChem ? " (chem dt)" : "");
                    if (b3)
                        std::printf(" ST=%.4f area=%.4f I0=%.4f", m->ST,
                                    m->area, m->ST / m->area);
                    if (b3 && tf)
                        std::printf(" Ebar=%.4f FEmax=%.4f", m->Ebar, m->FEmax);
                    std::printf("\n");
                    std::fflush(stdout);
                    cad.printed(st.steps);
                }
                cad.logged(st.steps);
            } else if (statsDue) {
                const double dxf = std::isfinite(xfPrev) && t > tPrev
                                       ? (m->xf - xfPrev) / (t - tPrev)
                                       : 0.0;
                const double row[] = {t,            dt,
                                      dtChem,       m->SL,
                                      m->uRef - dxf, m->xf,
                                      m->dth,       m->Tmax,
                                      m->mass,      m->energy,
                                      m->omegaEdge, m->yHits,
                                      m->uRef,      m->pRef,
                                      m->TRef,      m->yMin,
                                      m->yMax,      m->area,
                                      m->SL / m->area, m->uPrime,
                                      uNow,         m->xFresh,
                                      m->xLead};
                emitRow(row, m->Ebar, m->FEmax);
                xfPrev = m->xf;
                tPrev = t;
                if (cad.printDue(st.steps, t, r.tEnd)) {
                    std::printf("t=%.3f/%.3f SL=%.6f xf=%.4f dth=%.4f "
                                "Tmax/Tb=%.5f omegaEdge=%.2e yHits=%g dt=%.2e%s",
                                t, r.tEnd, m->SL, m->xf, m->dth,
                                m->Tmax / (r.sigma * r.Tu), m->omegaEdge,
                                m->yHits, dt, bindChem ? " (chem dt)" : "");
                    if (d3)
                        std::printf(" area=%.4f I0=%.4f u'=%.4f u_in=%.4f "
                                    "xI=%.4f lead=%.4f",
                                    m->area, m->SL / m->area, m->uPrime, uNow,
                                    m->xFresh, m->xLead);
                    if (tf)
                        std::printf(" Ebar=%.4f FEmax=%.4f", m->Ebar, m->FEmax);
                    std::printf("\n");
                    std::fflush(stdout);
                    cad.printed(st.steps);
                }
                cad.logged(st.steps);
                // --hold-flame: u_in tracks the consumption speed (EMA over
                // kHoldEmaTau) and pulls the fresh-mass thickness back to its
                // first-row value over kHoldTau, rate-limited; held over the
                // steps until the next row.
                if (r.hold && std::isnan(xI0)) xI0 = m->xFresh;
                if (r.hold && t > tCtl && std::isfinite(m->xFresh) &&
                    std::isfinite(m->SL)) {
                    const double dtc = t - tCtl;
                    tCtl = t;
                    sEma += (m->SL - sEma) * std::min(1.0, dtc / kHoldEmaTau);
                    const double want = sEma - (m->xFresh - xI0) / kHoldTau;
                    const double lim = kHoldMaxRate * dtc;
                    uNow = std::clamp(std::clamp(want, uNow - lim, uNow + lim),
                                      kHoldUMin, kHoldUMax);
                    nsc.setFreestream(uNow);
                }
            }
            if (profDue) {
                emitProfiles(t);
                prof.done(t);
            }
            if (r.quiescent) {
                // No fuel: nothing to guard (the rest check reads uMax).
            } else if (r.diffusion) {
                std::string why;
                if (m->TmaxIn < Tmid)
                    why = std::format("Tmax past the inflow band = {:.4g} T0 "
                                      "fell below the mid temperature {:.4g} "
                                      "T0 (extinction)",
                                      m->TmaxIn / r.Tu, Tmid / r.Tu);
                else if (!(m->jet.xBase <= 0.5 * Lx))
                    why = std::format("the flame base x={:.4g} lifted past "
                                      "half the box (lift-off)",
                                      m->jet.xBase);
                if (!why.empty())
                    return fail(8, std::format("diffusion flame lost at "
                                               "t={:.4g}: {} (raise "
                                               "--damkohler)",
                                               t, why));
            } else if (r.bunsen) {
                const ConeMeasure& c = m->cone;
                std::string why;
                if (c.axisHot)
                    why = "the slot axis is hot at the inflow (flash-back)";
                else if (!std::isfinite(c.xTip))
                    why = "the slot axis never reaches Tmid (blow-through)";
                else if (c.xTip > Lx - guard)
                    why = std::format("the tip x={:.4g} reached the outflow "
                                      "band (blow-through)",
                                      c.xTip);
                else if (!std::isfinite(c.xBase) || c.xBase > 0.5 * c.xTip)
                    why = std::format("the base x={:.4g} lifted past half the "
                                      "tip height {:.4g} (blow-off)",
                                      c.xBase, c.xTip);
                else if (!c.ok && !b3)
                    // 3D: a turbulent brush's span mean need not keep a
                    // straight flank; the fit's columns read NaN, the
                    // physical guards above stay.
                    why = std::format("fewer than 4 burning flank rows between "
                                      "base {:.4g} and tip {:.4g} (the cone "
                                      "opened: blow-off)",
                                      c.xBase, c.xTip);
                if (!why.empty())
                    return fail(8, std::format("flame not anchored at t={:.4g}: "
                                               "{} (adjust --jet-speed, "
                                               "--coflow-temp)",
                                               t, why));
            } else if (d3 && !(m->xLead >= nb * hx + r.dth))
                return fail(8, std::format(
                                   "flash-back at t={:.4g}: the brush's leading "
                                   "edge (plane-mean c >= {:g}) at x={:.4g} "
                                   "reached one delta_th of the inflow band "
                                   "(x_f={:.4g}; raise --inflow-speed or "
                                   "--upstream)",
                                   t, kLeadC, m->xLead, m->xf));
            else if (m->xf < guard || m->xf > g.nx * hx - guard)
                return fail(8, std::format(
                                   "flame not stabilized: drifted to the "
                                   "{} band at t={:.4g} (x_f={:.4g}; adjust "
                                   "--inflow-speed)",
                                   m->xf < guard ? "inflow" : "outflow", t,
                                   m->xf));
        }
        if (out.wantsAnyFrame(t)) writeFrame(t);
    }
    if (auto h = internal::checkDevice("the end of the run", &*red); !h)
        return std::unexpected(h.error());
    if (lastHits > 0 && r.diffusion)
        std::fprintf(stderr,
                     "dns: warning: %g cells with Z, Y_F or Y_O outside "
                     "[-1e-6, 1+1e-6] at the last stats row (counted, never "
                     "clamped): min Y %.3g, max Z 1%+.3g\n",
                     lastHits, lastMin, lastMax - 1);
    else if (lastHits > 0)
        std::fprintf(stderr,
                     "dns: warning: %g cells with Y outside [-1e-6, 1+1e-6] "
                     "at the last stats row (counted, never clamped): Y in "
                     "[%.3g, 1%+.3g], %.3g ulp of 1 above\n",
                     lastHits, lastMin, lastMax - 1,
                     (lastMax - 1) / std::ldexp(1.0, -23));
    std::printf("label: %s\n", r.label.c_str());
    finishTiming(st, wall0, out);
    st.tStop = t;
    return st;
}

} // namespace dns::gpu
