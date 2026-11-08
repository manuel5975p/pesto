// dns flame on the structured solvers: compressible (1D, 3D, bunsen, diffusion, candle), chamber, wick, low-Mach.
#include "dns/drive/internal.hpp"

#include "dns/checkpoint.hpp"
#include "dns/config.hpp"
#include "dns/flame1d.hpp"
#include "dns/lowmach.hpp"
#include "dns/output.hpp"
#include "dns/transport.hpp"

#if DNS_WITH_GPU
#include "dns/gpu/runner_flame.hpp"
#include "dns/gpu/runner_lowmach.hpp"
#include "dns/gpu/runner_chamber.hpp"
#include "dns/gpu/runner_wick.hpp"
#include "dns/gpu/comp.hpp"
#endif

#include <cmath>
#include <cstdio>
#include <format>
#include <limits>
#include <numbers>
#include <optional>
#include <string>
#include <vector>

namespace dns::drive {

using std::numbers::pi;
namespace {



// --mode chamber (docs/chem.md, "Chamber"): the closed 2D combustion chamber.
int runChamberCase(const Case& c)
{
    if (!haveGpu(c))
        return die(exitNoGpu, "no GPU device (dns flame is GPU-only)");
#if DNS_WITH_GPU
    const auto& g = c.grid;
    const ChamberSpec& s = c.chamber;
    const double Tu = c.flameTu, sig = c.flameModel.sigma;
    gpu::ChamberRun rr;
    rr.regions = chamberRegions(g, s);
    CompState host;
    std::vector<double> rhoY;
    const FlameEigen& tab = c.flameEig.tab;
    chamberIC(g, rr.regions, kGammaAir, Tu, sig, c.chamberCoflow,
              [&](double x) { return flameProfile(tab, x); }, s.xBase(), host,
              rhoY);

    FrameGeometry fg = FrameGeometry::centred(g, s.Hc);
    fg.mask = frameMask(g.size(), rr.regions, std::uint8_t(ChamberRegion::Solid));
    auto out = RunOutput::make(
        outputConfig(c, std::format("{{\"app\":\"dns\",\"flame\":\"chamber\","
                                    "\"sigma\":{},\"beta\":{},\"Lambda\":{},"
                                    "\"mach\":{},\"h\":{},\"xe\":{},"
                                    "\"yc\":{},\"He\":{},\"Hc\":{}",
                                    sig, c.flameModel.beta, c.flameEig.Lambda,
                                    c.mach, c.h, s.xe(), s.yc, s.He, s.Hc)),
        c.meta, g, std::move(fg));
    if (!out) return die(exitOutput, out.error());

    rr.tr = flameGas(c, Tu);
    rr.sp.nY = 1;
    rr.sp.fuel = 0;
    rr.sp.lnA = c.chamberReaction ? std::log(c.flameEig.Lambda)
                                  : -std::numeric_limits<double>::infinity();
    rr.sp.Ta = c.flameModel.theta() * Tu;
    rr.sp.q = rr.tr.cp() * (sig - 1) * Tu;
    rr.sp.invPr = 1.0 / c.prandtl;
    rr.sp.openX = true;
    rr.sp.yIn[0] = 0.0; // the ambient is inert
    rr.Tu = Tu;
    rr.sigma = sig;
    rr.uCo = c.chamberCoflow;
    rr.etaIn = c.nscbcEtaIn;
    rr.thermalInflow = c.nscbcThermalInflow;
    rr.sinkNb = c.nscbcSink;
    rr.sinkAmp = c.nscbcSinkAmp;
    rr.spec = s;
    rr.mdot = c.chamberMdot;
    rr.ramp = c.chamberRamp;
    rr.reaction = c.chamberReaction;
    rr.cfl = c.cfl;
    rr.tEnd = c.tEnd;
    rr.dtRecompute = c.dtRecompute;
    rr.label = chamberRunLabel(c);
    rr.shock = c.shock;
    rr.shockDtAuto = c.shockDtAuto;

    auto res = gpu::runChamber(g, host, rhoY, rr, *out);
    if (!res) return die(res.error().code, res.error().message);
    return reportDone(*res, *out);
#else
    return exitNoGpu;
#endif
}

// --mode candle --wick (docs/chem.md, "Wick candle"): the immersed candle
// body + wick, fuel from the wick surface, soot.
int runWickCase(const Case& c)
{
    if (!haveGpu(c))
        return die(exitNoGpu, "no GPU device (dns flame is GPU-only)");
#if DNS_WITH_GPU
    const auto& g = c.grid;
    const DiffusionModel& m = c.diff;
    const WickSpec& s = c.wick;
    const double T0 = c.flameTu, gx = c.flameGravity;
    gpu::WickRun rr;
    rr.regions = wickRegions(g, s);
    CompState host;
    std::vector<double> rhoY[3];
    wickIC(c, host, rhoY);

    FrameGeometry fg = FrameGeometry::centred(g, 1.0);
    fg.mask = frameMask(g.size(), rr.regions, std::uint8_t(WickRegion::Solid));
    auto out = RunOutput::make(
        outputConfig(c, std::format("{{\"app\":\"dns\",\"flame\":\"wick\","
                                    "\"stoich\":{},\"zst\":{},\"sigma\":{},"
                                    "\"Da\":{},\"mach\":{},\"gravity\":{},"
                                    "\"h\":{},\"yc\":{},\"xTop\":{},"
                                    "\"wick\":{},\"soot\":{}",
                                    m.s, m.zst(), m.sigma, m.Da, c.mach, gx,
                                    c.h, s.yc, s.xTop, s.wWick, c.wickSoot)),
        c.meta, g, std::move(fg));
    if (!out) return die(exitOutput, out.error());

    rr.tr = flameGas(c, T0);
    rr.sp.nY = c.wickSoot ? 3 : 2;
    rr.sp.fuel = 0;
    rr.sp.ox = 1;
    rr.sp.stoich = m.s;
    rr.sp.lnA = m.lnA();
    rr.sp.Ta = m.theta() * T0;
    rr.sp.q = rr.tr.cp() * T0 * m.qOverCpT0();
    rr.sp.invPr = 1.0 / c.prandtl;
    rr.sp.openX = true;
    if (c.wickSoot) {
        rr.sp.soot = 2;
        rr.sp.sootLnAf = c.soot.lnAf(m.sigma);
        rr.sp.sootTaf = c.soot.thetaF * T0;
        rr.sp.sootLnAox = c.soot.lnAox(m.sigma);
        rr.sp.sootTaox = c.soot.thetaOx * T0;
        rr.sp.sootDiff = 0.0;
    }
    // The ambient column (R T0 = p0 = T0): the inflow rows' density at
    // p_a(h/2), the outflow target p_a at the last row, and the hydrostatic
    // dp/dx = -g rho_a for the NSCBC waves.
    const double pIn = candleAmbientP(T0, gx, T0, 0.5 * g.hx);
    rr.pOut = candleAmbientP(T0, gx, T0, (g.nx - 0.5) * g.hx);
    rr.hydroLo = -gx * pIn / T0;
    rr.hydroHi = -gx * rr.pOut / T0;
    rr.sp.ambientHi = true;
    rr.sp.yHi[1] = m.yO2;
    FlameInflowRows rows = wickInflowRows(c, pIn);
    rr.rowU = std::move(rows.u);
    rr.rowJet = std::move(rows.jet);
    rr.rowTarget = std::move(rows.target);
    rr.sp.yInRow[0] = std::move(rows.yF);
    rr.sp.yInRow[1] = std::move(rows.yO);
    if (c.wickSoot) rr.sp.yInRow[2] = std::vector<double>(std::size_t(g.ny), 0.0);
    rr.diff = m;
    rr.spec = s;
    rr.T0 = T0;
    rr.gravity = gx;
    rr.uCo = c.bunsen.uCo;
    rr.etaIn = c.nscbcEtaIn;
    rr.thermalInflow = c.nscbcThermalInflow;
    rr.sinkNb = c.nscbcSink;
    rr.sinkAmp = c.nscbcSinkAmp;
    rr.cfl = c.cfl;
    rr.tEnd = c.tEnd;
    rr.dtRecompute = c.dtRecompute;
    rr.label = wickRunLabel(c);
    const std::span<const std::vector<double>> ry(rhoY,
                                                  std::size_t(rr.sp.nY));
    auto res = gpu::runWick(g, host, ry, rr, *out);
    if (!res) return die(res.error().code, res.error().message);
    return reportDone(*res, *out);
#else
    return exitNoGpu;
#endif
}

// dns flame --physics lowmach (masterplan real_bunsen S4): the premixed
// 1D/2D/3D flame on the zero-Mach solver (gpu/runner_lowmach.hpp). Box units
// Tu = P0 = R = rho_u = 1 (c.flameTu), mu = Pr, lambda = cp, rho D = 1.
int runLowMachCase(const Case& c, [[maybe_unused]] const CheckpointContext& ckx)
{
    if (!haveGpu(c))
        return die(exitNoGpu, "no GPU device (dns flame is GPU-only)");
#if DNS_WITH_GPU
    const auto& g = c.grid;
    const int nY = c.flamePremix2 ? 2 : 1;
    auto restart = readFlameRestart(c, nY);
    if (!restart) return restart.error();
    LowMachState ic = lowmachIC(c);
#if DNS_WITH_FFTW
    // --turb (M5, docs/chem.md "Low-Mach turbulence"): the box on the MAC faces.
    if (c.flameTurb > 0) lowmachAddTurbulence(c, ic);
#endif
    FrameGeometry fg = FrameGeometry::centred(g, 1.0);
    // S8: the tube (the one solid map: frames grey it out, the solver's
    // pressure codes derive from it).
    std::vector<std::uint8_t> solid = c.lmRim ? lmTubeSolid(c.bunsen, g)
                                              : std::vector<std::uint8_t>{};
    fg.mask = frameMask(g.size(), solid, 1);
    const double sig = flameBurnt(c).sigma;
    const BunsenGeom& bg = c.bunsen;
    // S8: the tube for the renderer (box units: rim top x, inner and outer
    // radius, cell size, wall T/Tu) -- the .json and the web manifest.
    const std::string rimJson =
        c.lmRim ? std::format("{{\"x\":{},\"ri\":{},\"ro\":{},\"h\":{},\"T\":{}}}",
                              bg.xRim, 0.5 * bg.W, 0.5 * bg.W + bg.rimLip, c.h,
                              bg.tRim)
                : std::string();
    RunMeta meta = c.meta;
    meta.rim = rimJson;
    // S9: the SI echo (--si-delta-d, --si-sl) for the renderer's real scale.
    if (c.siDeltaD > 0)
        meta.si = std::format("{{\"deltaD\":{:.9g},\"SL\":{:.9g}}}", c.siDeltaD, c.siSL);
    auto out = RunOutput::make(
        outputConfig(c, std::format("{{\"app\":\"dns\",\"flame\":\"{}\","
                                    "\"physics\":\"lowmach\",\"model\":\"{}\","
                                    "\"sigma\":{},\"beta\":{},\"Lambda\":{},"
                                    "\"mach\":0{}",
                                    c.flameBunsen ? "bunsen" : "premixed",
                                    c.flamePremix2 ? "C2" : "C0", sig,
                                    c.flameModel.beta, c.flameEig.Lambda,
                                    c.flameBunsen
                                        ? std::format(
                                              ",\"nozzle\":\"{}\",\"W\":{},"
                                              "\"uJet\":{},\"yc\":{},\"zc\":{},"
                                              "\"dth\":{},\"phi\":{},\"zst\":{}{}{}",
                                              bg.round ? "round" : "slot", bg.W,
                                              bg.uJet, bg.yc, bg.zc,
                                              c.flameEig.dthOverDD,
                                              c.flamePremix2 ? c.premix.phi : 0.0,
                                              c.flamePremix2 ? c.premix.zst() : 0.0,
                                              c.lmOpenY || c.lmInert
                                                  ? std::format(
                                                        ",\"laterals\":\"{}\","
                                                        "\"reaction\":{}",
                                                        c.lmOpenY ? "open"
                                                                  : "periodic",
                                                        c.lmInert ? "false"
                                                                  : "true")
                                                  : std::string(),
                                              c.lmGravity > 0
                                                  ? std::format(",\"gravity\":{}",
                                                                c.lmGravity)
                                                  : std::string()) +
                                              (c.lmRim ? ",\"rim\":" + rimJson
                                                       : std::string())
                                        : std::string())),
        meta, g, std::move(fg));
    if (!out) return die(exitOutput, out.error());

    const double Tu = c.flameTu; // 1
    gpu::LowMachRun rr;
    gpu::LowMachSetup& sp = rr.sp;
    // mu = Pr at Tref = Tu: lambda_u = mu cp / Pr = cp (delta_D = 1).
    sp.tr = flameGas(c, Tu);
    sp.prGas = Tu; // P0/R, rho_u = 1
    sp.nY = 1;
    sp.fuel = 0;
    sp.lnA = std::log(c.flameEig.Lambda);
    sp.Ta = c.flameModel.theta() * Tu;
    sp.qcp = (sig - 1) * Tu; // C0: q = cp (sigma - 1) Tu
    sp.mIn = c.flameUIn;     // rho_u u_in
    sp.TIn = Tu;
    sp.yIn[0] = 1.0;
    if (c.flamePremix2) {
        // Model C2 / C2b: the two-reactant step, q per unit fuel = cp Tu
        // (sigma - 1) s / Y_O1, the burnt premix carried in (--mix-phi).
        const PartialPremixModel& pm = c.premix;
        PartialPremixModel burn = pm;
        burn.phi = c.premixBurnPhi();
        sp.nY = 2;
        sp.ox = 1;
        sp.stoich = pm.s;
        sp.qcp = Tu * pm.qOverCpTu();
        sp.yIn[0] = burn.yF1();
        sp.yIn[1] = burn.yO1();
        sp.law = c.premixLaw;
        sp.zS = pm.s;
        sp.zO2 = pm.yO2;
        sp.zDen = pm.s * pm.yFrem() + pm.yO2;
        rr.yFBurnt = burn.phi > 1 ? burn.yFrem() : 0.0;
        rr.premix2 = true;
        rr.premix = pm;
    }
    if (c.flameBunsen) {
        // The burner (masterplan real_bunsen S5): the x-lo face per row j + ny
        // k (lmInflowRow); the scalars are the jet's.
        rr.bunsen = true;
        rr.geom = bg;
        sp.mIn = bg.uJet;
        // S6: open laterals draw the quiescent room air (T = Tu; air with
        // --equivalence, else inert); --reaction off = no chemistry.
        sp.openY = c.lmOpenY;
        sp.openZ = c.lmOpenZ;
        sp.ambT = Tu;
        sp.ambY[0] = 0.0;
        sp.ambY[1] = c.flamePremix2 ? c.premix.yO2 : 0.0;
        // S7: gravity along -x against that ambient (rho_a = P0/(R Tu)).
        sp.gravity = c.lmGravity;
        if (c.lmInert) sp.lnA = -HUGE_VAL;
        rr.inert = c.lmInert;
        // S8: the immersed isothermal tube.
        rr.rim = c.lmRim;
        sp.solid = std::move(solid);
        sp.wallT = Tu * bg.tRim;
        LmInflowRows rows = lmInflowRows(c);
        sp.rowM = std::move(rows.m);
        sp.rowT = std::move(rows.T);
        sp.rowY[0] = std::move(rows.yF);
        sp.rowY[1] = std::move(rows.yO);
    }
    // M4 (docs/chem.md "Low-Mach LES"): LES and the thickened flame.
    if (c.lesLabel) {
        sp.cs = c.cs;
        sp.sgs = c.sgsModel;
    }
    if (c.flameTfles > 0) sp.tfles = flameTflesSetup(c, sig, c.flameTflesDilate);
    rr.sigma = sig;
    rr.uIn = c.flameUIn;
    rr.dth = c.flameEig.dthOverDD;
    rr.cfl = c.cfl;
    rr.chemK = c.lmChemDt;
    rr.dtCell = c.lmDtCell;
    rr.tEnd = c.tEnd;
    rr.dtRecompute = c.dtRecompute;
    rr.profileDt = c.profileDt;
    rr.label = flameRunLabel(c);
    rr.ckpt = flamePolicy(c, ckx, *restart);
    std::optional<LowMachTurbSweeps> turbSweeps;
    if (c.flameTurb > 0) {
        sp.turb = true;
        turbSweeps = lowmachTurbSweeps(c);
        rr.turb = &*turbSweeps;
    }
    auto res = gpu::runLowMach(g, ic, rr, *out);
    if (!res) return die(res.error().code, res.error().message);
    return reportDone(*res, *out);
#else
    return exitNoGpu;
#endif
}
} // namespace


int runFlame(const Case& c, [[maybe_unused]] const CheckpointContext& ckx)
{
    if (c.chamberMode) return runChamberCase(c);
    if (c.flameWick) return runWickCase(c);
#if DNS_WITH_GPU
    if (c.physics == Physics::LowMach && c.fvMode != FvCompMode::Off) return runFvLowMachCase(c, ckx);
#endif
    if (c.physics == Physics::LowMach) return runLowMachCase(c, ckx);
    if (!haveGpu(c) && c.fvMode == FvCompMode::Off)
        return die(exitNoGpu, "no GPU device (dns flame is GPU-only)");
#if DNS_WITH_GPU
    if (c.fvMode != FvCompMode::Off) return runFvFlame(c, ckx);
    const auto& g = c.grid;
    // --restart: the whole checkpoint is this run's input (exit 3), read
    // before any output is touched; phase 1.5 checked its binary and case.
    auto restart = readFlameRestart(c, c.flameDiffusion || c.flamePremix2 ? 2 : 1);
    if (!restart) return restart.error();
    const CheckpointPolicy policy = flamePolicy(c, ckx, *restart);
    CompState host = CompState::zeros(g);
    std::vector<double> rhoY[2];
    if (c.flameDiffusion) diffusionIC(c, host, rhoY);
    else if (c.flameBunsen) bunsenIC(c, host, rhoY[0]);
    else if (c.flamePremix2) flameIC2(c, host, rhoY);
    else flameIC(c, host, rhoY[0]);
    // --turb: the frozen box (hash-seeded, host FP64), laid on the IC's
    // fresh gas and handed to the runner for the inflow sweep.
    std::array<std::vector<double>, 3> turbBoxHost;
#if DNS_WITH_FFTW
    if (c.flameTurb > 0) {
        turbBoxHost = turbBox(flameTurbSpec(c));
        if (c.flameBunsen) addBunsenTurbulence(c, turbBoxHost, rhoY[0], host);
        else addTurbulence(c, turbBoxHost, host);
    }
#endif

    FrameGeometry fg = FrameGeometry::centred(g, 1.0);
    fg.mask.assign(g.size(), 1.0f);
    auto out = RunOutput::make(
        outputConfig(
            c, c.flameCandle
                   ? std::format("{{\"app\":\"dns\",\"flame\":\"candle\","
                                 "\"stoich\":{},\"zst\":{},\"sigma\":{},"
                                 "\"beta\":{},\"Da\":{},\"mach\":{},"
                                 "\"gravity\":{},\"slot\":{},\"quiescent\":{}",
                                 c.diff.s, c.diff.zst(), c.diff.sigma,
                                 c.diff.beta, c.diff.Da, c.mach,
                                 c.flameGravity, c.bunsen.W,
                                 c.flameQuiescent)
               : c.flameDiffusion
                   ? std::format("{{\"app\":\"dns\",\"flame\":\"diffusion\","
                                 "\"stoich\":{},\"zst\":{},\"sigma\":{},"
                                 "\"beta\":{},\"Da\":{},\"mach\":{}",
                                 c.diff.s, c.diff.zst(), c.diff.sigma,
                                 c.diff.beta, c.diff.Da, c.mach)
                   : c.flamePremix2
                   ? std::format("{{\"app\":\"dns\",\"flame\":\"premixed\","
                                 "\"model\":\"C2\",\"phi\":{},\"stoich\":{},"
                                 "\"yO2\":{},\"yF1\":{},\"yO1\":{},"
                                 "\"yFrem\":{},\"zst\":{},\"sigmaSt\":{},"
                                 "\"sigma\":{},\"beta\":{},\"Lambda\":{},"
                                 "\"mach\":{}",
                                 c.premix.phi, c.premix.s, c.premix.yO2,
                                 c.premix.yF1(), c.premix.yO1(),
                                 c.premix.yFrem(), c.premix.zst(),
                                 c.premix.sigmaSt(), c.flameModel.sigma,
                                 c.flameModel.beta, c.flameEig.Lambda, c.mach)
                   : std::format("{{\"app\":\"dns\",\"flame\":\"premixed\","
                                 "\"sigma\":{},\"beta\":{},\"Lambda\":{},"
                                 "\"mach\":{}",
                                 c.flameModel.sigma, c.flameModel.beta,
                                 c.flameEig.Lambda, c.mach)),
        c.meta, g, std::move(fg));
    if (!out) return die(exitOutput, out.error());

    if (c.flameDiffusion) {
        // Two reactants (dns/diffusion.hpp): box units rho0 = R = U_jet = 1,
        // l = D0/U_jet = 1 (mu = Pr), T0 = p0; the bunsen per-row inflow
        // carrying the Burke-Schumann sheet of the inflow mixture fraction.
        const DiffusionModel& m = c.diff;
        const double T0 = c.flameTu;
        gpu::FlameRun rr;
        rr.tr = flameGas(c, T0);
        rr.sp.nY = 2;
        rr.sp.fuel = 0;
        rr.sp.ox = 1;
        rr.sp.stoich = m.s;
        rr.sp.lnA = m.lnA();
        rr.sp.Ta = m.theta() * T0;
        rr.sp.q = rr.tr.cp() * T0 * m.qOverCpT0();
        rr.sp.invPr = 1.0 / c.prandtl;
        rr.sp.openX = true;
        rr.Tu = T0;
        rr.sigma = m.sigma;
        rr.uIn = 1.0;
        rr.pB = T0;
        rr.etaIn = c.nscbcEtaIn;
        rr.outSigma = kDiffOutflowSigma;
        rr.thermalInflow = c.nscbcThermalInflow;
        rr.sinkNb = c.nscbcSink;
        rr.sinkAmp = c.nscbcSinkAmp;
        rr.dth = 1.0; // frame omega scale: the length unit l
        rr.cfl = c.cfl;
        rr.tEnd = c.tEnd;
        rr.dtRecompute = c.dtRecompute;
        rr.label = flameRunLabel(c);
        rr.diffusion = true;
        rr.diff = m;
        rr.geom = c.bunsen;
        // Candle: gravity along -x over the isothermal ambient column
        // p_a(x) (R T0 = p0 = T0): the inflow row's density at p_a(h/2), the
        // outflow target p_a at the last row, and the column's hydrostatic
        // dp/dx = -g rho_a there for the NSCBC waves. g == 0: the jet's own
        // numbers, bit for bit.
        const double gx = c.flameCandle ? c.flameGravity : 0.0;
        double pIn = T0;
        if (gx > 0) {
            pIn = candleAmbientP(T0, gx, T0, 0.5 * g.hx);
            const double pOut = candleAmbientP(T0, gx, T0, (g.nx - 0.5) * g.hx);
            rr.pB = pOut;
            rr.gravity = gx;
            rr.hydroLo = -gx * pIn / T0;
            rr.hydroHi = -gx * pOut / T0;
            // Air drawn down through the top is the ambient's.
            rr.sp.ambientHi = true;
            rr.sp.yHi[0] = 0.0;
            rr.sp.yHi[1] = m.yO2;
        }
        rr.candle = c.flameCandle;
        rr.quiescent = c.flameQuiescent;
        FlameInflowRows rows = diffusionInflowRows(c, gx, pIn);
        rr.rowU = std::move(rows.u);
        rr.rowJet = std::move(rows.jet);
        rr.rowTarget = std::move(rows.target);
        rr.rowRho = std::move(rows.rho);
        rr.rowY = std::move(rows.yF);
        rr.rowYO = std::move(rows.yO);
        rr.sp.yInRow[0] = rr.rowY;
        rr.sp.yInRow[1] = rr.rowYO;
        rr.ckpt = policy;
        auto res = gpu::runFlame(g, host, rhoY, rr, *out);
        if (!res) return die(res.error().code, res.error().message);
        return reportDone(*res, *out);
    }

    const double Tu = c.flameTu;
    // --mix-phi: the burnt state is that premix's (q stays the jet's).
    const double sig = c.flamePremix2 ? c.premix.sigmaAt(c.premixBurnPhi())
                                      : c.flameModel.sigma;
    gpu::FlameRun rr;
    // mu = Pr at Tref = Tu: lambda_u = mu cp / Pr = cp (delta_D = 1).
    rr.tr = flameGas(c, Tu);
    rr.sp.nY = 1;
    rr.sp.fuel = 0;
    rr.sp.lnA = std::log(c.flameEig.Lambda);
    rr.sp.Ta = c.flameModel.theta() * Tu;
    rr.sp.q = rr.tr.cp() * (sig - 1) * Tu;
    rr.sp.invPr = 1.0 / c.prandtl;
    rr.sp.openX = true;
    rr.sp.yIn[0] = 1.0;
    if (c.flamePremix2) {
        // Model C2: the rich jet on the two-reactant step (kChemCellPass2,
        // kSpeciesStage2<false, false>); q per unit fuel = cp Tu (sigma - 1)
        // s/Y_O1, fresh Y_F1, Y_O1 carried in.
        // Model C2b (--sl-law gri30): the A(phi) law rides the cell pass
        // (kChemCellPass2Phi) and chemStableDt. --mix-phi: the premix
        // carried in is the jet diluted with air (lean: all its fuel burns).
        const PartialPremixModel& pm = c.premix;
        PartialPremixModel burn = pm;
        burn.phi = c.premixBurnPhi();
        rr.sp.nY = 2;
        rr.sp.ox = 1;
        rr.sp.stoich = pm.s;
        rr.sp.q = rr.tr.cp() * Tu * pm.qOverCpTu();
        rr.sp.yIn[0] = burn.yF1();
        rr.sp.yIn[1] = burn.yO1();
        rr.sp.phiRate = c.premixLaw;
        rr.yFBurnt = burn.phi > 1 ? burn.yFrem() : 0.0;
        rr.premix2 = true;
        rr.premix = pm;
    }
    rr.Tu = Tu;
    rr.sigma = sig;
    rr.uIn = c.flameUIn;
    rr.pB = Tu - c.flameUIn * c.flameUIn * (sig - 1); // the IC's burnt p
    rr.etaIn = c.nscbcEtaIn;
    rr.thermalInflow = c.nscbcThermalInflow;
    rr.sinkNb = c.nscbcSink;
    rr.sinkAmp = c.nscbcSinkAmp;
    rr.dth = c.flameEig.dthOverDD;
    rr.cfl = c.cfl;
    rr.tEnd = c.tEnd;
    rr.dtRecompute = c.dtRecompute;
    rr.profileDt = c.profileDt;
    rr.label = flameRunLabel(c);
    rr.shock = c.shock;
    rr.shockDtAuto = c.shockDtAuto;
    if (c.flameTurb > 0) {
        rr.turbBox = std::move(turbBoxHost);
        rr.turbNbx = c.flameTurbNbx;
        rr.uPrime = c.flameTurb;
        rr.lt = c.flameTurbL;
        rr.hold = c.flameHold;
    }
    // Flame LES and TFLES (docs/chem.md, "LES species and TFLES").
    if (c.lesLabel) {
        rr.csLes = c.cs;
        rr.sgs = c.sgsModel;
    }
    if (c.flameTfles > 0) {
        rr.tflesOn = true;
        rr.tfles = flameTflesSetup(c, sig, c.flameTflesDilate);
    }
    if (c.flameBunsen) {
        // The slot burner: per-row inflow (dns/bunsen.hpp) and p = p0 at the
        // outflow (the cone's pressure jump is O(M^2) of p0, not the 1D one).
        rr.bunsen = true;
        rr.geom = c.bunsen;
        rr.pB = Tu;
        // Rows at j + ny k: the z-invariant slot repeated over the span, or
        // the round nozzle's r(y, z) profile.
        FlameInflowRows rows = bunsenInflowRows(c);
        rr.rowU = std::move(rows.u);
        rr.rowJet = std::move(rows.jet);
        rr.rowTarget = std::move(rows.target);
        rr.rowRho = std::move(rows.rho);
        rr.rowY = std::move(rows.yF);
        rr.sp.yInRow[0] = rr.rowY;
        // --turb: the inflow sees the box times the jet weight J(y) (the
        // coflow stays laminar); the IC's own field took Y (above).
        if (c.flameTurb > 0) bunsenTurbJetWeight(rr.turbBox, rr.rowJet, rr.turbNbx);
    }

    rr.ckpt = policy;
    auto res = gpu::runFlame(g, host, std::span(rhoY, c.flamePremix2 ? 2 : 1),
                             rr, *out);
    if (!res) return die(res.error().code, res.error().message);
    return reportDone(*res, *out);
#else
    return exitNoGpu;
#endif
}



} // namespace dns::drive
