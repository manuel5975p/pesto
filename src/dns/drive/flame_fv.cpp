// dns flame on a FaceMesh: compressible FV (premixed, C2b, jet, candle) and low-Mach FV.
#include "dns/drive/internal.hpp"
#include "dns/fvflame_case.hpp"

#include "dns/checkpoint.hpp"
#include "dns/config.hpp"
#include "dns/flame1d.hpp"
#include "dns/facemesh.hpp"
#include "dns/fvlowmach.hpp"
#include "dns/fvshock.hpp"
#include "dns/fvturb.hpp"
#include "dns/lowmach.hpp"
#include "dns/output.hpp"
#include "dns/transport.hpp"

#if DNS_WITH_GPU
#include "dns/gpu/runner_fvlowmach.hpp"
#include "dns/gpu/comp.hpp"
#include "dns/gpu/runner_fvcomp.hpp"
#endif

#include <algorithm>
#include <cstdio>
#include <format>
#include <numbers>
#include <optional>
#include <string>
#include <vector>

namespace dns::drive {

using std::numbers::pi;
#if DNS_WITH_GPU
namespace {

// dns flame --solver fv (docs/fvcomp.md "Species"): the premixed flame on a
// FaceMesh. The flags' uniform box (fresh Far, burnt Outflow, y and z as the
// structured box) or the polyMesh of --bc. Box units as the structured flame.
// dns flame --mode diffusion on a FaceMesh (docs/fvcomp.md "Diffusion flame"): the
// jet's per-face rows, the structured box's units, mesh and IC. --mode candle: the same
// jet under gravity along -dir over the hydrostatic ambient column (docs/fvcomp.md
// "Hydrostatic ambient": its open faces' rows, the IC's p); g 0 = the jet bit for bit.
int runFvJet(const Case& c, const FaceMesh& fm, const FvBcSpec& spec, std::array<double, 3> origin,
             const CheckpointContext& ckx)
{
    auto restart = readFvRestart(c, fm, 2);
    if (!restart) return restart.error();
    const DiffusionModel& md = c.diff;
    const FvJetSetup js = fvJetSetup(c, spec, origin);
    const double gx = c.flameCandle ? c.flameGravity : 0.0;
    auto probe = FvJetProbe::make(fm, js.js);
    if (!probe) return die(exitInput, probe.error());
    FvJetRows rows = fvJetCaseRows(c, fm, spec, js);
    auto raster = FvRaster::make(fm, {.longSide = c.fvFrameN, .box = c.fvFrameBox});
    if (!raster) return die(exitInput, raster.error());
    FrameGeometry fgeo = FrameGeometry::centred(raster->grid(), 1.0);
    fgeo.mask = raster->mask();
    RunMeta meta = c.meta;
    meta.setGrid(raster->grid());
    meta.farField = "fv";
    std::string json = std::format("{{\"app\":\"dns\",\"flame\":\"{}\",\"solver\":\"fv\",\"sigma\":{},\"beta\":{},\"mach\":{}",
                                   c.flameCandle ? "candle" : "diffusion", md.sigma, md.beta, c.mach);
    if (c.flameCandle) json += std::format(",\"gravity\":{},\"quiescent\":{}", gx, c.flameQuiescent);
    auto out = RunOutput::make(outputConfig(c, json),
                               meta, raster->grid(), std::move(fgeo));
    if (!out) return die(exitOutput, out.error());
    gpu::FvCompRun rr{.label = fvFlameLabel(c),
                      .solver = {.tr = js.gas,
                                 .bc = std::move(rows.bc),
                                 .bcRow = std::move(rows.faceRow),
                                 .precision = c.fvPrecision,
                                 .compensated = c.compensated,
                                 .chem = js.chem,
                                 .tallied = std::vector<std::uint8_t>(fm.patches.size(), 0),
                                 .pRef = js.fs.p,
                                 .gravity = js.g,
                                 .ambient = gx > 0 ? std::optional<FvAmbient>(fvJetAmbient(js.js)) : std::nullopt},
                      .spec = spec,
                      .fs = js.fs,
                      .refArea = 1.0,
                      .cfl = c.cfl,
                      .tEnd = c.tEnd,
                      .dtRecompute = c.dtRecompute,
                      .quiet = c.quiet,
                      .jet = &*probe,
                      .ckpt = flamePolicy(c, ckx, *restart)};
    auto res = gpu::runFvComp(fm, fvJetState(fm, js.js, c.flameIgnition), rr, *raster, *out);
    if (!res) return die(res.error().code, res.error().message);
    if (auto e = out->finish(); !e) return die(exitOutput, e.error());
    std::printf("done: %ld steps, %.3f ms/step\n", res->steps, res->msPerStep);
    return 0;
}
} // namespace


int runFvFlame(const Case& c, const CheckpointContext& ckx)
{
    const bool poly = c.fvMode == FvCompMode::PolyMesh;
    std::expected<FaceMesh, std::string> mesh = std::unexpected("");
    FvBcSpec spec;
    if (poly) {
        if (auto two = fvPolyCheck(c); !two) return die(two.error().first, two.error().second);
        auto pm = readPolyMesh(fvPolyDir(c.fvMeshDir));
        if (!pm) return die(exitInput, pm.error());
        mesh = FaceMesh::fromPolyMesh(*pm, CellOrder::Morton);
        spec = c.fvBc;
    } else {
        FvFlameBox box = fvFlameBox(c);
        mesh = FaceMesh::fromCurvMesh(CurvMesh::uniform(c.grid), box.ss, CellOrder::Morton);
        spec = std::move(box.spec);
    }
    if (!mesh) return die(exitInput, "FaceMesh: " + mesh.error());
    const FaceMesh& fm = *mesh;
    // --shock reads mesh lines: a cell with a face and no opposite is refused (masterplan_combnozzle R8).
    if (poly && c.shock != ShockMode::None)
        if (auto ok = fvShockMeshCheck(fm); !ok) return die(exitInput, c.fvMeshDir + ": " + ok.error());
    if (!haveGpu(c)) return die(exitNoGpu, "no GPU device (dns flame is GPU-only)");
    if (c.flameDiffusion)
        return runFvJet(c, fm, spec, poly ? spec.origin : std::array<double, 3>{0, c.bunsen.yc, 0}, ckx);
    auto restart = readFvRestart(c, fm, c.flamePremix2 ? 2 : 1);
    if (!restart) return restart.error();
    const FvFlameSetup fs = fvFlameSetup(c, spec);
    const double sig = fs.burnt.sigma;
    auto probe = FvFlameProbe::make(fm, fs.fl);
    if (!probe) return die(exitInput, probe.error());
    // Species LES and the thickened flame (docs/fvcomp.md "Species LES and TFLES"): the
    // structured flame's setup; a polyMesh measures its cells per delta_th off the probe.
    if (c.lesLabel && fm.dim != 3)
        return die(exitInput, "--les needs a 3D mesh: 2D LES is regularized 2D Euler, not a filtered turbulent field");
    std::optional<TflesSetup> tfles;
    if (c.flameTfles > 0) {
        auto t = fvFlameTfles(c, sig, probe->slab());
        if (!t) return die(exitInput, t.error());
        tfles = *t;
    }
    std::vector<FvBc> bcTable = fvBcsOf(fm, spec, fs.fs, 1.0);
    // masterplan_combnozzle P4: a [nozzle] run starts from the composition design (docs/nozzle.md "Reacting").
    FvState ic;
    if (spec.nozzle) {
        auto st = fvFlameNozzleStart(c, fm, spec, fs, *probe, bcTable);
        if (!st) return die(exitInput, st.error());
        const FvReactingDesign& d = st->design;
        if (!c.quiet)
            std::printf("# reacting nozzle design: eps_c %.10g, M_u %.10g, u_u %.10g, rho_u %.10g, T0b %.10g, p0b %.10g, "
                        "p0b/p0u %.10g, M_b %.10g, M_e %.10g, CFvac %.10g, A_ref %.10g (vol/L %.10g), chamber %d of %d slabs\n",
                        probe->area() / spec.nozzle->astar, d.Mu, d.uu, d.rhou, d.b.T0b, d.b.p0b,
                        d.b.p0b / (d.pu * std::pow(1 + 0.5 * (fs.fs.gamma - 1) * d.Mu * d.Mu, fs.fs.gamma / (fs.fs.gamma - 1))),
                        d.b.Mb, d.Me, d.CFvac, probe->area(), probe->meanArea(), probe->chamberBins(), probe->bins());
        ic = std::move(st->ic);
    } else {
        ic = fvFlameIC(c, fm, spec, fs);
    }
    // --turb (M5, docs/fvcomp.md "Turbulence"): the structured box's field at the
    // centroids (IC) and swept past the xlo faces (per-face Far rows after the patch rows).
    std::optional<FvTurb> turb;
    std::optional<TurbSweep> turbSweep;
    std::vector<int> bcRow;
    const int turbFirst = int(bcTable.size());
    if (c.flameTurb > 0) {
        turb = fvFlameTurb(c);
        fvAddTurbulence(fm, *turb, ic);
        auto rows = fvFaceRows(fm, bcTable, "xlo");
        if (!rows) return die(exitInput, rows.error());
        turbSweep = TurbSweep::make(turb->modes, turb->spec, rows->faces.cf);
        bcRow = std::move(rows->bcRow);
    }
    auto raster = FvRaster::make(fm, {.longSide = c.fvFrameN, .box = c.fvFrameBox});
    if (!raster) return die(exitInput, raster.error());
    FrameGeometry fgeo = FrameGeometry::centred(raster->grid(), 1.0);
    fgeo.mask = raster->mask();
    RunMeta meta = c.meta;
    meta.setGrid(raster->grid());
    meta.farField = "fv";
    // The C0 .json text bit for bit; C2 appends its facts (the low-Mach route's keys).
    std::string json = std::format("{{\"app\":\"dns\",\"flame\":\"premixed\",\"solver\":\"fv\","
                                   "\"sigma\":{},\"beta\":{},\"lambda\":{},\"mach\":{}",
                                   sig, c.flameModel.beta, c.flameEig.Lambda, c.mach);
    if (c.flamePremix2)
        json += std::format(",\"model\":\"{}\",\"phi\":{},\"zst\":{}", c.premixLaw.n > 0 ? "C2b" : "C2", c.premix.phi,
                            c.premix.zst());
    if (c.fvGamma != kGammaAir) json += std::format(",\"gamma\":{}", c.fvGamma);
    auto out = RunOutput::make(outputConfig(c, json), meta, raster->grid(), std::move(fgeo));
    if (!out) return die(exitOutput, out.error());
    gpu::FvCompRun rr{.label = fvFlameLabel(c),
                      .solver = {.tr = fs.gas,
                                 .bc = std::move(bcTable),
                                 .bcRow = std::move(bcRow),
                                 .precision = c.fvPrecision,
                                 .csLes = c.lesLabel ? c.cs : 0.0,
                                 .sgs = c.sgsModel,
                                 .compensated = c.compensated,
                                 .chem = fs.chem,
                                 .tfles = tfles,
                                 .tallied = spec.nozzle ? fvTallyOf(fm, spec) : std::vector<std::uint8_t>(fm.patches.size(), 0),
                                 .pRef = fs.fs.p,
                                 .turb = turb.has_value(),
                                 .shock = {.mode = c.shock, .eps4 = c.shockEps4}},
                      .spec = spec,
                      .fs = fs.fs,
                      .refArea = probe->area(),
                      .cfl = c.cfl,
                      .tEnd = c.tEnd,
                      .dtRecompute = c.dtRecompute,
                      .quiet = c.quiet,
                      .flame = &*probe,
                      .profileDt = c.profileDt,
                      .ckpt = flamePolicy(c, ckx, *restart)};
    if (c.flamePremix2) rr.premix = c.premix;
    rr.shockDtAuto = c.shockDtAuto;
    // masterplan_combnozzle P3: [nozzle] and [thrust] on the hll flame route (validate: no rest
    // state, total with y). The nozzle columns and the ledger as dns flow's; P4: the burnt station's
    // columns (p0b T0b CdB CfBvac CfB).
    if (spec.nozzle) {
        const auto total = std::ranges::find(spec.patches, FvPatchKind::Total, &FvPatchSpec::kind);
        rr.nozzle = FvNozzleRun{.astar = spec.nozzle->astar,
                                .axis = spec.nozzle->axis,
                                .pRef = rr.solver.pRef,
                                .p0 = total == spec.patches.end() ? 0.0 : total->p0 * fs.fs.p,
                                .t0 = total == spec.patches.end() ? 0.0 : total->t0 * fs.fs.T};
        if (spec.thrust)
            rr.thrust = FvThrustCase{.axis = rr.nozzle->axis,
                                     .pa = spec.thrust->pa * fs.fs.p,
                                     .p0 = rr.nozzle->p0,
                                     .t0 = rr.nozzle->t0,
                                     .astar = rr.nozzle->astar,
                                     .gamma = fs.fs.gamma,
                                     .Rgas = kRgasBox,
                                     .station = *spec.thrust->station};
    }
    if (turbSweep) {
        rr.turbSweep = &*turbSweep;
        rr.turbPatch = &spec.patches[0];
        rr.turbFirst = turbFirst;
    }
    auto res = gpu::runFvComp(fm, ic, rr, *raster, *out);
    if (!res) return die(res.error().code, res.error().message);
    if (auto e = out->finish(); !e) return die(exitOutput, e.error());
    std::printf("done: %ld steps, %.3f ms/step\n", res->steps, res->msPerStep);
    return 0;
}

int runFvLowMachCase(const Case& c, const CheckpointContext& ckx)
{
    const bool poly = c.fvMode == FvCompMode::PolyMesh;
    const bool burner = c.flameBunsen;
    const BunsenGeom& bg = c.bunsen;
    const double Tu = c.flameTu, m = burner ? bg.uJet : c.flameUIn;
    const FlameBurnt burnt = flameBurnt(c);
    const PartialPremixModel& burn = burnt.burn;
    const double sig = burnt.sigma;
    const FvBcSpec spec = fvLowMachCaseSpec(c);
    std::expected<FaceMesh, std::string> mesh = std::unexpected("");
    if (poly) {
        if (auto two = fvPolyCheck(c); !two) return die(two.error().first, two.error().second);
        auto pm = readPolyMesh(fvPolyDir(c.fvMeshDir));
        if (!pm) return die(exitInput, pm.error());
        mesh = FaceMesh::fromPolyMesh(*pm, CellOrder::Morton);
    } else {
        mesh = fvLowMachCaseBox(c);
    }
    if (!mesh) return die(exitInput, "FaceMesh: " + mesh.error());
    const FaceMesh& fm = *mesh;
    auto bcs = fvLowMachBcs(spec, m, Tu);
    if (!bcs) return die(exitInput, bcs.error());
    auto restart = readFvRestart(c, fm, c.flamePremix2 ? 2 : 1);
    if (!restart) return restart.error();
    // M4 (docs/fvincomp.md "Low-Mach LES"): the compressible FV flame's LES mesh rule.
    if (c.lesLabel && fm.dim != 3)
        return die(exitInput, "--les needs a 3D mesh: 2D LES is regularized 2D Euler, not a filtered turbulent field");
    if (c.flameTfles > 0 && poly && fm.dim != 3 && c.flameCharlette)
        return die(exitInput, "--efficiency charlette needs a 3D mesh (and --les)");
    if (!haveGpu(c)) return die(exitNoGpu, "no GPU device (dns flame is GPU-only)");
    FvBurnerFrame frame;
    if (burner) {
        auto f = fvBurnerFrame(c, fm, spec);
        if (!f) return die(exitInput, f.error());
        frame = *f;
    }
    FvLowMachIC ic = fvLowMachCaseIC(c, fm, spec, frame, sig);
    // --turb (M5, docs/chem.md "Low-Mach turbulence"): the planar box only (validate);
    // the box at the centroids (IC), swept past the xlo faces (the runner's rows).
    std::optional<FvTurb> turb;
    if (c.flameTurb > 0) {
        turb = fvFlameTurb(c);
        fvLowMachAddTurbulence(fm, *turb, ic);
    }
    FvLowMachModel model = fvLowMachCaseModel(c, spec, sig);
    const FvFlameSpec fl{.dir = spec.dir, .Tu = Tu, .sigma = sig, .dth = c.flameEig.dthOverDD, .gamma = kGammaAir,
                         .Rgas = kRgasBox, .q = model.tr.cp() * model.qcp, .fuel = 0,
                         .yBurnt = c.flamePremix2 && burn.phi > 1 ? burn.yFrem() : 0.0};
    auto probe = FvFlameProbe::make(fm, fl);
    if (!probe) return die(exitInput, probe.error());
    // M4: TFLES on faces (the structured route's setup; a polyMesh measures its cells per
    // delta_th off the probe).
    if (c.flameTfles > 0) {
        auto t = fvFlameTfles(c, sig, probe->slab());
        if (!t) return die(exitInput, t.error());
        model.tfles = *t;
    }
    auto raster = FvRaster::make(fm, {.longSide = c.fvFrameN, .box = c.fvFrameBox});
    if (!raster) return die(exitInput, raster.error());
    FrameGeometry fgeo = FrameGeometry::centred(raster->grid(), 1.0);
    fgeo.mask = raster->mask();
    RunMeta meta = c.meta;
    meta.setGrid(raster->grid());
    meta.farField = "fv";
    // The planar flame's .json text bit for bit; C2 and the burner append their facts.
    std::string json = std::format("{{\"app\":\"dns\",\"flame\":\"{}\",\"physics\":\"lowmach\",\"solver\":\"fv\","
                                   "\"sigma\":{},\"beta\":{},\"lambda\":{},\"mach\":0",
                                   burner ? "bunsen" : "premixed", sig, c.flameModel.beta, c.flameEig.Lambda);
    if (c.flamePremix2) json += std::format(",\"model\":\"C2\",\"phi\":{},\"zst\":{}", c.premix.phi, c.premix.zst());
    if (burner)
        json += std::format(",\"nozzle\":\"slot\",\"W\":{},\"uJet\":{},\"yc\":{},\"dth\":{}", bg.W, bg.uJet, bg.yc,
                            c.flameEig.dthOverDD);
    if (burner && c.lmOpenY) json += ",\"laterals\":\"open\"";
    if (c.lmGravity > 0) json += std::format(",\"gravity\":{}", c.lmGravity);
    if (burner && c.lmRim)
        json += std::format(",\"rim\":{{\"x\":{},\"ri\":{},\"ro\":{},\"h\":{},\"T\":{}}}", bg.xRim, 0.5 * bg.W,
                            0.5 * bg.W + bg.rimLip, c.h, bg.tRim);
    auto out = RunOutput::make(outputConfig(c, json), meta, raster->grid(), std::move(fgeo));
    if (!out) return die(exitOutput, out.error());
    gpu::FvLowMachRun rr{.model = std::move(model),
                         .ctl = {.tol = c.fvTol, .precision = c.fvPrecision},
                         .bcs = std::move(*bcs),
                         .cfl = c.cfl,
                         .chemK = c.lmChemDt,
                         .tEnd = c.tEnd,
                         .dtRecompute = c.dtRecompute,
                         .profileDt = c.profileDt,
                         .quiet = c.quiet,
                         .label = fvFlameLabel(c),
                         .flame = &*probe};
    if (c.flamePremix2) {
        const PartialPremixModel& pm = c.premix;
        rr.premix2 = true;
        rr.yFresh = burn.yF1();
        rr.zS = pm.s;
        rr.zO2 = pm.yO2;
        rr.zDen = pm.s * pm.yFrem() + pm.yO2;
        rr.zIn = (pm.s * burn.yF1() - burn.yO1() + pm.yO2) / rr.zDen;
    }
    if (c.lmDtCell) rr.posK = kLmPosKDefault;
    rr.ckpt = flamePolicy(c, ckx, *restart);
    // M4: LES on faces.
    if (c.lesLabel) {
        rr.model.cs = c.cs;
        rr.model.sgs = c.sgsModel;
    }
    if (burner) {
        auto& B = rr.burner;

        B.on = true;
        B.geom = bg;
        B.premix = c.premix;
        B.h = c.h;
        B.depth = frame.depth;
        B.lo = {frame.origin[0], frame.origin[1] - bg.yc};
        B.rim = !poly && c.lmRim;
        B.patch = poly ? std::ranges::find_if(spec.patches, &FvPatchSpec::jet)->name : std::string("xlo");
        B.row = fvBurnerRow(c, frame);
    }
    if (turb) {
        rr.turb = &*turb;
        rr.turbPatch = "xlo";
        rr.turbU = m;
    }
    auto res = gpu::runFvLowMach(fm, ic, std::move(rr), *raster, *out);
    if (!res) return die(res.error().code, res.error().message);
    if (auto e = out->finish(); !e) return die(exitOutput, e.error());
    std::printf("done: %ld steps, %.3f ms/step\n", res->steps, res->msPerStep);
    return 0;
}
#endif // DNS_WITH_GPU

} // namespace dns::drive
