#include "dns/fvflame_case.hpp"

#include "dns/chem.hpp"
#include "dns/comp.hpp"
#include "dns/flame1d.hpp"
#include "dns/lowmach.hpp"
#include "dns/turb_ic.hpp"

#include <cmath>
#include <format>
#include <span>
#include <utility>
#include <vector>

namespace dns {

std::expected<TflesSetup, std::string> fvFlameTfles(const Case& c, double sigma, double slab)
{
    const bool poly = c.fvMode == FvCompMode::PolyMesh;
    const double cells = poly ? c.flameEig.dthOverDD / slab : c.flameCells;
    if (c.flameTfles * cells < 4)
        return std::unexpected(std::format("--tfles {:g}: {:.3g} cells across the thickened flame on this mesh (F "
                                           "cells per delta_th >= 4 required)",
                                           c.flameTfles, c.flameTfles * cells));
    return flameTflesSetup(c, sigma,
                           poly ? int(std::ceil(kTflesDilate * c.flameTfles * cells - 1e-9)) : c.flameTflesDilate);
}

FvTurb fvFlameTurb(const Case& c)
{
    const TurbSpec ts = flameTurbSpec(c);
    return FvTurb{.spec = ts, .modes = turbModes(ts), .xw = c.flameXf0 - 1.5 * c.flameTurbL, .ew = 0.25 * c.flameTurbL};
}

// --- compressible -----------------------------------------------------------------

FvFlameBox fvFlameBox(const Case& c)
{
    const Grid& g = c.grid;
    const FoamPatchType zt = g.nz == 1 ? FoamPatchType::Empty : FoamPatchType::Cyclic;
    FvFlameBox b{.ss = {.lo = {0, 0, 0},
                        .hi = {g.nx - 1, g.ny - 1, g.nz - 1},
                        .name = {"xlo", "xhi", "ylo", "yhi", "back", "front"},
                        .type = {FoamPatchType::Patch, FoamPatchType::Patch, FoamPatchType::Cyclic,
                                 FoamPatchType::Cyclic, zt, zt}},
                  .spec = {}};
    b.spec.patches = {{.name = "xlo", .kind = FvPatchKind::Far, .hasY = true, .y = {1, 0, 0, 0}},
                      {.name = "xhi", .kind = FvPatchKind::Outflow, .burnt = true}};
    if (c.flameDiffusion)
        b.spec.patches = {{.name = "xlo", .kind = FvPatchKind::Inflow, .jet = true},
                          {.name = "xhi", .kind = FvPatchKind::Outflow}};
    if (c.flameCandle && c.flameGravity > 0)
        b.spec.patches[1] = {.name = "xhi", .kind = FvPatchKind::Far, .hasY = true, .y = {0, c.diff.yO2, 0, 0}};
    return b;
}

FvFlameSetup fvFlameSetup(const Case& c, FvBcSpec& spec)
{
    const bool poly = c.fvMode == FvCompMode::PolyMesh;
    const FlameBurnt burnt = flameBurnt(c);
    const PartialPremixModel& burn = burnt.burn;
    const double Tu = c.flameTu, m = c.flameUIn, sig = burnt.sigma;
    const std::array<double, 3> dir = poly ? c.fvBc.dir : std::array<double, 3>{1, 0, 0};
    // mu = Pr at Tref = Tu: lambda_u = mu cp / Pr = cp (delta_D = 1). --gamma (polyMesh
    // C0 only, else kGammaAir): one gamma for fresh and burnt gas, q = cp(gamma) (sigma - 1) Tu.
    const double gam = c.fvGamma;
    FvFlameSetup s{.burnt = burnt,
                   .gas = Transport::air(c.prandtl, Tu, c.tinfK, c.prandtl, gam, kRgasBox, c.sutherland),
                   .chem = {},
                   .fs = {.rho = 1.0, .U = m, .p = Tu, .T = Tu, .gamma = gam, .dir = dir,
                          .pBurnt = Tu - m * m * (sig - 1), .tBurnt = sig * Tu},
                   .fl = {}};
    s.chem = FvChem{.nY = 1,
                    .fuel = 0,
                    .lnA = std::log(c.flameEig.Lambda),
                    .Ta = c.flameModel.theta() * Tu,
                    .q = s.gas.cp() * (sig - 1) * Tu,
                    .invPr = 1.0 / c.prandtl};
    if (c.flamePremix2) {
        // Model C2: F + s O, q per unit fuel cp Tu (sigma - 1) s / Y_O1 (the jet's). C2b:
        // the A(phi) law (premixLaw.n > 0).
        s.chem.nY = 2;
        s.chem.ox = 1;
        s.chem.stoich = c.premix.s;
        s.chem.q = s.gas.cp() * Tu * c.premix.qOverCpTu();
        s.chem.law = c.premixLaw;
        const bool rich = burn.phi > 1;
        for (FvPatchSpec& p : spec.patches) {
            if (p.kind == FvPatchKind::Far || p.kind == FvPatchKind::Inflow) p.y = {burn.yF1(), burn.yO1(), 0, 0};
            if (p.kind == FvPatchKind::Outflow) p.y = {rich ? burn.yFrem() : 0.0, rich ? 0.0 : burn.yOrem(), 0, 0};
        }
    }
    s.fl = FvFlameSpec{.dir = dir, .Tu = Tu, .sigma = sig, .dth = c.flameEig.dthOverDD, .gamma = gam,
                       .Rgas = kRgasBox, .q = s.chem.q, .fuel = 0,
                       .yBurnt = c.flamePremix2 && burn.phi > 1 ? burn.yFrem() : 0.0};
    return s;
}

FvState fvFlameIC(const Case& c, const FaceMesh& fm, const FvBcSpec& spec, const FvFlameSetup& s)
{
    const int nY = s.chem.nY;
    if (c.fvMode == FvCompMode::PolyMesh) {
        const FlameEigen& tab = c.flameEig.tab;
        const double xf = c.flameXf0, F = c.flameTfles > 1 ? c.flameTfles : 1.0; // TFLES: the oracle thickened by F
        FvState ic =
            fvFlameState(fm, s.fl, spec.origin, c.flameUIn, nY, [&](double x) { return flameProfile(tab, (x - xf) / F); });
        if (c.flamePremix2) {
            // Row 5 holds rho s_p: flameIC2's progress -> reactants rule.
            const std::size_t n = std::size_t(fm.nC);
            const std::vector<double> rho(ic.var(0), ic.var(0) + n);
            std::vector<double> rhoY[2] = {{}, std::vector<double>(ic.var(kFvMeanVars), ic.var(kFvMeanVars) + n)};
            premix2Reactants(c, rho, rhoY);
            for (std::size_t k = 0; k < 2; ++k) std::ranges::copy(rhoY[k], ic.var(kFvMeanVars + int(k)));
        }
        return ic;
    }
    CompState host = CompState::zeros(c.grid);
    std::vector<double> rhoY[2];
    if (c.flamePremix2) flameIC2(c, host, rhoY);
    else flameIC(c, host, rhoY[0]);
    return fvStateOf(fm, host, std::span<const std::vector<double>>(rhoY, std::size_t(nY)));
}

FvJetSetup fvJetSetup(const Case& c, const FvBcSpec& spec, std::array<double, 3> origin)
{
    const DiffusionModel& md = c.diff;
    const double T0 = c.flameTu;
    FvJetSetup s{.gas = Transport::air(c.prandtl, T0, c.tinfK, c.prandtl, kGammaAir, kRgasBox, c.sutherland),
                 .chem = {},
                 .fs = {.rho = 1.0, .U = 1.0, .p = T0, .T = T0, .gamma = kGammaAir, .dir = spec.dir},
                 .g = {},
                 .js = {}};
    s.chem = FvChem{.nY = 2,
                    .fuel = 0,
                    .lnA = md.lnA(),
                    .Ta = md.theta() * T0,
                    .q = s.gas.cp() * T0 * md.qOverCpT0(),
                    .invPr = 1.0 / c.prandtl,
                    .ox = 1,
                    .stoich = md.s};
    const double gx = c.flameCandle ? c.flameGravity : 0.0;
    s.g = {-gx * spec.dir[0], -gx * spec.dir[1], -gx * spec.dir[2]};
    s.js = FvJetSpec{.dir = spec.dir, .lift = spec.lift, .origin = origin, .model = md, .geom = c.bunsen, .T0 = T0,
                     .gamma = kGammaAir, .Rgas = kRgasBox, .q = s.chem.q, .skip = std::max(c.nscbcSink, 4),
                     .candle = c.flameCandle, .quiescent = c.flameQuiescent, .g = s.g};
    return s;
}

FvJetRows fvJetCaseRows(const Case& c, const FaceMesh& fm, const FvBcSpec& spec, const FvJetSetup& s)
{
    const double T0 = c.flameTu;
    std::vector<FvBc> bc0 = fvBcsOf(fm, spec, s.fs, 1.0);
    const double uFar = c.flameQuiescent ? 0.0 : c.bunsen.uCo;
    for (FvBc& b : bc0)
        if (c.flameCandle && b.kind == FvBcKind::Far) {
            for (std::size_t k = 0; k < 3; ++k) b.qFar[1 + k] = b.qFar[0] * uFar * spec.dir[k];
            b.qFar[4] = T0 / (kGammaAir - 1.0) + 0.5 * b.qFar[0] * uFar * uFar;
        }
    return fvJetRows(fm, spec, std::move(bc0), s.js);
}

// --- low-Mach ---------------------------------------------------------------------

FvBcSpec fvLowMachCaseSpec(const Case& c)
{
    if (c.fvMode == FvCompMode::PolyMesh) return c.fvBc;
    FvBcSpec spec = fvLowMachBoxBc(c.lmOpenY, 1.0, {0.0, c.flamePremix2 ? c.premix.yO2 : 0.0});
    if (c.flamePremix2) {
        const PartialPremixModel burn = flameBurnt(c).burn;
        spec.patches[0].y = {burn.yF1(), burn.yO1(), 0, 0};
    }
    if (c.lmRim)
        spec.patches.push_back({.name = "rim", .kind = FvPatchKind::Wall, .isothermal = true, .tWall = c.bunsen.tRim});
    return spec;
}

std::expected<FaceMesh, std::string> fvLowMachCaseBox(const Case& c)
{
    return fvLowMachBox(c.grid, c.lmOpenY, c.lmRim ? lmTubeSolid(c.bunsen, c.grid) : std::vector<std::uint8_t>{});
}

std::array<double, 2> FvBurnerFrame::at(const std::array<double, 3>& p) const
{
    double x = 0, y = 0;
    for (std::size_t k = 0; k < 3; ++k) {
        x += (p[k] - origin[k]) * dir[k];
        y += (p[k] - origin[k]) * lift[k];
    }
    y += yc;
    return {x, y};
}

std::expected<FvBurnerFrame, std::string> fvBurnerFrame(const Case& c, const FaceMesh& fm, const FvBcSpec& spec)
{
    FvBurnerFrame f{.dir = spec.dir, .origin = {0.0, c.bunsen.yc, 0.0}, .lift = {0.0, 1.0, 0.0}, .yc = c.bunsen.yc,
                    .depth = c.grid.hz};
    if (c.fvMode != FvCompMode::PolyMesh) return f;
    f.origin = spec.origin;
    f.lift = spec.lift;
    if (fm.emptyAxis != 2) return std::unexpected("--mode bunsen on a polyMesh: a 2D mesh, empty along z");
    double vol = 0, ez = 0;
    for (double v : fm.V) vol += v;
    for (const MeshPatch& mp : fm.patches)
        if (mp.type == FoamPatchType::Empty)
            for (int q = mp.first; q < mp.first + mp.count; ++q) ez += std::abs(fm.S[2][std::size_t(fm.nI + q)]);
    if (!(ez > 0)) return std::unexpected("--mode bunsen on a polyMesh: no empty faces to read the depth from");
    f.depth = vol / (0.5 * ez);
    return f;
}

namespace {

// lowmachIC's structured state cell for cell (fm from fvLowMachBox: srcCell = Grid::idx);
// rho u = the MAC x faces' m (mac) or the uniform m.
FvLowMachIC lowMachICOfGrid(const Case& c, const FaceMesh& fm, bool mac, double m)
{
    const Grid& g = c.grid;
    const LowMachState h = lowmachIC(c);
    const std::size_t nC = std::size_t(fm.nC);
    FvLowMachIC ic;
    ic.rho.resize(nC);
    ic.ry0.resize(nC);
    if (c.flamePremix2) ic.ry1.resize(nC);
    ic.p.resize(nC);
    ic.ru.assign(3 * nC, 0.0);
    for (std::size_t q = 0; q < nC; ++q) {
        const std::size_t src = std::size_t(fm.srcCell[q]);
        ic.rho[q] = h.rho[src];
        ic.ry0[q] = h.rhoY[0][src];
        if (c.flamePremix2) ic.ry1[q] = h.rhoY[1][src];
        ic.p[q] = h.p[src];
        if (mac) {
            const std::size_t i = src % std::size_t(g.nx), row = src / std::size_t(g.nx);
            ic.ru[q] = h.mx[i + std::size_t(g.nx + 1) * row];
        } else {
            ic.ru[q] = m;
        }
    }
    return ic;
}

} // namespace

FvLowMachIC fvLowMachCaseIC(const Case& c, const FaceMesh& fm, const FvBcSpec& spec, const FvBurnerFrame& frame,
                            double sigma)
{
    const bool poly = c.fvMode == FvCompMode::PolyMesh;
    const double Tu = c.flameTu, m = c.flameBunsen ? c.bunsen.uJet : c.flameUIn;
    const std::size_t nC = std::size_t(fm.nC);
    if (c.flameBunsen && poly) {
        FvLowMachIC ic;
        ic.rho.resize(nC);
        ic.ry0.resize(nC);
        if (c.flamePremix2) ic.ry1.resize(nC);
        ic.p.assign(nC, 0.0);
        ic.ru.assign(3 * nC, 0.0);
        for (std::size_t q = 0; q < nC; ++q) {
            const auto [x, y] = frame.at({fm.C[0][q], fm.C[1][q], fm.C[2][q]});
            const LmBunsenCell b = lmBunsenPoint(c, x, y, 0.5 * c.grid.hz);
            ic.rho[q] = b.rho;
            ic.ry0[q] = b.ryF;
            if (c.flamePremix2) ic.ry1[q] = b.ryO;
            for (std::size_t k = 0; k < 3; ++k) ic.ru[k * nC + q] = b.m * spec.dir[k];
        }
        return ic;
    }
    // lowmachBunsenIC cell for cell: rho u = the row's m (the MAC x faces' value).
    if (c.flameBunsen) return lowMachICOfGrid(c, fm, true, m);
    if (!poly) return lowMachICOfGrid(c, fm, false, m);
    const FlameEigen& tab = c.flameEig.tab;
    const double xf = c.flameXf0;
    // --tfles F (M4): the oracle thickened by F.
    const double F = c.flameTfles > 1 ? c.flameTfles : 1.0;
    FvLowMachIC ic = fvLowMachFlameIC(fm, spec.dir, spec.origin, sigma, m, Tu, [&](double s) {
        return flameProfile(tab, F > 1 ? (s - xf) / F : s - xf);
    });
    if (c.flamePremix2) {
        // ry0 = rho s_p: flameIC2's progress -> reactants rule.
        std::vector<double> rhoY[2] = {{}, std::move(ic.ry0)};
        premix2Reactants(c, ic.rho, rhoY);
        ic.ry0 = std::move(rhoY[0]);
        ic.ry1 = std::move(rhoY[1]);
    }
    return ic;
}

FvLowMachModel fvLowMachCaseModel(const Case& c, const FvBcSpec& spec, double sigma)
{
    const double Tu = c.flameTu;
    // q/cp per unit fuel: C0 (sigma - 1) Tu, C2 Tu (sigma - 1) s / Y_O1 (gpu/lowmach's).
    const double qcp = c.flamePremix2 ? Tu * c.premix.qOverCpTu() : (sigma - 1) * Tu;
    return FvLowMachModel{.tr = Transport::air(c.prandtl, Tu, c.tinfK, c.prandtl, kGammaAir, kRgasBox, c.sutherland),
                          .prGas = Tu,
                          .lnA = std::log(c.flameEig.Lambda),
                          .Ta = c.flameModel.theta() * Tu,
                          .qcp = qcp,
                          .nY = c.flamePremix2 ? 2 : 1,
                          .ox = c.flamePremix2 ? 1 : -1,
                          .stoich = c.flamePremix2 ? c.premix.s : 1.0,
                          .law = c.flamePremix2 ? c.premixLaw : PhiRateLaw<double>{},
                          // --gravity: g along -x on the box, -reference.direction on a polyMesh
                          // (the jet axis points up).
                          .gravity = {-c.lmGravity * spec.dir[0], -c.lmGravity * spec.dir[1],
                                      -c.lmGravity * spec.dir[2]}};
}

std::function<FvLmRow(const std::array<double, 3>& cf, std::int64_t ownerSrc)>
fvBurnerRow(const Case& c, const FvBurnerFrame& frame)
{
    const double Tu = c.flameTu;
    const bool two = c.flamePremix2;
    if (c.fvMode == FvCompMode::PolyMesh)
        // The face centre's row, lmInflowRow's (the structured rows at the cell rows).
        return [&c, frame, Tu, two](const std::array<double, 3>& cf, std::int64_t) {
            const auto [x, y] = frame.at(cf);
            (void)x;
            const LmInflowRow in = lmInflowRow(c, y, 0.5 * c.grid.hz);
            return FvLmRow{.m = in.m, .t = Tu * in.t, .y = {in.yF, two ? in.yO : 0.0}};
        };
    // The owner's row j + ny k, exactly the structured row (sp.rowM, rowT, rowY).
    return [&c, Tu, two](const std::array<double, 3>&, std::int64_t src) {
        const Grid& g = c.grid;
        const std::int64_t row = src / g.nx;
        const int j = int(row % g.ny), k = int(row / g.ny);
        const LmInflowRow in = lmInflowRow(c, (j + 0.5) * g.hy, (k + 0.5) * g.hz);
        return FvLmRow{.m = in.m, .t = Tu * in.t, .y = {in.yF, two ? in.yO : 0.0}};
    };
}

} // namespace dns
