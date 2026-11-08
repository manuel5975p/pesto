#include "dns/fvcomp_case.hpp"

#include "dns/comp.hpp"
#include "dns/fvwalldist.hpp"

#include <algorithm>
#include <cassert>
#include <cmath>

namespace dns {

FvFreestream fvFreestreamOf(double mach, double gamma, double Rgas, std::array<double, 3> dir)
{
    const double p = 1.0 / (gamma * mach * mach);
    return {.rho = 1.0, .U = 1.0, .p = p, .T = p / Rgas, .gamma = gamma, .dir = dir};
}

FvBc fvBcOf(const FvPatchSpec& p, const FvFreestream& f, double frac)
{
    FvBc b;
    const double u = frac * f.U;
    std::ranges::copy(p.y, b.y);
    switch (p.kind) {
    case FvPatchKind::Wall:
        b.kind = FvBcKind::Wall;
        b.isothermal = p.isothermal;
        b.tWall = p.tWall * f.T;
        for (std::size_t k = 0; k < 3; ++k) {
            b.uWall[k] = p.uWall[k] * f.U;
            b.omega[k] = p.omega[k];
            b.origin[k] = p.origin[k];
        }
        return b;
    case FvPatchKind::Slip: b.kind = FvBcKind::Slip; return b;
    case FvPatchKind::Inflow:
        b.kind = FvBcKind::Inflow;
        b.tIn = p.tIn * f.T;
        for (std::size_t k = 0; k < 3; ++k) b.uIn[k] = p.hasU ? frac * p.uIn[k] * f.U : u * f.dir[k];
        return b;
    case FvPatchKind::Outflow:
        b.kind = FvBcKind::Outflow;
        b.pOut = p.burnt ? f.pBurnt : p.pOut * f.p;
        b.tBack = p.burnt ? f.tBurnt : p.tBack * f.T;
        b.exit = p.exit;
        return b;
    case FvPatchKind::Total:
        b.kind = FvBcKind::Total;
        b.pTotal = frac >= 1.0 ? p.p0 * f.p : f.pRest + frac * (p.p0 * f.p - f.pRest);
        b.tTotal = p.t0 * f.T;
        return b;
    case FvPatchKind::Wedge:
        b.kind = FvBcKind::Slip;
        b.wedge = true;
        return b;
    case FvPatchKind::Far: {
        b.kind = FvBcKind::Far;
        b.qFar[0] = f.rho;
        double ke = 0.0;
        for (std::size_t k = 0; k < 3; ++k) {
            const double m = f.rho * (u * f.dir[k]);
            b.qFar[1 + k] = m;
            ke += m * (u * f.dir[k]);
        }
        b.qFar[4] = f.p / (f.gamma - 1.0) + 0.5 * ke;
        return b;
    }
    case FvPatchKind::Empty: b.kind = FvBcKind::Empty; return b;
    case FvPatchKind::Open:
        // The low-Mach route's kind: validate() rejects it on this route by name.
        assert(false && "type = open reached the compressible table");
        b.kind = FvBcKind::Empty;
        return b;
    }
    return b;
}

std::vector<FvBc> fvBcsOf(const FaceMesh& fm, const FvBcSpec& spec, const FvFreestream& f, double frac)
{
    std::vector<FvBc> v;
    v.reserve(fm.patches.size());
    for (const MeshPatch& m : fm.patches) {
        if (m.type == FoamPatchType::Empty) {
            FvBc e;
            e.kind = FvBcKind::Empty;
            v.push_back(e);
            continue;
        }
        if (m.type == FoamPatchType::Wedge) { // a Slip row, bound or left out of the file
            FvBc w;
            w.kind = FvBcKind::Slip;
            w.wedge = true;
            v.push_back(w);
            continue;
        }
        v.push_back(fvBcOf(*std::ranges::find(spec.patches, m.name, &FvPatchSpec::name), f, frac));
    }
    return v;
}

std::vector<std::uint8_t> fvTallyOf(const FaceMesh& fm, const FvBcSpec& spec)
{
    std::vector<std::uint8_t> v(fm.patches.size(), 0);
    for (std::size_t i = 0; i < fm.patches.size(); ++i) {
        if (fm.patches[i].type == FoamPatchType::Empty || fm.patches[i].type == FoamPatchType::Wedge) continue;
        v[i] = std::ranges::find(spec.patches, fm.patches[i].name, &FvPatchSpec::name)->tallied ? 1 : 0;
    }
    return v;
}

std::expected<std::vector<double>, std::string> fvRansWallDistance(PolyMesh pm, const FaceMesh& fm,
                                                                   const FvBcSpec& spec)
{
    for (FoamPatch& p : pm.man.patches) {
        if (p.type == FoamPatchType::Empty || p.type == FoamPatchType::Cyclic) continue;
        const auto it = std::ranges::find(spec.patches, p.name, &FvPatchSpec::name);
        p.type = it != spec.patches.end() && it->kind == FvPatchKind::Wall ? FoamPatchType::Wall : FoamPatchType::Patch;
    }
    // fvWallDistance reads srcCell as a polyMesh cell; fromCurvMesh's holds Grid::idx,
    // ascending in the same order, so its rank is the polyMesh cell (identity else).
    FaceMesh ranked = fm;
    std::vector<std::int64_t> sorted = fm.srcCell;
    std::ranges::sort(sorted);
    for (auto& v : ranked.srcCell) v = std::ranges::lower_bound(sorted, v) - sorted.begin();
    auto d = fvWallDistance(pm, ranked, true, false);
    if (!d) return std::unexpected(std::move(d.error()));
    return std::move(d->y);
}

FvState fvWithScalars(const FvState& s, RansModel m, const std::array<double, 2>& phiInf)
{
    const int nT = ransScalars(m);
    FvState o = FvState::zeros(s.nC, kFvMeanVars + nT);
    std::ranges::copy(s.q, o.q.begin());
    for (int e = 0; e < nT; ++e)
        for (std::size_t c = 0; c < std::size_t(s.nC); ++c)
            o.var(kFvMeanVars + e)[c] = s.var(0)[c] * phiInf[std::size_t(e)];
    return o;
}

double fvMeshDepth(const FaceMesh& fm)
{
    if (fm.emptyAxis < 0) return 1.0;
    const std::size_t a = std::size_t(fm.emptyAxis);
    for (const MeshPatch& p : fm.patches)
        if (p.type == FoamPatchType::Empty && p.count > 0) {
            const std::size_t f = std::size_t(fm.nI + p.first);
            return fm.V[std::size_t(fm.own[f])] / std::abs(fm.S[a][f]);
        }
    return 1.0;
}

FvState fvStateOf(const FaceMesh& fm, const CompState& q, std::span<const std::vector<double>> rhoY)
{
    FvState s = FvState::zeros(fm.nC, kFvMeanVars + int(rhoY.size()));
    const std::vector<double>* mean[kFvMeanVars] = {&q.rho, &q.mx, &q.my, &q.mz, &q.E};
    for (int r = 0; r < kFvMeanVars + int(rhoY.size()); ++r) {
        const std::vector<double>& v = r < kFvMeanVars ? *mean[r] : rhoY[std::size_t(r - kFvMeanVars)];
        for (std::size_t n = 0; n < std::size_t(fm.nC); ++n) s.var(r)[n] = v[std::size_t(fm.srcCell[n])];
    }
    return s;
}

StructuredSpec fvCompOgridSpec(const Grid& g)
{
    const FoamPatchType zt = g.nz == 1 ? FoamPatchType::Empty : FoamPatchType::Cyclic;
    return StructuredSpec{.lo = {0, 1, 0},
                          .hi = {g.nx - 1, g.ny - 2, g.nz - 1},
                          .name = {"ilo", "ihi", "wall", "far", "back", "front"},
                          .type = {FoamPatchType::Cyclic, FoamPatchType::Cyclic, FoamPatchType::Wall,
                                   FoamPatchType::Patch, zt, zt}};
}

StructuredSpec fvCompBoxSpec(const Grid& g, bool periodic, bool wallY)
{
    const FoamPatchType xt = periodic ? FoamPatchType::Cyclic : FoamPatchType::Patch;
    const FoamPatchType yt = wallY ? FoamPatchType::Wall : xt;
    const FoamPatchType zt = g.nz == 1 ? FoamPatchType::Empty : FoamPatchType::Cyclic;
    return StructuredSpec{.lo = {0, 0, 0},
                          .hi = {g.nx - 1, g.ny - 1, g.nz - 1},
                          .name = {"xlo", "xhi", "ylo", "yhi", "back", "front"},
                          .type = {xt, xt, yt, yt, zt, zt}};
}

FvBcSpec fvCompBoxBc(const FaceMesh& fm)
{
    FvBcSpec bc;
    for (const MeshPatch& p : fm.patches) {
        if (p.type == FoamPatchType::Empty) continue;
        if (p.type == FoamPatchType::Wall)
            bc.patches.push_back({.name = p.name, .kind = FvPatchKind::Wall, .isothermal = false, .tallied = true});
        else
            bc.patches.push_back({.name = p.name, .kind = FvPatchKind::Far});
    }
    return bc;
}

FvState fvUniformState(const FaceMesh& fm, const FvFreestream& fs, double u0)
{
    FvState ic = FvState::zeros(fm.nC);
    for (std::size_t q = 0; q < std::size_t(fm.nC); ++q) {
        ic.var(0)[q] = fs.rho;
        for (int k = 0; k < 3; ++k) ic.var(1 + k)[q] = fs.rho * u0 * fs.dir[std::size_t(k)];
        ic.var(4)[q] = fs.p / (fs.gamma - 1.0) + 0.5 * fs.rho * u0 * u0;
    }
    return ic;
}

} // namespace dns
