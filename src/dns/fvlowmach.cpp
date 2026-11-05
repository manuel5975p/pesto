#include "dns/fvlowmach.hpp"

#include "dns/curvmesh.hpp"
#include "dns/polymesh.hpp"

#include <algorithm>
#include <cmath>
#include <format>

namespace dns {
namespace {

std::unexpected<std::string> bad(const std::string& m) { return std::unexpected("--bc: " + m); }

} // namespace

std::expected<FvLowMachBcs, std::string> fvLowMachBcs(const FvBcSpec& spec, double U, double Tu)
{
    FvLowMachBcs b;
    b.flow.uRef = U;
    bool open = false;
    for (const FvPatchSpec& p : spec.patches) {
        if (p.kind == FvPatchKind::Empty) continue;
        FvIncompPatch o{.name = p.name};
        switch (p.kind) {
        case FvPatchKind::Far:
            return bad(std::format("patch.{}: type = \"far\": a low-Mach face has no characteristic state "
                                   "(inflow, outflow)",
                                   p.name));
        case FvPatchKind::Wall:
            o.kind = FvIncompBc::Wall;
            o.hasT = p.isothermal;
            o.T = p.tWall * Tu;
            for (std::size_t k = 0; k < 3; ++k) {
                o.u[k] = p.uWall[k] * U;
                o.omega[k] = p.omega[k];
                o.origin[k] = p.origin[k];
            }
            break;
        case FvPatchKind::Slip: o.kind = FvIncompBc::Slip; break;
        case FvPatchKind::Inflow:
            // jet = true: the burner's rows set u, t and y per face (fvLowMachRows).
            if (!p.hasY && !p.jet)
                return bad(std::format("patch.{}: an inflow carries y = [...] on the low-Mach route", p.name));
            o.kind = FvIncompBc::Inflow;
            o.hasT = true;
            o.T = p.tIn * Tu;
            for (std::size_t k = 0; k < 3; ++k) o.u[k] = (p.hasU ? p.uIn[k] : spec.dir[k]) * U;
            b.y[p.name] = {p.y[0], p.y[1]};
            break;
        case FvPatchKind::Outflow:
            if (p.burnt)
                return bad(std::format("patch.{}: burnt: the low-Mach outflow is p = 0, no thermodynamic jump", p.name));
            if (p.hasY)
                return bad(std::format("patch.{}: y on an outflow: its scalars are zero gradient both ways", p.name));
            if (p.pOut != 1.0) return bad(std::format("patch.{}: p: the dynamic pressure is 0 on the outflow", p.name));
            if (p.tBack != 1.0)
                return bad(std::format("patch.{}: tback: its scalars are zero gradient both ways", p.name));
            o.kind = FvIncompBc::Outflow;
            open = true;
            break;
        case FvPatchKind::Open: {
            // p = 0 on the face as an outflow's; the sign-of-flux rule rides the
            // tables (bOpen). One ambient per run: gpu/lowmach's.
            const double t = p.tIn * Tu;
            const std::array<double, kFvLmMaxY> y{p.y[0], p.y[1]};
            if (b.open && (t != b.ambT || y != b.ambY))
                return bad(std::format("patch.{}: every open patch draws the one ambient: t, y must equal patch.{}'s",
                                       p.name, b.openPatches.front()));
            b.open = true;
            b.openPatches.push_back(p.name);
            b.ambT = t;
            b.ambY = y;
            o.kind = FvIncompBc::Outflow;
            open = true;
            break;
        }
        case FvPatchKind::Empty: break;
        }
        b.flow.patches.push_back(std::move(o));
    }
    if (!open) return bad("no outflow or open patch: a closed box needs a time-dependent P0, not built");
    return b;
}

FvLowMachTables buildFvLowMachTables(const FaceMesh& fm, const FvLowMachBcs& b, const FvIncompSetup& s, double prGas)
{
    const std::size_t nB = std::size_t(s.nB), nI = std::size_t(s.nI), nC = std::size_t(s.nC);
    FvLowMachTables t;
    for (auto& v : t.bY) v.assign(nB, 0.0);
    t.bYfix.assign(nB, 0);
    t.rUb.assign(3 * nB, 0.0);
    for (std::size_t q = 0; q < nB; ++q) {
        if (s.bCode[q] != std::uint8_t(FvIncompFaceBc::Inflow)) continue;
        const std::size_t fb = std::size_t(s.bFace[q]);
        const auto pit = std::ranges::find_if(fm.patches, [&](const MeshPatch& m) {
            return std::size_t(m.first) <= fb && fb < std::size_t(m.first + m.count);
        });
        const std::array<double, kFvLmMaxY>& y = b.y.at(pit->name);
        t.bY[0][q] = y[0];
        t.bY[1][q] = y[1];
        t.bYfix[q] = 1;
        const double rhoB = prGas / s.bT[q];
        for (std::size_t k = 0; k < 3; ++k) t.rUb[k * nB + q] = rhoB * s.bU[k * nB + q];
    }
    if (b.open) {
        t.ambT = b.ambT;
        t.ambY = b.ambY;
        t.bOpen.assign(nB, 0);
        for (std::size_t q = 0; q < nB; ++q) {
            if (s.bCode[q] != std::uint8_t(FvIncompFaceBc::Outflow)) continue;
            const std::size_t fb = std::size_t(s.bFace[q]);
            const auto pit = std::ranges::find_if(fm.patches, [&](const MeshPatch& m) {
                return std::size_t(m.first) <= fb && fb < std::size_t(m.first + m.count);
            });
            t.bOpen[q] = std::ranges::find(b.openPatches, pit->name) != b.openPatches.end() ? 2 : 1;
        }
    }
    t.first.assign(nI, 0);
    for (std::size_t c = 0; c < nC; ++c)
        for (int e = s.cellStart[c]; e < s.cellStart[c + 1]; ++e) {
            const int bc = s.cellEnt[std::size_t(e)];
            if (bc >= 0) continue;
            const std::size_t q = std::size_t(-bc - 1);
            for (int o = s.cellStart[c]; o < s.cellStart[c + 1]; ++o) {
                const int code = s.cellEnt[std::size_t(o)];
                if (code < 0) continue;
                const std::size_t f = std::size_t(code >> 1);
                const bool isNei = code & 1;
                // dn: toward the other cell, along the outward boundary normal.
                double dn = 0, dd = 0;
                for (std::size_t k = 0; k < 3; ++k) {
                    const double dk = s.d[k * nI + f];
                    dn += dk * s.bN[k * nB + q];
                    dd += dk * dk;
                }
                if (isNei) dn = -dn;
                if (-dn >= 0.5 * std::sqrt(dd)) t.first[f] |= isNei ? 2 : 1;
            }
        }
    return t;
}

std::expected<int, std::string> fvLowMachRows(const FaceMesh& fm, const std::string& patch, FvIncompSetup& s,
                                              FvLowMachTables& t, double prGas,
                                              const std::function<FvLmRow(std::size_t q)>& row)
{
    const auto pit = std::ranges::find_if(fm.patches, [&](const MeshPatch& m) { return m.name == patch; });
    if (pit == fm.patches.end()) return bad(std::format("the burner's inflow patch `{}` is not in the mesh", patch));
    const std::size_t nB = std::size_t(s.nB);
    int n = 0;
    for (std::size_t q = 0; q < nB; ++q) {
        const std::size_t fb = std::size_t(s.bFace[q]);
        if (fb < std::size_t(pit->first) || fb >= std::size_t(pit->first + pit->count)) continue;
        if (s.bCode[q] != std::uint8_t(FvIncompFaceBc::Inflow))
            return bad(std::format("patch.{}: the burner's rows need type = \"inflow\"", patch));
        double e[3], mag = 0;
        for (std::size_t k = 0; k < 3; ++k) {
            e[k] = s.bU[k * nB + q];
            mag += e[k] * e[k];
        }
        mag = std::sqrt(mag);
        if (!(mag > 0)) return bad(std::format("patch.{}: a zero inflow velocity has no direction for the rows", patch));
        const FvLmRow r = row(q);
        for (std::size_t k = 0; k < 3; ++k) {
            e[k] /= mag;
            s.bU[k * nB + q] = r.m * r.t / prGas * e[k];
            t.rUb[k * nB + q] = r.m * e[k];
        }
        s.bT[q] = r.t;
        for (std::size_t k = 0; k < std::size_t(kFvLmMaxY); ++k) t.bY[k][q] = r.y[k];
        ++n;
    }
    if (n == 0) return bad(std::format("patch.{}: no faces", patch));
    return n;
}

std::expected<FaceMesh, std::string> fvLowMachBox(const Grid& g, bool openY, std::span<const std::uint8_t> solid)
{
    const FoamPatchType zt = g.nz == 1 ? FoamPatchType::Empty : FoamPatchType::Cyclic;
    const FoamPatchType yt = openY ? FoamPatchType::Patch : FoamPatchType::Cyclic;
    const StructuredSpec ss{.lo = {0, 0, 0},
                            .hi = {g.nx - 1, g.ny - 1, g.nz - 1},
                            .name = {"xlo", "xhi", "ylo", "yhi", "back", "front"},
                            .type = {FoamPatchType::Patch, FoamPatchType::Patch, yt, yt, zt, zt}};
    if (solid.empty()) return FaceMesh::fromCurvMesh(CurvMesh::uniform(g), ss, CellOrder::Morton);
    if (solid.size() != g.size()) return std::unexpected("fv low-Mach box: the solid map does not match the grid");
    auto pm = FaceMesh::polyOf(CurvMesh::uniform(g), ss);
    if (!pm) return std::unexpected(pm.error());
    // polyOf's cells are Grid::idx. Keep the fluid cells (ascending: owner < neighbour
    // survives); a fluid/solid face becomes a `rim` face owned by its fluid cell (node
    // order reversed when the fluid cell was the neighbour: S points out of the fluid);
    // faces of solid cells only go. Patches keep their order, `rim` last.
    const std::size_t nCell = g.size();
    std::vector<std::int32_t> fluid(nCell, -1), cellOf;
    for (std::size_t c = 0; c < nCell; ++c)
        if (!solid[c]) {
            fluid[c] = std::int32_t(cellOf.size());
            cellOf.push_back(std::int32_t(c));
        }
    if (cellOf.empty()) return std::unexpected("fv low-Mach box: every cell is solid");
    const PolyMesh& src = *pm;
    PolyMesh out;
    out.px = src.px;
    out.py = src.py;
    out.pz = src.pz;
    out.faceStart.push_back(0);
    std::vector<std::int32_t> rimNodes, rimStart{0}, rimOwner;
    auto nodesOf = [&](std::int64_t f) {
        const std::size_t a = std::size_t(src.faceStart[std::size_t(f)]), e = std::size_t(src.faceStart[std::size_t(f) + 1]);
        return std::span<const std::int32_t>(src.faceNodes.data() + a, e - a);
    };
    auto put = [&](std::span<const std::int32_t> nd, std::int32_t own) {
        out.faceNodes.insert(out.faceNodes.end(), nd.begin(), nd.end());
        out.faceStart.push_back(std::int32_t(out.faceNodes.size()));
        out.owner.push_back(own);
    };
    const std::int64_t nInt = src.man.nInternalFaces;
    for (std::int64_t f = 0; f < nInt; ++f) {
        const std::int32_t o = fluid[std::size_t(src.owner[std::size_t(f)])],
                           n = fluid[std::size_t(src.neighbour[std::size_t(f)])];
        if (o >= 0 && n >= 0) {
            put(nodesOf(f), o);
            out.neighbour.push_back(n);
        } else if (o >= 0 || n >= 0) {
            const auto nd = nodesOf(f);
            if (o >= 0) {
                rimNodes.insert(rimNodes.end(), nd.begin(), nd.end());
            } else {
                rimNodes.push_back(nd[0]);
                rimNodes.insert(rimNodes.end(), nd.rbegin(), nd.rend() - 1);
            }
            rimStart.push_back(std::int32_t(rimNodes.size()));
            rimOwner.push_back(o >= 0 ? o : n);
        }
    }
    out.man = src.man;
    out.man.nInternalFaces = std::int64_t(out.neighbour.size());
    out.man.patches.clear();
    for (const FoamPatch& fp : src.man.patches) {
        if (fp.type == FoamPatchType::Cyclic)
            for (std::int64_t f = fp.startFace; f < fp.startFace + fp.nFaces; ++f)
                if (solid[std::size_t(src.owner[std::size_t(f)])])
                    return std::unexpected("fv low-Mach box: a solid cell on a cyclic face (the tube needs open "
                                           "laterals)");
        FoamPatch np = fp;
        np.startFace = std::int64_t(out.owner.size());
        for (std::int64_t f = fp.startFace; f < fp.startFace + fp.nFaces; ++f) {
            const std::int32_t o = fluid[std::size_t(src.owner[std::size_t(f)])];
            if (o >= 0) put(nodesOf(f), o);
        }
        np.nFaces = std::int64_t(out.owner.size()) - np.startFace;
        out.man.patches.push_back(std::move(np));
    }
    FoamPatch rim{.name = "rim", .type = FoamPatchType::Wall, .startFace = std::int64_t(out.owner.size())};
    for (std::size_t q = 0; q + 1 < rimStart.size(); ++q)
        put(std::span<const std::int32_t>(rimNodes.data() + rimStart[q], std::size_t(rimStart[q + 1] - rimStart[q])),
            rimOwner[q]);
    rim.nFaces = std::int64_t(out.owner.size()) - rim.startFace;
    out.man.patches.push_back(std::move(rim));
    out.man.nCells = std::int64_t(cellOf.size());
    out.man.nFaces = std::int64_t(out.owner.size());
    auto fm = FaceMesh::fromPolyMesh(out, CellOrder::Morton);
    if (!fm) return std::unexpected(fm.error());
    for (auto& c : fm->srcCell) c = cellOf[std::size_t(c)];
    return fm;
}

FvBcSpec fvLowMachBoxBc(bool openY, double ambT, std::array<double, kFvLmMaxY> ambY)
{
    FvBcSpec spec;
    spec.patches = {{.name = "xlo", .kind = FvPatchKind::Inflow, .hasY = true, .y = {1, 0, 0, 0}},
                    {.name = "xhi", .kind = FvPatchKind::Outflow}};
    if (openY)
        for (const char* n : {"ylo", "yhi"})
            spec.patches.push_back(
                {.name = n, .kind = FvPatchKind::Open, .tIn = ambT, .hasY = true, .y = {ambY[0], ambY[1], 0, 0}});
    return spec;
}

FvLowMachIC fvLowMachFlameIC(const FaceMesh& fm, const std::array<double, 3>& dir, const std::array<double, 3>& origin,
                             double sigma, double m, double prGas, const std::function<double(double)>& t)
{
    const std::size_t nC = std::size_t(fm.nC);
    FvLowMachIC ic{.rho = std::vector<double>(nC),
                   .ry0 = std::vector<double>(nC),
                   .ry1 = {},
                   .ru = std::vector<double>(3 * nC),
                   .p = std::vector<double>(nC)};
    for (std::size_t c = 0; c < nC; ++c) {
        double s = 0;
        for (std::size_t k = 0; k < 3; ++k) s += (fm.C[k][c] - origin[k]) * dir[k];
        const double T = t(s);
        ic.rho[c] = prGas / T;
        ic.ry0[c] = ic.rho[c] * (sigma - T) / (sigma - 1);
        ic.p[c] = m * m * (sigma - T);
        for (std::size_t k = 0; k < 3; ++k) ic.ru[k * nC + c] = m * dir[k];
    }
    return ic;
}

std::expected<void, std::string> fvLowMachTfles(const FaceMesh& fm, FvLowMachTables& t)
{
    // The walk needs the active flags only: AlongD, alpha 1 (the FV path's pinned scheme).
    auto geom = fvFaceGeom(fm, FvFaceGradRule::AlongD, 1.0);
    if (!geom) return std::unexpected(geom.error());
    auto w = fvTflesWalk(fm, *geom);
    if (!w) return std::unexpected(w.error());
    t.walk = std::move(*w);
    return {};
}

} // namespace dns
