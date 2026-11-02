#include "dns/fvlowmach.hpp"

#include "dns/curvmesh.hpp"

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
            if (!p.hasY) return bad(std::format("patch.{}: an inflow carries y = [...] on the low-Mach route", p.name));
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
        case FvPatchKind::Empty: break;
        }
        b.flow.patches.push_back(std::move(o));
    }
    if (!open) return bad("no outflow patch: a closed box needs a time-dependent P0, not built");
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

std::expected<FaceMesh, std::string> fvLowMachBox(const Grid& g)
{
    const FoamPatchType zt = g.nz == 1 ? FoamPatchType::Empty : FoamPatchType::Cyclic;
    const StructuredSpec ss{.lo = {0, 0, 0},
                            .hi = {g.nx - 1, g.ny - 1, g.nz - 1},
                            .name = {"xlo", "xhi", "ylo", "yhi", "back", "front"},
                            .type = {FoamPatchType::Patch, FoamPatchType::Patch, FoamPatchType::Cyclic,
                                     FoamPatchType::Cyclic, zt, zt}};
    return FaceMesh::fromCurvMesh(CurvMesh::uniform(g), ss, CellOrder::Morton);
}

FvBcSpec fvLowMachBoxBc()
{
    FvBcSpec spec;
    spec.patches = {{.name = "xlo", .kind = FvPatchKind::Inflow, .hasY = true, .y = {1, 0, 0, 0}},
                    {.name = "xhi", .kind = FvPatchKind::Outflow}};
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

} // namespace dns
