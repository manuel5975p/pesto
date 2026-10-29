#include "dns/fvsimple.hpp"

#include "dns/fvwalldist.hpp"

#include <algorithm>
#include <cmath>
#include <format>
#include <map>

namespace dns {
namespace {

using V3 = std::array<double, 3>;

double dot(const V3& a, const V3& b) { return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]; }
double mag(const V3& a) { return std::sqrt(dot(a, a)); }
V3 cross(const V3& a, const V3& b)
{
    return {a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]};
}

// Component q (0..nc-1) of value spec v at OpenFOAM index i.
double valueAt(const FvValue& v, std::int64_t i, int q, int nc)
{
    if (v.uniform) return v.v.empty() ? 0.0 : v.v[std::size_t(q)];
    return v.v[std::size_t(i * nc + q)];
}

} // namespace

std::expected<FoamCase, std::string> loadFoamCaseManifest(const std::filesystem::path& dir)
{
    auto man = readPolyManifest(dir / "constant" / "polyMesh");
    if (!man) return std::unexpected(man.error());
    auto t = readFoamCaseTexts(dir);
    if (!t) return std::unexpected(t.error());
    auto c = parseFoamCase(*t, *man);
    if (!c) return std::unexpected(c.error());
    c->dir = dir.string();
    return c;
}

std::expected<FvLoaded, std::string> loadFvCase(const std::filesystem::path& dir)
{
    FvLoaded L;
    auto pm = readPolyMesh(dir / "constant" / "polyMesh");
    if (!pm) return std::unexpected(pm.error());
    L.pm = std::move(*pm);
    auto t = readFoamCaseTexts(dir);
    if (!t) return std::unexpected(t.error());
    auto c = parseFoamCase(*t, L.pm.man);
    if (!c) return std::unexpected(c.error());
    L.c = std::move(*c);
    L.c.dir = dir.string();
    auto fm = FaceMesh::fromPolyMesh(L.pm, CellOrder::Source);
    if (!fm) return std::unexpected((dir / "constant" / "polyMesh").string() + ": " + fm.error());
    L.fm = std::move(*fm);
    return L;
}

std::expected<FvSimpleSetup, std::string> buildSimpleSetup(const FaceMesh& fm, const FoamCase& c,
                                                           const la::CellGraph& g, const la::EllPattern& pat)
{
    FvSimpleSetup s;
    s.nC = fm.nC;
    s.nI = fm.nI;
    s.emptyAxis = fm.emptyAxis;
    s.srcCell = fm.srcCell;
    const std::size_t nC = std::size_t(fm.nC), nI = std::size_t(fm.nI);

    // Manifest patch index of every FaceMesh patch (by name).
    std::map<std::string, int> casePatch;
    for (std::size_t i = 0; i < c.patches.size(); ++i) casePatch[c.patches[i].name] = int(i);
    std::vector<int> facePatch(std::size_t(fm.nB), -1); // FaceMesh boundary face -> case patch
    std::vector<int> faceMeshPatch(std::size_t(fm.nB), -1);
    for (std::size_t mp = 0; mp < fm.patches.size(); ++mp) {
        const MeshPatch& p = fm.patches[mp];
        const auto it = casePatch.find(p.name);
        if (it == casePatch.end()) return std::unexpected("FaceMesh patch '" + p.name + "' has no case entry");
        for (int i = 0; i < p.count; ++i) {
            facePatch[std::size_t(p.first + i)] = it->second;
            faceMeshPatch[std::size_t(p.first + i)] = int(mp);
        }
    }

    // --- internal faces ----------------------------------------------------
    s.own.assign(fm.own.begin(), fm.own.begin() + fm.nI);
    s.nei = fm.nei;
    s.S.resize(3 * nI);
    s.Cf.resize(3 * nI);
    s.rO.resize(3 * nI);
    s.rN.resize(3 * nI);
    s.k.resize(3 * nI);
    s.magS.resize(nI);
    s.w.resize(nI);
    s.dc.resize(nI);
    s.internalSrc.resize(nI);
    std::vector<std::uint8_t> cyclicFace(nI, 0); // merged cyclic pair (MRF rejects it)
    for (std::size_t f = 0; f < nI; ++f) {
        const std::size_t o = std::size_t(fm.own[f]), n = std::size_t(fm.nei[f]);
        const V3 S{fm.S[0][f], fm.S[1][f], fm.S[2][f]};
        V3 Cf{fm.Cf[0][f], fm.Cf[1][f], fm.Cf[2][f]};
        const V3 Co{fm.C[0][o], fm.C[1][o], fm.C[2][o]};
        const V3 Cn{fm.C[0][n], fm.C[1][n], fm.C[2][n]};
        const V3 delta{fm.d[0][f], fm.d[1][f], fm.d[2][f]}; // C_n (+ cyclic offset) - C_o
        V3 rO{Cf[0] - Co[0], Cf[1] - Co[1], Cf[2] - Co[2]};
        V3 rN{Cf[0] - Cn[0], Cf[1] - Cn[1], Cf[2] - Cn[2]};
        // A merged cyclic face: its Cf may sit in either cell's frame
        // (FaceMesh contract). rO in the owner's frame (the closer image, as
        // fvgrad's rOwn), rN = rO - d in the neighbour's.
        const V3 T{Co[0] + delta[0] - Cn[0], Co[1] + delta[1] - Cn[1], Co[2] + delta[2] - Cn[2]};
        if (dot(T, T) > 1e-16 * dot(delta, delta)) {
            cyclicFace[f] = 1;
            const V3 rt{rO[0] + T[0], rO[1] + T[1], rO[2] + T[2]};
            if (dot(rt, rt) < dot(rO, rO)) rO = rt;
            rN = {rO[0] - delta[0], rO[1] - delta[1], rO[2] - delta[2]};
            Cf = {Co[0] + rO[0], Co[1] + rO[1], Co[2] + rO[2]};
        }
        const double magS = mag(S);
        const double SfdOwn = std::abs(dot(S, rO));
        const double SfdNei = std::abs(dot(S, V3{-rN[0], -rN[1], -rN[2]}));
        const double w = std::abs(SfdOwn + SfdNei) > 1e-300 ? SfdNei / (SfdOwn + SfdNei) : 0.5;
        const V3 nh{S[0] / magS, S[1] / magS, S[2] / magS};
        const double dcNO = 1.0 / std::max(dot(nh, delta), 0.05 * mag(delta));
        for (int q = 0; q < 3; ++q) {
            s.S[std::size_t(q) * nI + f] = S[std::size_t(q)];
            s.Cf[std::size_t(q) * nI + f] = Cf[std::size_t(q)];
            s.rO[std::size_t(q) * nI + f] = rO[std::size_t(q)];
            s.rN[std::size_t(q) * nI + f] = rN[std::size_t(q)];
            s.k[std::size_t(q) * nI + f] = nh[std::size_t(q)] - delta[std::size_t(q)] * dcNO;
        }
        s.magS[f] = magS;
        s.w[f] = w;
        s.dc[f] = dcNO;
        s.internalSrc[f] = int(fm.srcFace[f]);
    }

    // --- boundary faces (non-empty) ------------------------------------------
    std::vector<int> bIndex(std::size_t(fm.nB), -1);
    for (int fb = 0; fb < fm.nB; ++fb) {
        const int cp = facePatch[std::size_t(fb)];
        if (cp < 0) return std::unexpected("boundary face outside every FaceMesh patch");
        if (c.patches[std::size_t(cp)].meshType == FoamPatchType::Empty) continue;
        bIndex[std::size_t(fb)] = s.nB++;
    }
    const std::size_t nB = std::size_t(s.nB);
    s.bOwn.resize(nB);
    s.bFace.resize(nB);
    s.bPatch.resize(nB);
    s.bLocal.resize(nB);
    s.bS.resize(3 * nB);
    s.bN.resize(3 * nB);
    s.bCf.resize(3 * nB);
    s.bMagS.resize(nB);
    s.bDc.resize(nB);
    s.bU.resize(nB);
    s.bP.resize(nB);
    s.bAssign.resize(nB);
    s.bSym.resize(nB);
    s.bAdjust.resize(nB);
    s.bRefU.resize(3 * nB);
    s.bRefP.resize(nB);
    s.bU0.resize(3 * nB);
    s.bP0.resize(nB);
    s.bForce.assign(nB, 0);
    s.patchFaces.resize(c.patches.size());
    // initial internal fields (OpenFOAM cell order -> FaceMesh)
    s.U0.resize(3 * nC);
    s.p0.resize(nC);
    for (std::size_t cc = 0; cc < nC; ++cc) {
        const std::int64_t src = fm.srcCell[cc];
        for (int q = 0; q < 3; ++q) s.U0[std::size_t(q) * nC + cc] = valueAt(c.uInternal, src, q, 3);
        s.p0[cc] = valueAt(c.pInternal, src, 0, 1);
    }
    // patch start (OpenFOAM face index) by case patch: srcFace - start = local
    std::vector<std::int64_t> start(c.patches.size(), 0);
    {
        // The manifest start faces are not in FoamCase; recover them as the
        // minimum srcFace per patch (patch faces are contiguous in OpenFOAM).
        std::vector<std::int64_t> mn(c.patches.size(), std::int64_t(1) << 62), cnt(c.patches.size(), 0);
        for (int fb = 0; fb < fm.nB; ++fb) {
            const int cp = facePatch[std::size_t(fb)];
            mn[std::size_t(cp)] = std::min(mn[std::size_t(cp)], fm.srcFace[std::size_t(fm.nI + fb)]);
            ++cnt[std::size_t(cp)];
        }
        for (std::size_t p = 0; p < c.patches.size(); ++p) {
            start[p] = cnt[p] ? mn[p] : 0;
            s.patchFaces[p].assign(std::size_t(cnt[p]), -1);
        }
    }
    for (int fb = 0; fb < fm.nB; ++fb) {
        const int cp = facePatch[std::size_t(fb)];
        const FvPatchBc& bc = c.patches[std::size_t(cp)];
        const std::size_t f = std::size_t(fm.nI + fb);
        const std::int64_t local = fm.srcFace[f] - start[std::size_t(cp)];
        const int b = bIndex[std::size_t(fb)];
        if (local < 0 || local >= std::int64_t(s.patchFaces[std::size_t(cp)].size()))
            return std::unexpected("patch '" + bc.name + "': face numbering is not contiguous");
        s.patchFaces[std::size_t(cp)][std::size_t(local)] = b;
        if (b < 0) continue;
        const std::size_t bb = std::size_t(b);
        const int o = fm.own[f];
        s.bOwn[bb] = o;
        s.bFace[bb] = int(f);
        s.bPatch[bb] = cp;
        s.bLocal[bb] = int(local);
        const V3 S{fm.S[0][f], fm.S[1][f], fm.S[2][f]};
        const double magS = mag(S);
        const V3 n{S[0] / magS, S[1] / magS, S[2] / magS};
        const V3 r{fm.d[0][f], fm.d[1][f], fm.d[2][f]}; // Cf - C_own
        const double nr = dot(n, r);
        const V3 delta{n[0] * nr, n[1] * nr, n[2] * nr};
        for (int q = 0; q < 3; ++q) {
            s.bS[std::size_t(q) * nB + bb] = S[std::size_t(q)];
            s.bN[std::size_t(q) * nB + bb] = n[std::size_t(q)];
            s.bCf[std::size_t(q) * nB + bb] = fm.Cf[std::size_t(q)][f];
        }
        s.bMagS[bb] = magS;
        s.bDc[bb] = 1.0 / mag(delta);
        FvUCode u{};
        bool assign = false;
        switch (bc.u) {
        case FvUBc::Fixed: u = FvUCode::Fixed; break;
        case FvUBc::ZeroGrad: u = FvUCode::ZeroGrad; assign = true; break;
        case FvUBc::InletOutlet: u = FvUCode::InletOutlet; assign = true; break;
        case FvUBc::Symmetry: u = FvUCode::Symmetry; assign = bc.uType != "slip"; break;
        case FvUBc::Freestream: u = FvUCode::FreestreamV; break;
        case FvUBc::Empty:
            return std::unexpected("patch '" + bc.name + "': velocity BC empty on a non-empty patch");
        }
        FvPCode pc{};
        switch (bc.p) {
        case FvPBc::Fixed: pc = FvPCode::Fixed; break;
        case FvPBc::ZeroGrad: pc = FvPCode::ZeroGrad; break;
        case FvPBc::Freestream: pc = FvPCode::FreestreamP; break;
        case FvPBc::Empty: return std::unexpected("patch '" + bc.name + "': pressure BC empty on a non-empty patch");
        }
        s.bU[bb] = std::uint8_t(u);
        s.bP[bb] = std::uint8_t(pc);
        s.bAssign[bb] = assign;
        s.bSym[bb] = bc.meshType == FoamPatchType::Symmetry || bc.meshType == FoamPatchType::SymmetryPlane;
        s.bAdjust[bb] = !(u == FvUCode::Fixed || u == FvUCode::FreestreamV);
        V3 Up{};
        for (int q = 0; q < 3; ++q) Up[std::size_t(q)] = s.U0[std::size_t(q) * nC + std::size_t(o)];
        for (int q = 0; q < 3; ++q) {
            const double ref = valueAt(bc.uValue, local, q, 3);
            s.bRefU[std::size_t(q) * nB + bb] = ref;
            double u0 = 0;
            switch (u) {
            case FvUCode::Fixed: u0 = ref; break;
            case FvUCode::ZeroGrad: u0 = Up[std::size_t(q)]; break;
            case FvUCode::InletOutlet: u0 = bc.uInit.v.empty() ? Up[std::size_t(q)] : valueAt(bc.uInit, local, q, 3); break;
            case FvUCode::FreestreamV: u0 = bc.uInit.v.empty() ? ref : valueAt(bc.uInit, local, q, 3); break;
            case FvUCode::Symmetry: {
                // 0.5 (pif + (I - 2 n n).pif), OpenFOAM's evaluate.
                double t = 0;
                for (int j = 0; j < 3; ++j)
                    t += ((q == j ? 1.0 : 0.0) - 2.0 * (n[std::size_t(q)] * n[std::size_t(j)])) * Up[std::size_t(j)];
                u0 = 0.5 * (Up[std::size_t(q)] + t);
                break;
            }
            }
            s.bU0[std::size_t(q) * nB + bb] = u0;
        }
        const double pref = valueAt(bc.pValue, local, 0, 1);
        s.bRefP[bb] = pref;
        const double pP = s.p0[std::size_t(o)];
        s.bP0[bb] = pc == FvPCode::Fixed      ? pref
                    : pc == FvPCode::ZeroGrad ? pP
                                              : (bc.pInit.v.empty() ? pref : valueAt(bc.pInit, local, 0, 1));
        if (c.forces && std::ranges::find(c.forces->patches, bc.name) != c.forces->patches.end()) s.bForce[bb] = 1;
    }

    // --- cells ---------------------------------------------------------------
    s.V = fm.V;
    s.C.resize(3 * nC);
    for (int q = 0; q < 3; ++q)
        for (std::size_t cc = 0; cc < nC; ++cc) s.C[std::size_t(q) * nC + cc] = fm.C[std::size_t(q)][cc];
    s.cellStart.assign(nC + 1, 0);
    for (std::size_t cc = 0; cc < nC; ++cc) {
        for (int k = fm.cellStart[cc]; k < fm.cellStart[cc + 1]; ++k) {
            const int f = fm.cellFace[std::size_t(k)];
            if (f < fm.nI) {
                const bool isOwn = fm.own[std::size_t(f)] == int(cc);
                s.cellEnt.push_back(2 * f + (isOwn ? 0 : 1));
                const int slot = isOwn ? g.ownSlot[std::size_t(f)] : g.neiSlot[std::size_t(f)];
                s.cellEll.push_back(pat.slotPos[std::size_t(slot)]);
            } else {
                const int b = bIndex[std::size_t(f - fm.nI)];
                if (b < 0) continue; // empty
                s.cellEnt.push_back(-(b + 1));
                s.cellEll.push_back(-1);
            }
        }
        s.cellStart[cc + 1] = int(s.cellEnt.size());
    }

    // --- MRF -------------------------------------------------------------------
    std::vector<const FvMrfZone*> active;
    for (const FvMrfZone& z : c.mrf)
        if (z.active) active.push_back(&z);
    if (active.size() > std::size_t(FvSimpleSetup::kMaxMrfZones))
        return std::unexpected(std::format("MRFProperties: {} active zones; the FV path supports at most {}",
                                           active.size(), FvSimpleSetup::kMaxMrfZones));
    if (!active.empty()) {
        s.mrf = true;
        s.zone.assign(nC, 0);
        for (std::size_t k = 0; k < active.size(); ++k) {
            const FvMrfZone& z = *active[k];
            for (std::size_t j = 0; j < k; ++j)
                if (active[j]->cellZone == z.cellZone)
                    return std::unexpected("MRF zones '" + active[j]->name + "' and '" + z.name +
                                           "' share cellZone '" + z.cellZone +
                                           "' (overlapping MRF zones are not supported)");
            const auto zi = std::ranges::find(fm.zoneNames, z.cellZone);
            if (zi == fm.zoneNames.end())
                return std::unexpected("MRF zone '" + z.name + "': cellZone '" + z.cellZone + "' not in the FaceMesh");
            // cellZones are disjoint (readPolyMesh rejects overlap).
            for (int cc : fm.zoneCells[std::size_t(zi - fm.zoneNames.begin())])
                s.zone[std::size_t(cc)] = std::uint8_t(k + 1);
            V3 om{}, org{};
            for (int q = 0; q < 3; ++q) {
                om[std::size_t(q)] = z.omega * z.axis[std::size_t(q)];
                org[std::size_t(q)] = z.origin[std::size_t(q)];
            }
            s.omega.push_back(om);
            s.origin.push_back(org);
        }
        // (Omega_z x (x - o_z)) . S, zone z = code - 1.
        auto frameOf = [&](int code, const V3& x, const V3& S) {
            const V3& om = s.omega[std::size_t(code - 1)];
            const V3& o = s.origin[std::size_t(code - 1)];
            const V3 r{x[0] - o[0], x[1] - o[1], x[2] - o[2]};
            return dot(cross(om, r), S);
        };
        s.frame.assign(nI, 0.0);
        for (std::size_t f = 0; f < nI; ++f) {
            const int a = s.zone[std::size_t(s.own[f])], b = s.zone[std::size_t(s.nei[f])];
            if (!a && !b) continue;
            if (cyclicFace[f])
                return std::unexpected("MRF zone '" + active[std::size_t(std::max(a, b) - 1)]->name +
                                       "': a zone cell touches a cyclic face (OpenFOAM subtracts each cyclic "
                                       "half's own frame flux: not reproduced)");
            const V3 Cf{s.Cf[f], s.Cf[nI + f], s.Cf[2 * nI + f]};
            const V3 S{s.S[f], s.S[nI + f], s.S[2 * nI + f]};
            const int lo = (a && b) ? std::min(a, b) : std::max(a, b);
            s.frame[f] = frameOf(lo, Cf, S);
            if (a && b && a != b) {
                if (!s.mrfTwo) {
                    s.mrfTwo = true;
                    s.frame2.assign(nI, 0.0);
                }
                s.frame2[f] = frameOf(std::max(a, b), Cf, S);
            }
        }
        s.bMrf.assign(nB, 0);
        s.bFrame.assign(nB, 0.0);
        s.bUmrf.assign(3 * nB, 0.0);
        for (std::size_t b = 0; b < nB; ++b) {
            const int zc = s.zone[std::size_t(s.bOwn[b])];
            if (!zc) continue;
            const FvMrfZone& z = *active[std::size_t(zc - 1)];
            const std::string& pn = c.patches[std::size_t(s.bPatch[b])].name;
            const bool excluded = std::ranges::find(z.nonRotatingPatches, pn) != z.nonRotatingPatches.end();
            s.bMrf[b] = excluded ? 2 : 1;
            const V3& om = s.omega[std::size_t(zc - 1)];
            const V3& o = s.origin[std::size_t(zc - 1)];
            const V3 r{s.bCf[b] - o[0], s.bCf[nB + b] - o[1], s.bCf[2 * nB + b] - o[2]};
            const V3 S{s.bS[b], s.bS[nB + b], s.bS[2 * nB + b]};
            const V3 u = cross(om, r);
            s.bFrame[b] = dot(u, S);
            for (int q = 0; q < 3; ++q) s.bUmrf[std::size_t(q) * nB + b] = u[std::size_t(q)];
        }
    }

    s.ras.model = c.ras.model;

    // --- pressure reference --------------------------------------------------
    s.needRef = std::ranges::none_of(c.patches, [](const FvPatchBc& b) {
        return b.p == FvPBc::Fixed || b.p == FvPBc::Freestream;
    });
    if (s.needRef) {
        if (!c.simple.pRefGiven)
            return std::unexpected("system/fvSolution: p needs a reference level (no fixedValue p patch): set "
                                   "SIMPLE/pRefCell and pRefValue");
        for (std::size_t cc = 0; cc < nC; ++cc)
            if (fm.srcCell[cc] == c.simple.pRefCell) s.refCell = int(cc);
        if (s.refCell < 0) return std::unexpected("pRefCell not found in the FaceMesh");
        s.refValue = c.simple.pRefValue;
    }
    return s;
}

std::expected<void, std::string> buildRasSetup(const PolyMesh& pm, const FaceMesh& fm, const FoamCase& c,
                                               FvSimpleSetup& s)
{
    const FvRas& r = c.ras;
    FvRasSetup& R = s.ras;
    R.model = r.model;
    const std::size_t nC = std::size_t(s.nC), nB = std::size_t(s.nB);
    auto wd = fvWallDistance(pm, fm, r.exactDist, r.correctWalls);
    if (!wd) return std::unexpected(wd.error());
    R.y = std::move(wd->y);
    R.yWall.assign(nB, 0.0);
    R.bWall.assign(nB, 0);
    for (std::size_t b = 0; b < nB; ++b) {
        const FvPatchBc& pb = c.patches[std::size_t(s.bPatch[b])];
        R.bWall[b] = pb.meshType == FoamPatchType::Wall;
        R.yWall[b] = wd->yWall[std::size_t(s.bFace[b] - fm.nI)];
    }
    // per-patch wall-function constants
    R.wf.assign(c.patches.size(), FvWallFnPatch{});
    const bool sst = r.model == FvTurbModel::KOmegaSST;
    for (std::size_t p = 0; p < c.patches.size(); ++p) {
        FvWallFnPatch& w = R.wf[p];
        const FvWallCoeffs* wc = nullptr;
        if (sst && r.omega[p].kind == FvSBc::OmegaWall) {
            wc = &r.omega[p].wc;
            w.blend = int(r.omega[p].blend);
            w.n = r.omega[p].n;
            w.beta1 = r.omega[p].beta1;
        }
        const FvNutPatch& np = r.nut[p];
        if (np.kind == FvNutBc::Spalding || np.kind == FvNutBc::KWall || np.kind == FvNutBc::LowRe) {
            if (wc && (wc->Cmu != np.wc.Cmu || wc->kappa != np.wc.kappa || wc->E != np.wc.E))
                return std::unexpected("patch '" + c.patches[p].name +
                                       "': the omega and nut wall functions must share Cmu, kappa, E");
            wc = &np.wc;
            if (np.kind == FvNutBc::KWall) {
                w.nutBlend = int(np.blend);
                w.nutN = np.n;
            }
            w.maxIter = np.maxIter;
        }
        if (wc) {
            w.Cmu = wc->Cmu;
            w.kappa = wc->kappa;
            w.E = wc->E;
            w.yPlusLam = wc->yPlusLam();
        }
    }
    // transported scalars
    const std::vector<FvScalarBc>* bcs[2] = {nullptr, nullptr};
    const FvValue* ints[2] = {nullptr, nullptr};
    if (sst) {
        R.nS = 2;
        bcs[0] = &r.k;
        bcs[1] = &r.omega;
        ints[0] = &r.kInt;
        ints[1] = &r.omegaInt;
    } else {
        R.nS = 1;
        bcs[0] = &r.nuTilda;
        ints[0] = &r.nuTildaInt;
    }
    for (int q = 0; q < R.nS; ++q) {
        R.s0[std::size_t(q)].resize(nC);
        for (std::size_t cc = 0; cc < nC; ++cc) R.s0[std::size_t(q)][cc] = valueAt(*ints[q], fm.srcCell[cc], 0, 1);
        R.bS[std::size_t(q)].resize(nB);
        R.bRef[std::size_t(q)].assign(nB, 0.0);
        R.bS0[std::size_t(q)].resize(nB);
        for (std::size_t b = 0; b < nB; ++b) {
            const FvScalarBc& bc = (*bcs[q])[std::size_t(s.bPatch[b])];
            const std::int64_t local = s.bLocal[b];
            const double xP = R.s0[std::size_t(q)][std::size_t(s.bOwn[b])];
            FvSCode code{};
            double ref = 0, x0 = xP;
            switch (bc.kind) {
            case FvSBc::Fixed: code = FvSCode::Fixed; ref = x0 = valueAt(bc.value, local, 0, 1); break;
            case FvSBc::ZeroGrad: code = FvSCode::ZeroGrad; break; // zeroGradient evaluates at construction
            case FvSBc::InletOutlet:
                code = FvSCode::InletOutlet;
                ref = valueAt(bc.value, local, 0, 1);
                x0 = bc.init.v.empty() ? xP : valueAt(bc.init, local, 0, 1);
                break;
            case FvSBc::OmegaWall: code = FvSCode::OmegaWall; break; // extrapolateInternal()
            case FvSBc::Empty:
                return std::unexpected("patch '" + c.patches[std::size_t(s.bPatch[b])].name +
                                       "': turbulence BC empty on a non-empty patch");
            }
            R.bS[std::size_t(q)][b] = std::uint8_t(code);
            R.bRef[std::size_t(q)][b] = ref;
            R.bS0[std::size_t(q)][b] = x0;
        }
    }
    const std::size_t nI = std::size_t(s.nI);
    R.dlt.resize(3 * nI);
    for (int q = 0; q < 3; ++q)
        for (std::size_t f = 0; f < nI; ++f) R.dlt[std::size_t(q) * nI + f] = fm.d[std::size_t(q)][f];
    R.nut0.resize(nC);
    for (std::size_t cc = 0; cc < nC; ++cc) R.nut0[cc] = valueAt(r.nutInt, fm.srcCell[cc], 0, 1);
    R.bNut.resize(nB);
    R.bNut0.resize(nB);
    for (std::size_t b = 0; b < nB; ++b) {
        const FvNutPatch& np = r.nut[std::size_t(s.bPatch[b])];
        const std::int64_t local = s.bLocal[b];
        FvNutCode code{};
        switch (np.kind) {
        case FvNutBc::Calculated: code = FvNutCode::Calculated; break;
        case FvNutBc::LowRe: code = FvNutCode::LowRe; break;
        case FvNutBc::Spalding: code = FvNutCode::Spalding; break;
        case FvNutBc::KWall: code = FvNutCode::KWall; break;
        case FvNutBc::Cell: code = FvNutCode::Cell; break;
        case FvNutBc::Fixed: code = FvNutCode::Fixed; break;
        case FvNutBc::Empty:
            return std::unexpected("patch '" + c.patches[std::size_t(s.bPatch[b])].name + "': nut empty on a non-empty patch");
        }
        if (code == FvNutCode::KWall && !sst)
            return std::unexpected("patch '" + c.patches[std::size_t(s.bPatch[b])].name +
                                   "': nutkWallFunction needs k (kOmegaSST)");
        R.bNut[b] = std::uint8_t(code);
        R.bNut0[b] = code == FvNutCode::Cell ? R.nut0[std::size_t(s.bOwn[b])] : valueAt(np.value, local, 0, 1);
    }
    return {};
}

} // namespace dns
