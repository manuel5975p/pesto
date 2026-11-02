#include "dns/fvincomp.hpp"

#include "dns/fvgrad.hpp"
#include "dns/la/graph.hpp"

#include <tomlplusplus/toml.hpp>

#include <algorithm>
#include <cmath>
#include <format>
#include <map>
#include <numbers>
#include <optional>
#include <set>

namespace dns {
namespace {

using V3 = std::array<double, 3>;

double dot(const V3& a, const V3& b) { return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]; }

V3 cross(const V3& a, const V3& b)
{
    return {a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]};
}

V3 at(const std::array<std::vector<double>, 3>& a, std::size_t i) { return {a[0][i], a[1][i], a[2][i]}; }

std::optional<double> num(const toml::node* n)
{
    if (!n) return std::nullopt;
    if (const auto d = n->value<double>(); d && std::isfinite(*d)) return d;
    return std::nullopt;
}

std::unexpected<std::string> bcErr(std::string m) { return std::unexpected("--bc: " + std::move(m)); }

// A 3-vector entry of t, or an error. Absent: def.
std::expected<V3, std::string> vec3(const toml::table& t, const char* key, const std::string& where, const V3& def = {})
{
    const toml::node* n = t.get(key);
    if (!n) return def;
    const toml::array* a = n->as_array();
    if (!a || a->size() != 3) return bcErr(std::format("{}.{}: must be an array of 3 numbers", where, key));
    V3 v{};
    for (std::size_t i = 0; i < 3; ++i) {
        const auto d = num(a->get(i));
        if (!d) return bcErr(std::format("{}.{}: must be an array of 3 finite numbers", where, key));
        v[i] = *d;
    }
    return v;
}

std::expected<void, std::string> onlyKeys(const toml::table& t, const std::string& where,
                                          std::initializer_list<std::string_view> keys)
{
    for (const auto& [k, v] : t)
        if (std::ranges::find(keys, k.str()) == keys.end())
            return bcErr(std::format("{}: unknown key '{}'", where, k.str()));
    return {};
}

std::expected<FvIncompBc, std::string> kindOf(std::string_view s, const std::string& where)
{
    static constexpr std::pair<std::string_view, FvIncompBc> names[] = {
        {"wall", FvIncompBc::Wall},       {"slip", FvIncompBc::Slip},
        {"inflow", FvIncompBc::Inflow},   {"outflow", FvIncompBc::Outflow},
        {"convective", FvIncompBc::Convective}, {"far", FvIncompBc::Far}};
    for (const auto& [n, k] : names)
        if (n == s) return k;
    return bcErr(std::format("{}.kind: '{}' is not one of wall, slip, inflow, outflow, convective, far", where, s));
}

std::expected<FvIncompPatch, std::string> parsePatch(const toml::table& t, const std::string& name)
{
    const std::string where = "patch." + name;
    if (t.get("type"))
        return bcErr(where + ": key 'type' is the --physics compressible --bc schema; this solver spells it kind");
    if (auto e = onlyKeys(t, where, {"kind", "u", "omega", "origin", "p", "uc", "force", "t"}); !e)
        return std::unexpected(e.error());
    const auto ks = t.get("kind") ? t.get("kind")->value<std::string>() : std::nullopt;
    if (!ks) return bcErr(where + ": needs kind = \"...\"");
    auto kind = kindOf(*ks, where);
    if (!kind) return std::unexpected(kind.error());
    FvIncompPatch p{.name = name, .kind = *kind};
    auto u = vec3(t, "u", where);
    auto om = vec3(t, "omega", where);
    auto org = vec3(t, "origin", where);
    if (!u) return std::unexpected(u.error());
    if (!om) return std::unexpected(om.error());
    if (!org) return std::unexpected(org.error());
    p.u = *u;
    p.omega = *om;
    p.origin = *org;
    const bool isWall = p.kind == FvIncompBc::Wall;
    if (!isWall && (t.get("omega") || t.get("origin"))) return bcErr(where + ": omega / origin belong to a wall");
    const bool usesU = isWall || p.kind == FvIncompBc::Inflow || p.kind == FvIncompBc::Far;
    if (!usesU && t.get("u")) return bcErr(where + ": u belongs to wall, inflow, far");
    if (const toml::node* n = t.get("p")) {
        const auto v = num(n);
        if (!v) return bcErr(where + ".p: must be a finite number");
        if (p.kind != FvIncompBc::Outflow && p.kind != FvIncompBc::Far) return bcErr(where + ": p belongs to outflow, far");
        p.p = *v;
    }
    if (const toml::node* n = t.get("uc")) {
        const auto v = num(n);
        if (!v || !(*v > 0)) return bcErr(where + ".uc: must be > 0");
        if (p.kind != FvIncompBc::Convective) return bcErr(where + ": uc belongs to convective");
        p.uc = *v;
    }
    if (const toml::node* n = t.get("force")) {
        const auto v = n->value<bool>();
        if (!v) return bcErr(where + ".force: must be true or false");
        p.force = *v;
    }
    if (const toml::node* n = t.get("t")) {
        const auto v = num(n);
        if (!v) return bcErr(where + ".t: must be a finite number");
        if (!isWall) return bcErr(where + ": t belongs to a wall");
        p.hasT = true;
        p.T = *v;
    }
    return p;
}

} // namespace

std::expected<FvIncompBcs, std::string> parseFvIncompBcs(std::string_view text)
{
    const auto res = toml::parse(text);
    if (!res)
        return bcErr(std::format("TOML parse error: {} (line {})", res.error().description(),
                                 res.error().source().begin.line));
    const toml::table& root = res.table();
    if (auto e = onlyKeys(root, "top level", {"reference", "initial", "patch"}); !e) return std::unexpected(e.error());
    FvIncompBcs b;
    if (const toml::node* n = root.get("reference")) {
        const toml::table* t = n->as_table();
        if (!t) return bcErr("reference: must be a table");
        b.hasReference = true;
        if (auto e = onlyKeys(*t, "reference", {"u", "length", "area"}); !e) return std::unexpected(e.error());
        const std::pair<const char*, double*> f[] = {{"u", &b.uRef}, {"length", &b.length}, {"area", &b.area}};
        for (const auto& [k, d] : f)
            if (const toml::node* v = t->get(k)) {
                const auto x = num(v);
                if (!x || !(*x > 0)) return bcErr(std::format("reference.{}: must be > 0", k));
                *d = *x;
            }
    }
    if (const toml::node* n = root.get("initial")) {
        const toml::table* t = n->as_table();
        if (!t) return bcErr("initial: must be a table");
        b.hasInitial = true;
        if (auto e = onlyKeys(*t, "initial", {"u"}); !e) return std::unexpected(e.error());
        auto u = vec3(*t, "u", "initial");
        if (!u) return std::unexpected(u.error());
        b.u0 = *u;
    }
    const toml::node* pn = root.get("patch");
    const toml::table* pt = pn ? pn->as_table() : nullptr;
    if (!pt) return bcErr("needs [patch.NAME] tables");
    for (const auto& [k, v] : *pt) {
        const toml::table* t = v.as_table();
        if (!t) return bcErr(std::format("patch.{}: must be a table", k.str()));
        auto p = parsePatch(*t, std::string(k.str()));
        if (!p) return std::unexpected(p.error());
        b.patches.push_back(std::move(*p));
    }
    return b;
}

std::expected<void, std::string> bindFvIncompBcs(FvIncompBcs& b, const FaceMesh& fm)
{
    std::map<std::string, FvIncompPatch> given;
    for (auto& p : b.patches) given.emplace(p.name, std::move(p));
    std::vector<FvIncompPatch> out;
    for (const MeshPatch& m : fm.patches) {
        const auto it = given.find(m.name);
        if (m.type == FoamPatchType::Empty) {
            if (it != given.end()) return bcErr(std::format("patch.{}: is an `empty` patch of the mesh", m.name));
            continue;
        }
        if (it == given.end()) return bcErr(std::format("mesh patch '{}' has no [patch.{}] entry", m.name, m.name));
        if (fm.emptyAxis >= 0 && it->second.kind == FvIncompBc::Wall) {
            const std::size_t e = std::size_t(fm.emptyAxis);
            for (std::size_t a = 0; a < 3; ++a)
                if (a != e && it->second.omega[a] != 0)
                    return bcErr(std::format("patch.{}: a 2D mesh rotates about its empty axis only", m.name));
        }
        if (fm.emptyAxis >= 0 && it->second.u[std::size_t(fm.emptyAxis)] != 0)
            return bcErr(std::format("patch.{}: u along the empty axis of a 2D mesh", m.name));
        out.push_back(std::move(it->second));
        given.erase(it);
    }
    if (!given.empty()) return bcErr(std::format("patch.{}: names no patch of the mesh", given.begin()->first));
    b.patches = std::move(out);
    return {};
}

std::expected<std::string, std::string> describeFvIncompBcs(const FvIncompBcs& b, const PolyManifest& man)
{
    static constexpr std::array<const char*, 6> names = {"wall", "slip", "inflow", "outflow", "convective", "far"};
    std::set<std::string> left;
    for (const auto& p : b.patches) left.insert(p.name);
    std::string s = std::format("  reference  u {:g}, length {:g}, area {:g}\n", b.uRef, b.length, b.area);
    for (const auto& m : man.patches) {
        if (m.type == FoamPatchType::Empty || m.type == FoamPatchType::Cyclic) {
            if (left.contains(m.name)) return bcErr(std::format("patch.{}: is a constraint patch of the mesh", m.name));
            s += std::format("  patch      {:<24} {}\n", m.name, foamPatchTypeName(m.type));
            continue;
        }
        const auto it = std::ranges::find_if(b.patches, [&](const FvIncompPatch& p) { return p.name == m.name; });
        if (it == b.patches.end()) return bcErr(std::format("mesh patch '{}' has no [patch.{}] entry", m.name, m.name));
        left.erase(m.name);
        s += std::format("  patch      {:<24} {}\n", m.name, names[std::size_t(it->kind)]);
    }
    if (!left.empty()) return bcErr(std::format("patch.{}: names no patch of the mesh", *left.begin()));
    return s;
}

std::expected<FvIncompSetup, std::string> buildFvIncompSetup(const FaceMesh& fm, const FvIncompBcs& b, int threads)
{
    auto geom = fvFaceGeom(fm, FvFaceGradRule::AlongD, 1.0, threads);
    if (!geom) return std::unexpected(geom.error());
    auto gw = fvGradWeights(fm, *geom, FvCellGrad::LeastSquares, threads);
    if (!gw) return std::unexpected(gw.error());

    FvIncompSetup s;
    const std::size_t nC = std::size_t(fm.nC), nI = std::size_t(fm.nI);
    s.nC = fm.nC;
    s.nI = fm.nI;
    s.emptyAxis = fm.emptyAxis;
    const int ea = fm.emptyAxis;
    // The offsets are not face data the FaceMesh zeroes on the empty axis.
    auto planar = [&](V3 v) {
        if (ea >= 0) v[std::size_t(ea)] = 0.0;
        return v;
    };
    s.own.assign(fm.own.begin(), fm.own.begin() + fm.nI);
    s.nei = fm.nei;
    s.S.resize(3 * nI);
    s.d.resize(3 * nI);
    s.rO.resize(3 * nI);
    s.k.resize(3 * nI);
    s.id2.resize(nI);
    s.cS.resize(nI);
    s.wO.resize(nI);
    s.a.resize(nI);
    s.gN.resize(3 * nI);
    constexpr double kOrthoTol = 1e-12;
    double cosMin = 1.0;
    for (std::size_t f = 0; f < nI; ++f) {
        const V3 S = at(fm.S, f), d = at(fm.d, f), r = planar(at(geom->rOwn, f));
        const double d2 = dot(d, d), SS = dot(S, S), Sd = dot(S, d), magS = std::sqrt(SS), magD = std::sqrt(d2);
        // The owner weight of OpenFOAM's `linear`: the neighbour side's share.
        const V3 rN{r[0] - d[0], r[1] - d[1], r[2] - d[2]};
        const double so = std::abs(dot(S, r)), sn = std::abs(dot(S, rN));
        const double nd = std::max(Sd / magS, 0.05 * magD);
        s.wO[f] = sn / (so + sn);
        s.id2[f] = 1.0 / d2;
        s.cS[f] = Sd / d2;
        s.a[f] = magS / nd;
        cosMin = std::min(cosMin, Sd / (magS * magD));
        for (std::size_t q = 0; q < 3; ++q) {
            s.S[q * nI + f] = S[q];
            s.d[q * nI + f] = d[q];
            s.rO[q * nI + f] = r[q];
            s.k[q * nI + f] = S[q] - s.a[f] * d[q];
            s.gN[q * nI + f] = gw->nei[q][f];
        }
        const V3 kv{S[0] - s.a[f] * d[0], S[1] - s.a[f] * d[1], S[2] - s.a[f] * d[2]};
        if (std::sqrt(dot(kv, kv)) > kOrthoTol * magS) s.orthogonal = false;
    }
    s.nonOrthMaxDeg = std::acos(std::clamp(cosMin, -1.0, 1.0)) * 180.0 / std::numbers::pi;

    const std::size_t nB = std::size_t(std::ranges::fold_left(fm.patches, 0, [](int n, const MeshPatch& m) {
        return n + (m.type == FoamPatchType::Empty ? 0 : m.count);
    }));
    s.nB = int(nB);
    for (auto* v : {&s.bS, &s.bR, &s.bN, &s.bU}) v->resize(3 * nB);
    std::size_t iPatch = 0, bb = 0;
    for (const MeshPatch& m : fm.patches) {
        if (m.type == FoamPatchType::Empty) continue;
        const FvIncompPatch& p = b.patches[iPatch++];
        for (int i = 0; i < m.count; ++i) {
            const std::size_t f = nI + std::size_t(m.first + i);
            const V3 S = at(fm.S, f), r = planar(at(geom->rOwn, f));
            const double magS = std::sqrt(dot(S, S));
            const V3 n{S[0] / magS, S[1] / magS, S[2] / magS};
            const double Sr = dot(S, r);
            s.bOwn.push_back(fm.own[f]);
            s.bFace.push_back(m.first + i);
            for (std::size_t q = 0; q < 3; ++q) {
                s.bS[q * nB + bb] = S[q];
                s.bR[q * nB + bb] = r[q];
                s.bN[q * nB + bb] = n[q];
            }
            s.bCs.push_back(Sr / dot(r, r));
            s.bA.push_back(dot(S, S) / Sr);
            s.bDn.push_back(Sr / magS);
            s.bMagS.push_back(magS);
            V3 u = p.u;
            FvIncompFaceBc code{};
            switch (p.kind) {
            case FvIncompBc::Wall: {
                code = FvIncompFaceBc::Wall;
                const V3 cf = at(fm.Cf, f);
                const V3 w = cross(p.omega, V3{cf[0] - p.origin[0], cf[1] - p.origin[1], cf[2] - p.origin[2]});
                u = {u[0] + w[0], u[1] + w[1], u[2] + w[2]};
                break;
            }
            case FvIncompBc::Slip: code = FvIncompFaceBc::Slip; break;
            case FvIncompBc::Inflow: code = FvIncompFaceBc::Inflow; break;
            case FvIncompBc::Outflow: code = FvIncompFaceBc::Outflow; break;
            case FvIncompBc::Convective: code = FvIncompFaceBc::Convective; break;
            case FvIncompBc::Far: code = dot(p.u, S) < 0 ? FvIncompFaceBc::Inflow : FvIncompFaceBc::Outflow; break;
            }
            if (ea >= 0) u[std::size_t(ea)] = 0.0;
            s.bCode.push_back(std::uint8_t(code));
            s.bForce.push_back(p.force ? 1 : 0);
            for (std::size_t q = 0; q < 3; ++q) s.bU[q * nB + bb] = u[q];
            s.bP.push_back(p.p);
            s.bUc.push_back(p.kind == FvIncompBc::Convective ? (p.uc > 0 ? p.uc : b.uRef) : 0.0);
            s.bT.push_back(p.T);
            s.bTfix.push_back(p.hasT ? 1 : 0);
            if (code == FvIncompFaceBc::Outflow) s.pureNeumann = false;
            if (code == FvIncompFaceBc::Convective) {
                s.hasConvective = true;
                s.convArea += magS;
            }
            ++bb;
        }
    }
    s.gO.resize(3 * (nI + nB));
    // bFace[b] -> FaceMesh boundary index: re-express the owner weights.
    std::vector<int> bOf(std::size_t(fm.nB), -1);
    for (std::size_t i = 0; i < nB; ++i) bOf[std::size_t(s.bFace[i])] = int(i);
    for (std::size_t q = 0; q < 3; ++q) {
        for (std::size_t f = 0; f < nI; ++f) s.gO[q * (nI + nB) + f] = gw->own[q][f];
        for (std::size_t i = 0; i < nB; ++i) s.gO[q * (nI + nB) + nI + i] = gw->own[q][nI + std::size_t(s.bFace[i])];
    }

    s.V = fm.V;
    s.volume = 0;
    for (double v : fm.V) s.volume += v;
    s.C.resize(3 * nC);
    s.delta2.resize(nC);
    for (std::size_t c = 0; c < nC; ++c) {
        for (std::size_t q = 0; q < 3; ++q) s.C[q * nC + c] = fm.C[q][c];
        s.delta2[c] = std::cbrt(fm.V[c] * fm.V[c]);
    }
    if (ea >= 0) // Delta^2 = cell area (an empty face's), the structured LES's hx hy at nz == 1
        for (std::size_t c = 0; c < nC; ++c)
            for (int e = fm.cellStart[c]; e < fm.cellStart[c + 1]; ++e) {
                const std::size_t f = std::size_t(fm.cellFace[std::size_t(e)]);
                if (f >= nI && bOf[f - nI] < 0) s.delta2[c] = std::abs(fm.S[std::size_t(ea)][f]);
            }
    s.cellStart.assign(fm.cellStart.begin(), fm.cellStart.end());
    s.cellEnt.clear();
    s.cellEnt.reserve(fm.cellFace.size());
    std::vector<int> start{0};
    for (std::size_t c = 0; c < nC; ++c) {
        for (int e = fm.cellStart[c]; e < fm.cellStart[c + 1]; ++e) {
            const int f = fm.cellFace[std::size_t(e)];
            if (std::size_t(f) < nI) {
                const int side = fm.own[std::size_t(f)] == int(c) ? 0 : 1;
                s.cellEnt.push_back(2 * f + side);
            } else {
                const int bf = bOf[std::size_t(f) - nI];
                if (bf >= 0) s.cellEnt.push_back(-(bf + 1));
            }
        }
        start.push_back(int(s.cellEnt.size()));
    }
    s.cellStart = std::move(start);

    // The compact pressure matrix: positive definite with off = -a.
    const auto g = la::graphOf(fm);
    s.slotA.assign(g.col.size(), 0.0);
    s.diagA.assign(nC, 0.0);
    for (std::size_t f = 0; f < nI; ++f) {
        s.slotA[std::size_t(g.ownSlot[f])] -= s.a[f];
        s.slotA[std::size_t(g.neiSlot[f])] -= s.a[f];
        s.diagA[std::size_t(s.own[f])] += s.a[f];
        s.diagA[std::size_t(s.nei[f])] += s.a[f];
    }
    for (std::size_t i = 0; i < nB; ++i)
        if (s.bCode[i] == std::uint8_t(FvIncompFaceBc::Outflow)) s.diagA[std::size_t(s.bOwn[i])] += s.bA[i];
    return s;
}

namespace {

std::uint64_t mix64(std::uint64_t z)
{
    z += 0x9E3779B97F4A7C15ull;
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
    return z ^ (z >> 31);
}

} // namespace

double fvHashUnit(std::uint64_t seed, std::uint64_t node, int axis)
{
    const std::uint64_t h = mix64(mix64(seed ^ (node * 3 + std::uint64_t(axis))));
    return double(h >> 11) * 0x1.0p-52 - 1.0;
}

std::expected<FaceMesh, std::string> fvPeriodicBox(const Grid& g, FvPerturb kind, double amp, std::uint64_t seed,
                                                   int threads)
{
    CurvMesh m = kind == FvPerturb::Sine ? CurvMesh::sinusoidal(g, amp) : CurvMesh::uniform(g);
    if (kind == FvPerturb::Random)
        for (std::size_t c = 0; c < m.x.size(); ++c) {
            m.x[c] += amp * g.hx * fvHashUnit(seed, c, 0);
            m.y[c] += amp * g.hy * fvHashUnit(seed, c, 1);
            if (g.nz > 1) m.z[c] += amp * g.hz * fvHashUnit(seed, c, 2);
        }
    const bool twoD = g.nz == 1;
    const FoamPatchType zt = twoD ? FoamPatchType::Empty : FoamPatchType::Cyclic;
    const StructuredSpec sp{.lo = {0, 0, 0},
                            .hi = {g.nx - 1, g.ny - 1, g.nz - 1},
                            .name = {"xlo", "xhi", "ylo", "yhi", "zlo", "zhi"},
                            .type = {FoamPatchType::Cyclic, FoamPatchType::Cyclic, FoamPatchType::Cyclic,
                                     FoamPatchType::Cyclic, zt, zt}};
    return FaceMesh::fromCurvMesh(m, sp, CellOrder::Morton, threads);
}

std::expected<FaceMesh, std::string> fvOgridMesh(const OGrid& og, int threads)
{
    const Grid& g = og.mesh.g;
    const bool twoD = g.nz == 1;
    const FoamPatchType zt = twoD ? FoamPatchType::Empty : FoamPatchType::Cyclic;
    const StructuredSpec sp{.lo = {0, 1, 0},
                            .hi = {g.nx - 1, g.ny - 2, g.nz - 1},
                            .name = {"ilo", "ihi", "wall", "far", "zlo", "zhi"},
                            .type = {FoamPatchType::Cyclic, FoamPatchType::Cyclic, FoamPatchType::Wall,
                                     FoamPatchType::Patch, zt, zt}};
    return FaceMesh::fromCurvMesh(og.mesh, sp, CellOrder::Morton, threads);
}

std::string fvIncompLabel(const FvIncompControls& c, std::string_view mesh)
{
    const std::string what =
        c.les > 0 ? std::format("LES {} {} {:g}", c.sgs == SgsModel::Sigma ? "sigma" : "Smagorinsky",
                                c.sgs == SgsModel::Sigma ? "Csigma" : "Cs", c.les)
                  : std::string("DNS");
    return std::format("{} [FV path, {}, {}] incompressible collocated SSP-RK3 + per-stage projection, "
                       "skew-symmetric central convection, nu {:g}",
                       what, mesh, fvPrecisionName(c.precision), c.nu);
}

} // namespace dns
