#include "dns/fvconv.hpp"

#include <algorithm>
#include <cmath>
#include <format>
#include <numbers>

namespace dns {
namespace {

constexpr double kPlateTol = 1e-9;

bool same(double a, double b) { return std::abs(a - b) <= 1e-12 * (1.0 + std::abs(b)); }

std::unexpected<std::string> bcErr(std::string m) { return std::unexpected("--bc: " + std::move(m)); }

bool nonzero(const std::array<double, 3>& v)
{
    return std::ranges::any_of(v, [](double x) { return x != 0.0; });
}

// Every boundary face of patch p within tol of height y.
bool flatAt(const FaceMesh& fm, const MeshPatch& p, double y)
{
    for (int i = 0; i < p.count; ++i)
        if (std::abs(fm.Cf[1][std::size_t(fm.nI + p.first + i)] - y) > kPlateTol) return false;
    return true;
}

} // namespace

CurvMesh fvConvCurv(const Grid& g, bool closed, FvPerturb kind, double amp, std::uint64_t seed)
{
    CurvMesh m = CurvMesh::uniform(g);
    const bool twoD = g.nz == 1;
    const int ox = closed ? 1 : 0, oz = closed && !twoD ? 1 : 0;
    const double H = double(g.ny - 2) * g.hy, Lx = double(g.nx - 2 * ox) * g.hx;
    const double pi = std::numbers::pi;
    for (int k = 0; k < g.nz; ++k)
        for (int j = 0; j < g.ny; ++j)
            for (int i = 0; i < g.nx; ++i) {
                const std::size_t c = g.idx(i, j, k);
                m.x[c] -= ox * g.hx;
                m.y[c] -= g.hy;
                m.z[c] -= oz * g.hz;
                const double x = m.x[c], y = m.y[c];
                if (kind == FvPerturb::Sine) {
                    m.y[c] += amp * H * std::sin(2 * pi * x / Lx) * std::sin(pi * y / H);
                    if (!closed) m.x[c] += amp * H * std::sin(2 * pi * y / H);
                }
                if (kind == FvPerturb::Random) {
                    const bool sideNode = closed && (i <= ox || i >= g.nx - ox);
                    const bool plateNode = j <= 1 || j >= g.ny - 1;
                    const bool zNode = twoD || (closed && (k <= oz || k >= g.nz - oz));
                    if (!sideNode) m.x[c] += amp * g.hx * fvHashUnit(seed, c, 0);
                    if (!plateNode) m.y[c] += amp * g.hy * fvHashUnit(seed, c, 1);
                    if (!zNode) m.z[c] += amp * g.hz * fvHashUnit(seed, c, 2);
                }
            }
    return m;
}

StructuredSpec fvConvSpec(const Grid& g, bool closed)
{
    const bool twoD = g.nz == 1;
    const int ox = closed ? 1 : 0, oz = closed && !twoD ? 1 : 0;
    const FoamPatchType xt = closed ? FoamPatchType::Wall : FoamPatchType::Cyclic;
    const FoamPatchType zt = twoD ? FoamPatchType::Empty : (closed ? FoamPatchType::Wall : FoamPatchType::Cyclic);
    return {.lo = {ox, 1, oz},
            .hi = {g.nx - 1 - ox, g.ny - 2, g.nz - 1 - oz},
            .name = {"xlo", "xhi", "floor", "lid", "back", "front"},
            .type = {xt, xt, FoamPatchType::Wall, FoamPatchType::Wall, zt, zt}};
}

std::expected<FaceMesh, std::string> fvConvBox(const Grid& g, bool closed, FvPerturb kind, double amp, int threads)
{
    return FaceMesh::fromCurvMesh(fvConvCurv(g, closed, kind, amp), fvConvSpec(g, closed), CellOrder::Morton, threads);
}

std::expected<FvConvPlates, std::string> fvConvPlates(const FaceMesh& fm, std::span<const FvConvWall> walls, double tBot,
                                                      double tTop, double H)
{
    FvConvPlates o;
    for (std::size_t p = 0; p < fm.patches.size(); ++p) {
        const MeshPatch& m = fm.patches[p];
        const auto it = std::ranges::find(walls, m.name, &FvConvWall::name);
        if (it == walls.end() || !it->fixed) continue;
        const bool isFloor = same(it->T, tBot);
        if (!isFloor && !same(it->T, tTop))
            return std::unexpected(std::format("patch `{}`: T {:g} is neither the floor's {:g} nor the lid's {:g}", m.name,
                                               it->T, tBot, tTop));
        if (!flatAt(fm, m, isFloor ? 0.0 : H))
            return std::unexpected(std::format("patch `{}`: the {} must be flat at y = {:g} (box units, y up)", m.name,
                                               isFloor ? "floor" : "lid", isFloor ? 0.0 : H));
        (isFloor ? o.floor : o.lid).push_back(int(p));
    }
    if (o.floor.empty()) return std::unexpected(std::format("no patch at the floor temperature {:g}", tBot));
    if (o.lid.empty()) return std::unexpected(std::format("no patch at the lid temperature {:g}", tTop));
    const auto [lo, hi] = std::ranges::minmax(fm.C[1]);
    if (!(lo > 0.0) || !(hi < H)) return std::unexpected(std::format("cells outside the layer 0 < y < {:g}", H));
    return o;
}

std::expected<std::vector<FvConvWall>, std::string> fvConvWalls(const FvBcSpec& s)
{
    if (s.length != 1.0 || s.area != 0.0 || s.dir != std::array<double, 3>{1, 0, 0} ||
        s.lift != std::array<double, 3>{0, 1, 0})
        return bcErr("[reference] has no meaning in dns conv (box units, y up)");
    std::vector<FvConvWall> w;
    for (const FvPatchSpec& p : s.patches) {
        if (p.kind == FvPatchKind::Empty) continue;
        if (p.kind != FvPatchKind::Wall && p.kind != FvPatchKind::Slip)
            return bcErr(std::format("patch.{}: dns conv takes wall, slip and empty patches: the layer is closed (an "
                                     "open face on a hydrostatic ambient is dns flame --mode candle's)",
                                     p.name));
        if (nonzero(p.uWall) || nonzero(p.omega)) return bcErr(std::format("patch.{}: dns conv walls rest", p.name));
        w.push_back({.name = p.name, .fixed = p.isothermal, .T = p.tWall});
    }
    return w;
}

std::expected<std::vector<FvConvWall>, std::string> fvConvWalls(const FvIncompBcs& b)
{
    if (b.hasReference || b.hasInitial) return bcErr("[reference] and [initial] have no meaning in dns conv");
    std::vector<FvConvWall> w;
    for (const FvIncompPatch& p : b.patches) {
        if (p.kind != FvIncompBc::Wall && p.kind != FvIncompBc::Slip)
            return bcErr(std::format("patch.{}: dns conv takes wall and slip patches", p.name));
        if (nonzero(p.u) || nonzero(p.omega)) return bcErr(std::format("patch.{}: dns conv walls rest", p.name));
        w.push_back({.name = p.name, .fixed = p.hasT, .T = p.T});
    }
    return w;
}

std::vector<FvConvWall> fvConvBoxWalls(const FaceMesh& fm, double tBot, double tTop)
{
    std::vector<FvConvWall> w;
    for (const MeshPatch& p : fm.patches) {
        if (p.type == FoamPatchType::Empty) continue;
        const bool fl = p.name == "floor", li = p.name == "lid";
        w.push_back({.name = p.name, .fixed = fl || li, .T = fl ? tBot : tTop});
    }
    return w;
}

FvIncompBcs fvConvIncompBcsOf(std::span<const FvConvWall> walls)
{
    FvIncompBcs b;
    for (const FvConvWall& w : walls)
        b.patches.push_back({.name = w.name, .kind = FvIncompBc::Wall, .hasT = w.fixed, .T = w.T});
    return b;
}

FvBcSpec fvConvCompBcOf(std::span<const FvConvWall> walls)
{
    FvBcSpec spec;
    for (const FvConvWall& w : walls)
        spec.patches.push_back({.name = w.name, .kind = FvPatchKind::Wall, .isothermal = w.fixed, .tWall = w.T});
    return spec;
}

} // namespace dns
