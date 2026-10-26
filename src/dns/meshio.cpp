#include "dns/meshio.hpp"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <format>
#include <numbers>

namespace dns {

namespace {

void put(std::vector<char>& b, const void* p, std::size_t n)
{
    const auto* c = static_cast<const char*>(p);
    b.insert(b.end(), c, c + n);
}
void u32(std::vector<char>& b, std::uint32_t v) { put(b, &v, sizeof v); }
void f64(std::vector<char>& b, double v) { put(b, &v, sizeof v); }

// Sequential little-endian reader over a byte buffer; ok() false once a read
// ran past the end (every later read returns 0).
struct Reader
{
    const std::vector<char>& b;
    std::size_t at = 0;
    bool bad = false;
    bool take(void* p, std::size_t n)
    {
        if (bad || at + n > b.size()) {
            bad = true;
            std::memset(p, 0, n);
            return false;
        }
        std::memcpy(p, b.data() + at, n);
        at += n;
        return true;
    }
    std::uint32_t u32()
    {
        std::uint32_t v = 0;
        take(&v, sizeof v);
        return v;
    }
    double f64()
    {
        double v = 0;
        take(&v, sizeof v);
        return v;
    }
    bool magic(const char* m)
    {
        char s[8];
        return take(s, 8) && std::memcmp(s, m, 8) == 0;
    }
    bool ok() const { return !bad; }
};

} // namespace

std::vector<char> encodeMesh(const OGrid& og, const double centre[3],
                             const std::vector<double>& d1)
{
    const Grid& g = og.mesh.g;
    const std::size_t n = g.size(), nf = std::size_t(g.nx) * g.nz;
    assert(d1.empty() || d1.size() == nf);
    std::vector<char> b;
    b.reserve(96 + 8 * (3 * n + nf));
    put(b, "DNSMESH1", 8);
    u32(b, 1);
    u32(b, std::uint32_t(g.nx));
    u32(b, std::uint32_t(g.ny));
    u32(b, std::uint32_t(g.nz));
    for (const AxisWrap w : og.mesh.wrap) u32(b, std::uint32_t(w));
    const std::uint32_t flags = 1u | (og.revolved ? 2u : 0u) |
                                (og.revolved && og.rev.wedge ? 4u : 0u) |
                                (og.revolved ? 8u : 0u);
    u32(b, flags);
    u32(b, std::uint32_t(og.nTheta));
    f64(b, og.revolved ? og.rev.dphi : 0.0);
    for (int a = 0; a < 3; ++a) f64(b, centre[a]);
    f64(b, og.mesh.mirrorZ0);
    for (const auto* v : {&og.mesh.x, &og.mesh.y, &og.mesh.z})
        put(b, v->data(), n * sizeof(double));
    if (d1.empty()) b.insert(b.end(), nf * sizeof(double), 0);
    else put(b, d1.data(), nf * sizeof(double));
    return b;
}

std::vector<char> encodeFinal(const Grid& g,
                              const std::vector<std::string>& names,
                              const std::vector<std::vector<double>>& fields)
{
    assert(names.size() == fields.size());
    std::vector<char> b;
    put(b, "DNSFINL1", 8);
    u32(b, 1);
    u32(b, std::uint32_t(g.nx));
    u32(b, std::uint32_t(g.ny));
    u32(b, std::uint32_t(g.nz));
    u32(b, std::uint32_t(names.size()));
    for (const auto& s : names) {
        assert(s.size() < 16);
        char name[16] = {};
        std::memcpy(name, s.data(), s.size());
        put(b, name, 16);
    }
    for (const auto& f : fields) {
        assert(f.size() == g.size());
        put(b, f.data(), f.size() * sizeof(double));
    }
    return b;
}

std::expected<WedgeStart, std::string>
revolveWedgeState(const std::vector<char>& mesh, const std::vector<char>& fin,
                  const OGrid& og, const double centre[3],
                  const std::vector<std::string>& scalars, double pInf)
{
    assert(og.revolved && !og.rev.wedge);
    auto bad = [](std::string s) { return std::unexpected(std::move(s)); };
    const Grid& g = og.mesh.g;

    // --- the wedge mesh: header, then its meridional nodes against ours.
    Reader m{mesh};
    if (!m.magic("DNSMESH1") || m.u32() != 1)
        return bad("the mesh file is not a dns --emit mesh file (DNSMESH1 v1)");
    const std::uint32_t nx = m.u32(), ny = m.u32(), nz = m.u32();
    for (int a = 0; a < 3; ++a) (void)m.u32(); // wrap
    const std::uint32_t flags = m.u32(), nTheta = m.u32();
    (void)m.f64(); // dphi
    for (int a = 0; a < 4; ++a) (void)m.f64(); // centre, mirrorZ0
    if (!m.ok() || !(flags & 8u) || !(flags & 4u) || nz != 3)
        return bad("the mesh file is not a revolved WEDGE (--ogrid --body "
                   "sphere --nz 1): --init-from takes a wedge run");
    if (int(nx) != g.nx || int(ny) != g.ny || int(nTheta) != og.nTheta)
        return bad(std::format("the wedge is {} meridional x {} radial cells, "
                               "this run {} x {}: --n and --ny must match",
                               nTheta, ny, og.nTheta, g.ny));
    const std::size_t nw = std::size_t(nx) * ny * nz;
    std::vector<double> X(nw), Y(nw), Z(nw);
    m.take(X.data(), nw * sizeof(double));
    m.take(Y.data(), nw * sizeof(double));
    m.take(Z.data(), nw * sizeof(double));
    if (!m.ok()) return bad("the mesh file is truncated");
    const Grid gw = Grid::make(int(nx), int(ny), 3, 1, 1, 1);
    double scale = 0;
    for (std::size_t q = 0; q < og.mesh.x.size(); ++q)
        scale = std::max({scale, std::abs(og.mesh.x[q] - centre[0]),
                          std::abs(og.mesh.y[q] - centre[1])});
    const double tol = 1e-9 * scale;
    for (int j = 0; j < g.ny; ++j)
        for (int i = 1; i <= og.nTheta + 1; ++i) {
            const std::size_t w = gw.idx(i, j, 1), t = g.idx(i, j, 0);
            const double dx = std::abs(X[w] - og.mesh.x[t]);
            const double dr = std::abs(
                std::hypot(Y[w] - centre[1], Z[w] - centre[2]) -
                std::hypot(og.mesh.y[t] - centre[1], og.mesh.z[t] - centre[2]));
            if (!(dx <= tol && dr <= tol))
                return bad(std::format(
                    "the wedge's meridional node ({}, {}) is {:.3e} off this "
                    "run's (tolerance {:.1e}): --ogrid-beta, --ogrid-far and "
                    "the body must match",
                    i, j, std::max(dx, dr), tol));
        }

    // --- the wedge state.
    Reader f{fin};
    if (!f.magic("DNSFINL1") || f.u32() != 1)
        return bad("the state file is not a dns --emit mesh _final.bin "
                   "(DNSFINL1 v1)");
    if (f.u32() != nx || f.u32() != ny || f.u32() != nz)
        return bad("the state file's grid is not its mesh file's");
    const std::uint32_t nf = f.u32();
    std::vector<std::string> names(nf);
    for (auto& s : names) {
        char c[16];
        f.take(c, 16);
        s.assign(c, strnlen(c, 16));
    }
    std::vector<std::vector<double>> fld(nf, std::vector<double>(nw));
    for (auto& v : fld) f.take(v.data(), nw * sizeof(double));
    if (!f.ok()) return bad("the state file is truncated");
    auto field = [&](const std::string& n) -> const std::vector<double>* {
        const auto it = std::ranges::find(names, n);
        return it == names.end() ? nullptr : &fld[std::size_t(it - names.begin())];
    };
    const std::vector<double>* prim[5];
    const char* pn[5] = {"rho", "u", "v", "w", "p"};
    for (int a = 0; a < 5; ++a)
        if (!(prim[a] = field(pn[a])))
            return bad(std::format("the state file has no '{}' field", pn[a]));
    std::vector<std::string> have;
    for (const char* s : {"nutilde", "k", "omega"})
        if (field(s)) have.emplace_back(s);
    if (have != scalars) {
        auto list = [](const std::vector<std::string>& v) {
            std::string s;
            for (const auto& x : v) s += (s.empty() ? "" : ", ") + x;
            return s.empty() ? std::string("none (laminar)") : s;
        };
        return bad(std::format("the wedge carries closure scalars {}, this "
                               "run needs {}: --rans must match",
                               list(have), list(scalars)));
    }
    double pFar = 0;
    for (int i = 1; i <= og.nTheta; ++i)
        pFar += (*prim[4])[gw.idx(i, int(ny) - 2, 1)] / og.nTheta;
    if (!(std::abs(pFar / pInf - 1) <= 0.02))
        return bad(std::format("the wedge's far-row pressure {:.6g} is not "
                               "this run's freestream {:.6g}: --mach must "
                               "match",
                               pFar, pInf));
    for (std::size_t q = 0; q < nw; ++q)
        if (!((*prim[0])[q] > 0 && (*prim[4])[q] > 0 &&
              std::isfinite((*prim[1])[q] + (*prim[2])[q] + (*prim[3])[q])))
            return bad("the state file holds a non-physical cell (rho or p "
                       "<= 0, or a non-finite velocity)");

    // --- revolve the real plane over the N planes.
    WedgeStart s;
    for (auto* v : {&s.rho, &s.u, &s.v, &s.w, &s.p}) v->resize(g.size());
    s.phi.assign(scalars.size(), std::vector<double>(g.size()));
    std::vector<const std::vector<double>*> sf;
    for (const auto& n : scalars) sf.push_back(field(n));
    for (int k = 0; k < g.nz; ++k) {
        const double ph = 2.0 * std::numbers::pi * (k + 0.5) / g.nz;
        const double c = std::cos(ph), sn = std::sin(ph);
        for (int j = 0; j < g.ny; ++j)
            for (int i = 0; i < g.nx; ++i) {
                const std::size_t w = gw.idx(i, j, 1), t = g.idx(i, j, k);
                const double v = (*prim[2])[w], ww = (*prim[3])[w];
                s.rho[t] = (*prim[0])[w];
                s.u[t] = (*prim[1])[w];
                s.v[t] = c * v - sn * ww;
                s.w[t] = sn * v + c * ww;
                s.p[t] = (*prim[4])[w];
                for (std::size_t a = 0; a < scalars.size(); ++a)
                    s.phi[a][t] = (*sf[a])[w];
            }
    }
    return s;
}

} // namespace dns
