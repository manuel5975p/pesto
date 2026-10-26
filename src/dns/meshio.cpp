#include "dns/meshio.hpp"

#include <cassert>
#include <cstdint>
#include <cstring>

namespace dns {

namespace {

void put(std::vector<char>& b, const void* p, std::size_t n)
{
    const auto* c = static_cast<const char*>(p);
    b.insert(b.end(), c, c + n);
}
void u32(std::vector<char>& b, std::uint32_t v) { put(b, &v, sizeof v); }
void f64(std::vector<char>& b, double v) { put(b, &v, sizeof v); }

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

} // namespace dns
