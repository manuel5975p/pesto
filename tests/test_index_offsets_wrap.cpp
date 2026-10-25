// offsetsAt (gpu/index.hpp) honours the gidx contract: an index ONE cell off
// the grid wraps, and the offsets returned are those of the wrapped cell --
// not offsets computed from the unwrapped index, which disagree exactly at the
// wrap where the sparse boundary and surface-force kernels live (that drift
// cost an afternoon: a 0.4 freestream error at the NSCBC corner rows).
//
// Exhaustive over every cell and every off-grid index on every axis, with
// nz == 1 (p[2] == m[2] == 0, the exact self-wrap) and degenerate-ish small
// axes. Pure host arithmetic; needs the HIP headers, never a device.
#include "dns/gpu/index.hpp"
#include "check.hpp"

#include <cstdio>
#include <string>

namespace {

using dnstest::check;
using dns::gpu::GpuGrid;
using dns::gpu::Offsets;

bool same(const Offsets& a, const Offsets& b)
{
    for (int d = 0; d < 3; ++d)
        if (a.p[d] != b.p[d] || a.m[d] != b.m[d]) return false;
    return true;
}

// Offsets o must land on the gidx neighbours of the in-grid cell (i, j, k).
bool landsOnNeighbours(const GpuGrid& g, const Offsets& o, int i, int j, int k)
{
    const auto c = std::ptrdiff_t(dns::gpu::gidx(g, i, j, k));
    const int d3[3][3] = {{1, 0, 0}, {0, 1, 0}, {0, 0, 1}};
    for (int d = 0; d < 3; ++d)
    {
        const auto p = std::ptrdiff_t(
            dns::gpu::gidx(g, i + d3[d][0], j + d3[d][1], k + d3[d][2]));
        const auto m = std::ptrdiff_t(
            dns::gpu::gidx(g, i - d3[d][0], j - d3[d][1], k - d3[d][2]));
        if (c + o.p[d] != p || c + o.m[d] != m) return false;
    }
    return true;
}

// One grid: every in-range cell, plus each axis stepped one cell off either
// end with the other two indices swept over their whole range.
void grid(int nx, int ny, int nz)
{
    const GpuGrid g{nx, ny, nz, 1.0f, 1.0f, 1.0f};
    const std::string tag = std::to_string(nx) + "x" + std::to_string(ny) + "x"
                            + std::to_string(nz);
    int badIn = 0, badWrap = 0, badNb = 0, tested = 0;
    for (int k = 0; k < nz; ++k)
        for (int j = 0; j < ny; ++j)
            for (int i = 0; i < nx; ++i)
                if (!landsOnNeighbours(g, dns::gpu::offsetsAt(g, i, j, k), i,
                                       j, k))
                    ++badIn;

    const int n[3] = {nx, ny, nz};
    for (int a = 0; a < 3; ++a)
    {
        const int b = (a + 1) % 3, c = (a + 2) % 3;
        for (int u = 0; u < n[b]; ++u)
            for (int v = 0; v < n[c]; ++v)
                for (int off : {-1, n[a]})
                {
                    int idx[3], wr[3];
                    idx[a] = off;
                    idx[b] = u;
                    idx[c] = v;
                    for (int d = 0; d < 3; ++d) wr[d] = idx[d];
                    wr[a] = off < 0 ? off + n[a] : off - n[a];
                    const Offsets o =
                        dns::gpu::offsetsAt(g, idx[0], idx[1], idx[2]);
                    const Offsets w =
                        dns::gpu::offsetsAt(g, wr[0], wr[1], wr[2]);
                    if (!same(o, w)) ++badWrap;
                    if (!landsOnNeighbours(g, o, wr[0], wr[1], wr[2])) ++badNb;
                    ++tested;
                }
    }
    std::printf("%-10s off-grid probes %5d: wrap mismatches %d, neighbour "
                "misses %d, in-grid misses %d\n",
                tag.c_str(), tested, badWrap, badNb, badIn);
    check(badIn == 0, tag + ": in-grid offsets reach the gidx neighbours");
    check(badWrap == 0,
          tag + ": an index one off the grid gives the WRAPPED cell's offsets");
    check(badNb == 0,
          tag + ": off-grid offsets reach the wrapped cell's neighbours");

    if (nz == 1)
    {
        bool zero = true;
        for (int j = -1; j <= ny; ++j)
            for (int i = -1; i <= nx; ++i)
                for (int k : {-1, 0, 1})
                {
                    const Offsets o = dns::gpu::offsetsAt(g, i, j, k);
                    if (o.p[2] != 0 || o.m[2] != 0) zero = false;
                }
        check(zero, tag + ": nz == 1 gives p[2] == m[2] == 0 everywhere");
    }
}

} // namespace

int main()
{
    grid(7, 5, 1);    // native 2D, odd sizes
    grid(8, 8, 1);
    grid(6, 5, 4);    // all three axes periodic
    grid(2, 3, 2);    // smallest non-degenerate axes: p and m both wrap
    grid(5, 1, 3);    // a degenerate y axis: offsets still self-consistent
    return dnstest::finish();
}
