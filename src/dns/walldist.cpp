#include "dns/walldist.hpp"

#include <algorithm>
#include <cassert>

namespace dns {

std::vector<double> channelWallDistance(const CurvMesh& m, double delta)
{
    const Grid& g = m.g;
    assert(delta > 0.0 && g.ny >= 4);
    std::vector<double> d(g.size());
    // The channel mesh is a y-only map, so the eight-node mean y of a cell is
    // the mean of its two node rows -- taken in that form, which rounds once.
    auto centroidY = [&](int i, int j, int k) {
        return 0.5 * (m.node(i, j, k)[1] + m.node(i, j + 1, k)[1]);
    };
    for (int k = 0; k < g.nz; ++k)
        for (int j = 0; j < g.ny; ++j)
            for (int i = 0; i < g.nx; ++i) {
                const int js = std::clamp(j, 1, g.ny - 2); // ghost -> mirrored row
                const double y = centroidY(i, js, k);
                d[g.idx(i, j, k)] = std::min(y, 2.0 * delta - y);
            }
    return d;
}

} // namespace dns
