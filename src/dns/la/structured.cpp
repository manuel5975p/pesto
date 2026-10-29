#include "dns/la/structured.hpp"

#include <algorithm>
#include <cassert>

namespace dns::la {

CellGraph structuredGraph(std::array<int, 3> n, std::array<bool, 3> periodic)
{
    assert(n[0] >= 1 && n[1] >= 1 && n[2] >= 1);
    CellGraph g;
    g.n = n[0] * n[1] * n[2];
    g.rowPtr.reserve(std::size_t(g.n) + 1);
    g.rowPtr.push_back(0);
    const int stride[3] = {1, n[0], n[0] * n[1]};
    std::vector<int> nb;
    for (int k = 0; k < n[2]; ++k)
        for (int j = 0; j < n[1]; ++j)
            for (int i = 0; i < n[0]; ++i) {
                const int ijk[3] = {i, j, k};
                const int c = i + n[0] * (j + n[1] * k);
                nb.clear();
                for (int a = 0; a < 3; ++a) {
                    if (n[a] == 1) continue;
                    for (const int s : {-1, 1}) {
                        int m = ijk[a] + s;
                        if (m < 0 || m >= n[a]) {
                            if (!periodic[a]) continue;
                            m = (m + n[a]) % n[a];
                        }
                        nb.push_back(c + (m - ijk[a]) * stride[a]);
                    }
                }
                std::ranges::sort(nb);
                const auto u = std::ranges::unique(nb);
                nb.erase(u.begin(), u.end());
                std::erase(nb, c);
                g.col.insert(g.col.end(), nb.begin(), nb.end());
                g.rowPtr.push_back(int(g.col.size()));
            }
    return g;
}

LineSet gridLines(std::array<int, 3> n, int axis)
{
    assert(axis >= 0 && axis < 3);
    const int a1 = axis == 0 ? 1 : 0, a2 = axis == 2 ? 1 : 2; // the other two, lower fastest
    const int stride[3] = {1, n[0], n[0] * n[1]};
    LineSet ls;
    ls.n = n[0] * n[1] * n[2];
    ls.start.push_back(0);
    for (int q = 0; q < n[a2]; ++q)
        for (int p = 0; p < n[a1]; ++p) {
            for (int t = 0; t < n[axis]; ++t) ls.cells.push_back(p * stride[a1] + q * stride[a2] + t * stride[axis]);
            ls.start.push_back(int(ls.cells.size()));
        }
    return ls;
}

} // namespace dns::la
