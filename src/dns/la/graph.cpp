#include "dns/la/graph.hpp"

#include <algorithm>
#include <array>
#include <cassert>
#include <numeric>

namespace dns::la {

CellGraph graphOf(const FaceMesh& fm)
{
    CellGraph g;
    g.n = fm.nC;
    // Neighbour lists from the internal faces, then sorted and made unique:
    // a cell pair joined by several faces (a 2-cell periodic axis, split
    // faces) is ONE entry.
    std::vector<int> deg(std::size_t(fm.nC) + 1, 0);
    for (int f = 0; f < fm.nI; ++f) {
        ++deg[std::size_t(fm.own[std::size_t(f)]) + 1];
        ++deg[std::size_t(fm.nei[std::size_t(f)]) + 1];
    }
    std::vector<int> start(deg);
    std::partial_sum(start.begin(), start.end(), start.begin());
    std::vector<int> nb(std::size_t(start.back()));
    std::vector<int> fill(start.begin(), start.end() - 1);
    for (int f = 0; f < fm.nI; ++f) {
        const int o = fm.own[std::size_t(f)], n = fm.nei[std::size_t(f)];
        nb[std::size_t(fill[std::size_t(o)]++)] = n;
        nb[std::size_t(fill[std::size_t(n)]++)] = o;
    }
    g.rowPtr.assign(std::size_t(fm.nC) + 1, 0);
    for (int c = 0; c < fm.nC; ++c) {
        const auto b = nb.begin() + start[std::size_t(c)], e = nb.begin() + start[std::size_t(c) + 1];
        std::sort(b, e);
        const auto u = std::unique(b, e);
        g.col.insert(g.col.end(), b, u);
        g.rowPtr[std::size_t(c) + 1] = int(g.col.size());
    }
    auto slot = [&](int row, int colIdx) {
        const auto b = g.col.begin() + g.rowPtr[std::size_t(row)],
                   e = g.col.begin() + g.rowPtr[std::size_t(row) + 1];
        const auto it = std::lower_bound(b, e, colIdx);
        assert(it != e && *it == colIdx);
        return int(it - g.col.begin());
    };
    g.ownSlot.resize(std::size_t(fm.nI));
    g.neiSlot.resize(std::size_t(fm.nI));
    for (int f = 0; f < fm.nI; ++f) {
        const int o = fm.own[std::size_t(f)], n = fm.nei[std::size_t(f)];
        g.ownSlot[std::size_t(f)] = slot(o, n);
        g.neiSlot[std::size_t(f)] = slot(n, o);
    }
    return g;
}

std::vector<std::uint8_t> greedyColour(const CellGraph& g)
{
    std::vector<std::uint8_t> colour(std::size_t(g.n), 0);
    std::array<bool, 256> used{};
    for (int r = 0; r < g.n; ++r) {
        used.fill(false);
        for (int q = g.rowPtr[std::size_t(r)]; q < g.rowPtr[std::size_t(r) + 1]; ++q) {
            const int c = g.col[std::size_t(q)];
            if (c < r) used[colour[std::size_t(c)]] = true; // rows below r are coloured
        }
        int k = 0;
        while (used[std::size_t(k)]) ++k;
        assert(k < 256);
        colour[std::size_t(r)] = std::uint8_t(k);
    }
    return colour;
}

std::vector<int> colourMajorOrder(std::span<const std::uint8_t> colours)
{
    std::vector<int> perm(colours.size());
    std::iota(perm.begin(), perm.end(), 0);
    std::ranges::stable_sort(perm, {}, [&](int c) { return colours[std::size_t(c)]; });
    return perm;
}

} // namespace dns::la
