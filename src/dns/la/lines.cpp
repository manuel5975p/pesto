#include "dns/la/lines.hpp"

#include <algorithm>
#include <cassert>
#include <deque>
#include <limits>
#include <numeric>

namespace dns::la {

namespace {

// The slot of column j in row c, -1 when absent.
int slotOf(const CellGraph& g, int c, int j)
{
    const auto b = g.col.begin() + g.rowPtr[std::size_t(c)], e = g.col.begin() + g.rowPtr[std::size_t(c) + 1];
    const auto it = std::lower_bound(b, e, j);
    return it != e && *it == j ? int(it - g.col.begin()) : -1;
}

} // namespace

int LineSet::maxLength() const
{
    int m = 0;
    for (int l = 0; l < count(); ++l) m = std::max(m, start[std::size_t(l) + 1] - start[std::size_t(l)]);
    return m;
}

std::expected<LineSet, std::string> lineSet(const CellGraph& g, std::span<const std::vector<int>> chains)
{
    LineSet ls;
    ls.n = g.n;
    ls.start.push_back(0);
    std::vector<char> used(std::size_t(g.n), 0);
    for (std::size_t l = 0; l < chains.size(); ++l) {
        const std::vector<int>& ch = chains[l];
        const std::string at = "line " + std::to_string(l);
        if (ch.empty()) return std::unexpected(at + ": empty chain");
        for (std::size_t i = 0; i < ch.size(); ++i) {
            const int c = ch[i];
            if (c < 0 || c >= g.n) return std::unexpected(at + ": row " + std::to_string(c) + " out of range");
            if (used[std::size_t(c)]) return std::unexpected(at + ": row " + std::to_string(c) + " in two chains");
            used[std::size_t(c)] = 1;
            if (i > 0 && slotOf(g, ch[i - 1], c) < 0)
                return std::unexpected(at + ": rows " + std::to_string(ch[i - 1]) + " and " + std::to_string(c) +
                                       " are not adjacent");
            ls.cells.push_back(c);
        }
        ls.start.push_back(int(ls.cells.size()));
    }
    for (int c = 0; c < g.n; ++c) {
        if (used[std::size_t(c)]) continue;
        ls.cells.push_back(c);
        ls.start.push_back(int(ls.cells.size()));
    }
    return ls;
}

LineSet strongLines(const CellGraph& g, std::span<const double> slotWeight, double ratio)
{
    assert(slotWeight.size() == g.col.size() && ratio > 1);
    const std::size_t n = std::size_t(g.n);
    std::vector<double> wMin(n, 0.0), wMax(n, 0.0);
    std::vector<char> aniso(n, 0);
    std::vector<double> key(n, 0.0);
    for (std::size_t c = 0; c < n; ++c) {
        const int b = g.rowPtr[c], e = g.rowPtr[c + 1];
        if (b == e) continue;
        const auto [mn, mx] = std::minmax_element(slotWeight.begin() + b, slotWeight.begin() + e);
        wMin[c] = *mn;
        wMax[c] = *mx;
        aniso[c] = wMax[c] > 0 && wMax[c] >= ratio * wMin[c];
        key[c] = !aniso[c] ? 0.0 : wMin[c] > 0 ? wMax[c] / wMin[c] : std::numeric_limits<double>::infinity();
    }
    auto strong = [&](int c, int slot) {
        const int j = g.col[std::size_t(slot)];
        if (!aniso[std::size_t(c)] || !aniso[std::size_t(j)]) return false;
        const int back = slotOf(g, j, c);
        assert(back >= 0);
        return slotWeight[std::size_t(slot)] >= ratio * wMin[std::size_t(c)] &&
               slotWeight[std::size_t(back)] >= ratio * wMin[std::size_t(j)];
    };
    std::vector<char> used(n, 0);
    // The strongest unassigned strong neighbour of c, -1 when none.
    auto next = [&](int c) {
        int best = -1;
        double bw = -1;
        for (int q = g.rowPtr[std::size_t(c)]; q < g.rowPtr[std::size_t(c) + 1]; ++q) {
            const int j = g.col[std::size_t(q)];
            if (used[std::size_t(j)] || !strong(c, q)) continue;
            if (slotWeight[std::size_t(q)] > bw) { // ascending columns: ties keep the lower
                bw = slotWeight[std::size_t(q)];
                best = j;
            }
        }
        return best;
    };
    std::vector<int> seeds(n);
    std::iota(seeds.begin(), seeds.end(), 0);
    std::ranges::stable_sort(seeds, std::greater<>{}, [&](int c) { return key[std::size_t(c)]; });
    std::vector<std::vector<int>> chains;
    for (const int s : seeds) {
        if (!aniso[std::size_t(s)] || used[std::size_t(s)]) continue;
        std::deque<int> ch{s};
        used[std::size_t(s)] = 1;
        for (int c = next(s); c >= 0; c = next(c)) {
            ch.push_back(c);
            used[std::size_t(c)] = 1;
        }
        for (int c = next(s); c >= 0; c = next(c)) {
            ch.push_front(c);
            used[std::size_t(c)] = 1;
        }
        if (ch.size() > 1) chains.emplace_back(ch.begin(), ch.end());
        else used[std::size_t(s)] = 0; // a lone seed stays a singleton
    }
    auto ls = lineSet(g, chains);
    assert(ls);
    return std::move(*ls);
}

} // namespace dns::la
