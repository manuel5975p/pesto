#include "dns/la/amg.hpp"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <numeric>
#include <utility>

namespace dns::la {

namespace {

// One pairwise pass (Notay 2010): S_i = {j : w_ij >= beta max_k w_ik > 0};
// the next row is an unassigned one with the FEWEST unassigned rows j having
// it in S_j (a bucket queue; within a bucket the most recently filed first);
// it takes its strongest unassigned strong neighbour (ties: the lower
// column) or stays a singleton. Deterministic. Returns agg and the count.
std::pair<std::vector<int>, int> pairwise(const CellGraph& g, std::span<const double> w, double beta)
{
    const std::size_t n = std::size_t(g.n);
    std::vector<char> strong(g.col.size(), 0);
    std::vector<int> cnt(n, 0);
    for (int i = 0; i < g.n; ++i) {
        const int k0 = g.rowPtr[std::size_t(i)], k1 = g.rowPtr[std::size_t(i) + 1];
        double mx = 0;
        for (int k = k0; k < k1; ++k) mx = std::max(mx, w[std::size_t(k)]);
        if (!(mx > 0)) continue;
        for (int k = k0; k < k1; ++k)
            if (w[std::size_t(k)] >= beta * mx) {
                strong[std::size_t(k)] = 1;
                ++cnt[std::size_t(g.col[std::size_t(k)])];
            }
    }
    // bucket queue over cnt: doubly linked, head insertion
    const int maxCnt = n ? *std::ranges::max_element(cnt) : 0;
    std::vector<int> head(std::size_t(maxCnt) + 1, -1), next(n, -1), prev(n, -1);
    auto unlink = [&](int i) {
        const std::size_t u = std::size_t(i);
        if (prev[u] >= 0) next[std::size_t(prev[u])] = next[u];
        else head[std::size_t(cnt[u])] = next[u];
        if (next[u] >= 0) prev[std::size_t(next[u])] = prev[u];
    };
    auto file = [&](int i) {
        const std::size_t u = std::size_t(i);
        prev[u] = -1;
        next[u] = head[std::size_t(cnt[u])];
        if (next[u] >= 0) prev[std::size_t(next[u])] = i;
        head[std::size_t(cnt[u])] = i;
    };
    for (int i = g.n - 1; i >= 0; --i) file(i); // ascending rows at each bucket's head
    std::vector<int> agg(n, -1);
    int lo = 0, nc = 0;
    auto take = [&](int i) { // assign i: rows strongly coupled FROM i lose one count
        unlink(i);
        agg[std::size_t(i)] = nc;
        for (int k = g.rowPtr[std::size_t(i)]; k < g.rowPtr[std::size_t(i) + 1]; ++k) {
            const int j = g.col[std::size_t(k)];
            if (!strong[std::size_t(k)] || agg[std::size_t(j)] >= 0) continue;
            unlink(j);
            --cnt[std::size_t(j)];
            file(j);
            lo = std::min(lo, cnt[std::size_t(j)]);
        }
    };
    for (int done = 0; done < g.n;) {
        while (head[std::size_t(lo)] < 0) ++lo;
        const int i = head[std::size_t(lo)];
        int best = -1;
        double bw = 0;
        for (int k = g.rowPtr[std::size_t(i)]; k < g.rowPtr[std::size_t(i) + 1]; ++k) {
            const int j = g.col[std::size_t(k)];
            if (strong[std::size_t(k)] && agg[std::size_t(j)] < 0 && w[std::size_t(k)] > bw) {
                best = j;
                bw = w[std::size_t(k)];
            }
        }
        take(i);
        ++done;
        if (best >= 0) {
            take(best);
            ++done;
        }
        ++nc;
    }
    return {std::move(agg), nc};
}

struct Coarse
{
    CellGraph g;
    std::vector<double> w;
    AmgTransfer t;
};

// The coarse graph of agg (nc rows) with summed weights; the full transfer
// when `transfer` (else only g, w). Deterministic: members ascending, each
// coarse row's external slots sorted by (coarse column, fine slot).
Coarse coarsen(const CellGraph& g, std::span<const double> w, std::vector<int> agg, int nc, bool transfer)
{
    Coarse c;
    c.g.n = nc;
    AmgTransfer& t = c.t;
    t.memberPtr.assign(std::size_t(nc) + 1, 0);
    for (const int a : agg) ++t.memberPtr[std::size_t(a) + 1];
    std::partial_sum(t.memberPtr.begin(), t.memberPtr.end(), t.memberPtr.begin());
    t.member.resize(std::size_t(g.n));
    {
        std::vector<int> fill(t.memberPtr.begin(), t.memberPtr.end() - 1);
        for (int i = 0; i < g.n; ++i) t.member[std::size_t(fill[std::size_t(agg[std::size_t(i)])]++)] = i;
    }
    c.g.rowPtr.assign(std::size_t(nc) + 1, 0);
    c.g.col.reserve(g.col.size() / 2);
    c.w.reserve(g.col.size() / 2);
    if (transfer) {
        t.intraPtr.assign(std::size_t(nc) + 1, 0);
        t.offPtr.assign(1, 0);
        t.offPtr.reserve(g.col.size() / 2 + 1);
        t.offSlot.reserve(g.col.size());
    }
    std::vector<std::pair<int, int>> ext; // (coarse column, fine slot)
    for (int I = 0; I < nc; ++I) {
        ext.clear();
        for (int m = t.memberPtr[std::size_t(I)]; m < t.memberPtr[std::size_t(I) + 1]; ++m) {
            const int i = t.member[std::size_t(m)];
            for (int k = g.rowPtr[std::size_t(i)]; k < g.rowPtr[std::size_t(i) + 1]; ++k) {
                const int J = agg[std::size_t(g.col[std::size_t(k)])];
                if (J != I) ext.emplace_back(J, k);
                else if (transfer) t.intraSlot.push_back(k);
            }
        }
        std::ranges::sort(ext);
        for (std::size_t e = 0; e < ext.size();) {
            const int J = ext[e].first;
            double s = 0;
            for (; e < ext.size() && ext[e].first == J; ++e) {
                s += w[std::size_t(ext[e].second)];
                if (transfer) t.offSlot.push_back(ext[e].second);
            }
            c.g.col.push_back(J);
            c.w.push_back(s);
            if (transfer) t.offPtr.push_back(int(t.offSlot.size()));
        }
        c.g.rowPtr[std::size_t(I) + 1] = int(c.g.col.size());
        if (transfer) t.intraPtr[std::size_t(I) + 1] = int(t.intraSlot.size());
    }
    if (transfer) t.agg = std::move(agg);
    return c;
}

std::vector<std::uint8_t> coloursOf(const CellGraph& g)
{
    for (int c = 0; c < g.n; ++c)
        if (g.rowPtr[std::size_t(c) + 1] - g.rowPtr[std::size_t(c)] > 254) return {};
    return greedyColour(g);
}

std::expected<void, std::string> checkSetup(const AmgSetup& s)
{
    if (!(s.strongBeta > 0 && s.strongBeta <= 1)) return std::unexpected("amg: strongBeta outside (0, 1]");
    if (s.passes < 1 || s.passes > 4) return std::unexpected("amg: passes outside [1, 4]");
    if (s.coarseUnknowns < 1 || s.coarseUnknowns > kAmgMaxCoarseUnknowns)
        return std::unexpected("amg: coarseUnknowns outside [1, " + std::to_string(kAmgMaxCoarseUnknowns) + "]");
    if (s.maxLevels < 1) return std::unexpected("amg: maxLevels < 1");
    if (!(s.minRatio > 1)) return std::unexpected("amg: minRatio <= 1");
    if (!(s.lineRatio == 0 || s.lineRatio > 1)) return std::unexpected("amg: lineRatio neither 0 nor > 1");
    return {};
}

} // namespace

double AmgHierarchy::gridComplexity() const
{
    if (rows.empty() || rows[0] == 0) return 0;
    return double(std::accumulate(rows.begin(), rows.end(), 0.0)) / double(rows[0]);
}

double AmgHierarchy::operatorComplexity() const
{
    if (rows.empty()) return 0;
    double s = 0;
    for (std::size_t l = 0; l < rows.size(); ++l) s += double(rows[l]) + double(nnz[l]);
    const double f = double(rows[0]) + double(nnz[0]);
    return f > 0 ? s / f : 0;
}

std::expected<AmgHierarchy, std::string> amgHierarchy(const CellGraph& g, std::span<const double> slotWeight, int B,
                                                      const AmgSetup& s)
{
    if (auto ok = checkSetup(s); !ok) return std::unexpected(ok.error());
    if (B < 1) return std::unexpected("amg: block size < 1");
    if (slotWeight.size() != g.col.size())
        return std::unexpected("amg: " + std::to_string(slotWeight.size()) + " slot weights for " +
                               std::to_string(g.col.size()) + " slots");
    for (std::size_t k = 0; k < slotWeight.size(); ++k)
        if (!(slotWeight[k] >= 0) || !std::isfinite(slotWeight[k]))
            return std::unexpected("amg: slot weight " + std::to_string(k) + " negative or non-finite");
    AmgHierarchy h;
    h.B = B;
    auto addLevel = [&](const CellGraph& gl, std::span<const double> wl) {
        h.rows.push_back(gl.n);
        h.nnz.push_back(int(gl.col.size()));
        h.colours.push_back(coloursOf(gl));
        if (s.lineRatio > 1) h.lines.push_back(strongLines(gl, wl, s.lineRatio));
    };
    addLevel(g, slotWeight);
    const CellGraph* cur = &g;
    std::vector<double> curW(slotWeight.begin(), slotWeight.end());
    while (h.levels() < s.maxLevels && std::int64_t(cur->n) * B > s.coarseUnknowns) {
        auto [agg, nc] = pairwise(*cur, curW, s.strongBeta);
        for (int p = 1; p < s.passes; ++p) {
            const Coarse mid = coarsen(*cur, curW, agg, nc, false);
            auto [a2, n2] = pairwise(mid.g, mid.w, s.strongBeta);
            for (int& a : agg) a = a2[std::size_t(a)];
            nc = n2;
        }
        if (double(cur->n) < s.minRatio * double(nc)) break; // stalled: cur is the coarsest
        Coarse c = coarsen(*cur, curW, std::move(agg), nc, true);
        h.transfers.push_back(std::move(c.t));
        h.graphs.push_back(std::move(c.g));
        curW = std::move(c.w);
        cur = &h.graphs.back();
        addLevel(*cur, curW);
    }
    if (std::int64_t(h.rows.back()) * B > kAmgMaxCoarseUnknowns)
        return std::unexpected("amg: coarsening stalled at level " + std::to_string(h.levels() - 1) + " with " +
                               std::to_string(h.rows.back()) + " rows x B " + std::to_string(B) + " > " +
                               std::to_string(kAmgMaxCoarseUnknowns) +
                               " coarsest unknowns (rows without strong couplings?)");
    return h;
}

std::vector<double> amgStrength(const BlockMatrix& m)
{
    std::vector<double> w(m.nnz(), 0.0);
    const int B = m.shape.B;
    for (std::size_t k = 0; k < m.nnz(); ++k) {
        if (B == 1) {
            w[k] = std::max(-m.off[k], 0.0);
            continue;
        }
        double s = 0;
        for (int r = 0; r < B; ++r)
            for (int q = 0; q < B; ++q) {
                const double v = m.offAt(int(k), r, q);
                s += v * v;
            }
        w[k] = std::sqrt(s);
    }
    return w;
}

BlockMatrix amgGalerkin(const BlockMatrix& m, const CellGraph& coarse, const AmgTransfer& t)
{
    BlockMatrix c = BlockMatrix::zeros(coarse, m.shape);
    const int B = m.shape.B, Bd = m.shape.Bd, nS = m.shape.nS;
    const std::size_t BB = std::size_t(B) * std::size_t(B);
    for (int I = 0; I < coarse.n; ++I)
        for (int r = 0; r < B; ++r)
            for (int q = 0; q < B; ++q) {
                double s = 0;
                for (int k = t.memberPtr[std::size_t(I)]; k < t.memberPtr[std::size_t(I) + 1]; ++k)
                    s += m.diagAt(t.member[std::size_t(k)], r, q);
                for (int k = t.intraPtr[std::size_t(I)]; k < t.intraPtr[std::size_t(I) + 1]; ++k)
                    s += m.offAt(t.intraSlot[std::size_t(k)], r, q);
                c.diag[std::size_t(I) * BB + std::size_t(r * B + q)] = s;
            }
    for (std::size_t sl = 0; sl < coarse.col.size(); ++sl) {
        for (int e = 0; e < Bd * Bd; ++e) {
            double s = 0;
            for (int k = t.offPtr[sl]; k < t.offPtr[sl + 1]; ++k)
                s += m.off[std::size_t(t.offSlot[std::size_t(k)]) * std::size_t(Bd * Bd) + std::size_t(e)];
            c.off[sl * std::size_t(Bd * Bd) + std::size_t(e)] = s;
        }
        for (int e = 0; e < nS; ++e) {
            double s = 0;
            for (int k = t.offPtr[sl]; k < t.offPtr[sl + 1]; ++k)
                s += m.sc[std::size_t(t.offSlot[std::size_t(k)]) * std::size_t(nS) + std::size_t(e)];
            c.sc[sl * std::size_t(nS) + std::size_t(e)] = s;
        }
    }
    return c;
}

} // namespace dns::la
