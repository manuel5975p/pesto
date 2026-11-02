#include "dns/fvraster.hpp"

#include "dns/fvgrad.hpp"
#include "dns/parallel.hpp"

#include <algorithm>
#include <cmath>
#include <format>
#include <limits>

namespace dns {
namespace {

using V3 = std::array<double, 3>;

double dot(const V3& a, const V3& b) { return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]; }

// Nearest-centroid bucket grid: the walk's first guess only.
struct Buckets
{
    std::array<double, 3> lo{}, inv{};
    std::array<int, 3> n{};
    std::vector<int> start, item;

    int axis(double x, int a) const
    {
        return std::clamp(int((x - lo[std::size_t(a)]) * inv[std::size_t(a)]), 0, n[std::size_t(a)] - 1);
    }
    std::size_t idx(int i, int j, int k) const
    {
        return std::size_t(i) + std::size_t(n[0]) * (std::size_t(j) + std::size_t(n[1]) * std::size_t(k));
    }
};

Buckets bucketsOf(const FaceMesh& fm)
{
    Buckets b;
    const int side = std::clamp(int(std::cbrt(double(fm.nC))), 1, 256);
    for (std::size_t a = 0; a < 3; ++a) {
        const auto [mn, mx] = std::ranges::minmax(fm.C[a]);
        b.lo[a] = mn;
        const bool flat = !(mx > mn);
        b.n[a] = flat ? 1 : side;
        b.inv[a] = flat ? 0.0 : double(side) / (mx - mn);
    }
    const std::size_t total = std::size_t(b.n[0]) * std::size_t(b.n[1]) * std::size_t(b.n[2]);
    b.start.assign(total + 1, 0);
    auto at = [&](std::size_t c) {
        return b.idx(b.axis(fm.C[0][c], 0), b.axis(fm.C[1][c], 1), b.axis(fm.C[2][c], 2));
    };
    for (std::size_t c = 0; c < std::size_t(fm.nC); ++c) ++b.start[at(c) + 1];
    for (std::size_t i = 0; i < total; ++i) b.start[i + 1] += b.start[i];
    b.item.resize(std::size_t(fm.nC));
    std::vector<int> pos(b.start.begin(), b.start.end() - 1);
    for (std::size_t c = 0; c < std::size_t(fm.nC); ++c) b.item[std::size_t(pos[at(c)]++)] = int(c);
    return b;
}

int nearest(const FaceMesh& fm, const Buckets& b, const V3& p)
{
    const int ci = b.axis(p[0], 0), cj = b.axis(p[1], 1), ck = b.axis(p[2], 2);
    int best = -1;
    double bd = std::numeric_limits<double>::infinity();
    const int maxR = std::max({b.n[0], b.n[1], b.n[2]});
    for (int r = 0; r <= maxR; ++r) {
        for (int k = std::max(0, ck - r); k <= std::min(b.n[2] - 1, ck + r); ++k)
            for (int j = std::max(0, cj - r); j <= std::min(b.n[1] - 1, cj + r); ++j)
                for (int i = std::max(0, ci - r); i <= std::min(b.n[0] - 1, ci + r); ++i) {
                    if (std::max({std::abs(i - ci), std::abs(j - cj), std::abs(k - ck)}) != r) continue;
                    const std::size_t s = b.idx(i, j, k);
                    for (int e = b.start[s]; e < b.start[s + 1]; ++e) {
                        const std::size_t c = std::size_t(b.item[std::size_t(e)]);
                        const V3 d = {fm.C[0][c] - p[0], fm.C[1][c] - p[1], fm.C[2][c] - p[2]};
                        const double q = dot(d, d);
                        if (q < bd || (q == bd && int(c) < best)) {
                            bd = q;
                            best = int(c);
                        }
                    }
                }
        // A hit in shell r bounds the answer within shell r + 1: bucket widths vary
        // little enough for a first guess, the walk corrects the rest.
        if (best >= 0 && r >= 1) break;
    }
    return best;
}

// Walk from c toward p across the most violated face. Returns the containing
// cell, -1 when the walk leaves the mesh through a boundary face.
int locate(const FaceMesh& fm, const FvFaceGeom& fg, int c, const V3& p)
{
    const std::size_t nI = std::size_t(fm.nI);
    for (int steps = 0; steps < 4096; ++steps) {
        const std::size_t cc = std::size_t(c);
        const V3 rel = {p[0] - fm.C[0][cc], p[1] - fm.C[1][cc], p[2] - fm.C[2][cc]};
        double worst = 0.0;
        int wf = -1;
        for (int q = fm.cellStart[cc]; q < fm.cellStart[cc + 1]; ++q) {
            const std::size_t f = std::size_t(fm.cellFace[std::size_t(q)]);
            if (!fg.active[f]) continue;
            const bool own = std::size_t(fm.own[f]) == cc;
            const double sg = own ? 1.0 : -1.0;
            const V3 S = {sg * fm.S[0][f], sg * fm.S[1][f], sg * fm.S[2][f]};
            V3 r = {fg.rOwn[0][f], fg.rOwn[1][f], fg.rOwn[2][f]};
            if (!own) r = {r[0] - fm.d[0][f], r[1] - fm.d[1][f], r[2] - fm.d[2][f]};
            const V3 x = {rel[0] - r[0], rel[1] - r[1], rel[2] - r[2]};
            const double h = dot(x, S) / std::sqrt(dot(S, S));
            if (h > worst) {
                worst = h;
                wf = int(f);
            }
        }
        if (wf < 0) return c;
        const std::size_t f = std::size_t(wf);
        if (f >= nI) return -1;
        // A cyclic face: the mesh wraps here, the point lies past it.
        const std::size_t o = std::size_t(fm.own[f]), n = std::size_t(fm.nei[f]);
        const V3 T = {fm.C[0][o] + fm.d[0][f] - fm.C[0][n], fm.C[1][o] + fm.d[1][f] - fm.C[1][n],
                      fm.C[2][o] + fm.d[2][f] - fm.C[2][n]};
        const V3 dv = {fm.d[0][f], fm.d[1][f], fm.d[2][f]};
        if (dot(T, T) > 1e-16 * dot(dv, dv)) return -1;
        c = int(o) == c ? int(n) : int(o);
    }
    return -1;
}

} // namespace

std::expected<FvRaster, std::string> FvRaster::make(const FaceMesh& fm, const FvRasterSpec& sp, int threads)
{
    auto fg = fvFaceGeom(fm, FvFaceGradRule::AlongD, 1.0, threads);
    if (!fg) return std::unexpected(std::move(fg.error()));
    const int nAxis = fm.emptyAxis >= 0 ? fm.emptyAxis : 2;
    const int a0 = nAxis == 0 ? 1 : 0, a1 = nAxis == 2 ? 1 : 2;
    const auto [lo0, hi0] = std::ranges::minmax(fm.C[std::size_t(a0)]);
    const auto [lo1, hi1] = std::ranges::minmax(fm.C[std::size_t(a1)]);
    const auto [loN, hiN] = std::ranges::minmax(fm.C[std::size_t(nAxis)]);
    const bool whole = std::ranges::all_of(sp.box, [](double v) { return v == 0.0; });
    const std::array<double, 4> b = whole ? std::array<double, 4>{lo0, hi0, lo1, hi1} : sp.box;
    if (!(b[1] > b[0]) || !(b[3] > b[2])) return std::unexpected("--frame-box: an empty or inverted window");
    const double w = b[1] - b[0], h = b[3] - b[2], dx = std::max(w, h) / sp.longSide;
    const int nx = std::max(4, int(std::lround(w / dx))), ny = std::max(4, int(std::lround(h / dx)));
    FvRaster r;
    r.g_ = Grid::make(nx, ny, 1, w, h, dx);
    const std::size_t n = std::size_t(nx) * std::size_t(ny);
    r.cell_.assign(n, -1);
    r.mask_.assign(n, 0.0f);
    const Buckets bk = bucketsOf(fm);
    const double zc = 0.5 * (loN + hiN);
    const double hx = w / nx, hy = h / ny;
    forBlocks(workerCount(threads, std::size_t(ny)), std::size_t(ny), [&](int, std::size_t lo, std::size_t hi) {
        for (std::size_t j = lo; j < hi; ++j)
            for (std::size_t i = 0; i < std::size_t(nx); ++i) {
                V3 p{};
                p[std::size_t(a0)] = b[0] + (double(i) + 0.5) * hx;
                p[std::size_t(a1)] = b[2] + (double(j) + 0.5) * hy;
                p[std::size_t(nAxis)] = zc;
                const int c0 = nearest(fm, bk, p);
                const int c = c0 < 0 ? -1 : locate(fm, *fg, c0, p);
                const std::size_t q = j * std::size_t(nx) + i;
                r.cell_[q] = c;
                r.mask_[q] = c >= 0 ? 1.0f : 0.0f;
            }
    });
    return r;
}

void FvRaster::sample(std::span<const double> cellValue, std::span<float> out, float fill) const
{
    for (std::size_t q = 0; q < cell_.size(); ++q)
        out[q] = cell_[q] >= 0 ? float(cellValue[std::size_t(cell_[q])]) : fill;
}

} // namespace dns
