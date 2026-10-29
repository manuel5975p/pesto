// Host FP64 oracles of the solver library (gpu/la): random block matrices on a
// CellGraph, rounding to the device value type, and the dense restatements of
// SpMV, block-Jacobi, multicolour GS and the line solve that test_la and
// test_gpu_la compare against.
#pragma once

#include "dns/blockband.hpp"
#include "dns/facemesh.hpp"
#include "dns/la/ell.hpp"
#include "dns/la/lines.hpp"

#include <algorithm>
#include <array>
#include <span>
#include <string>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

namespace latest {

using dns::la::BlockMatrix;
using dns::la::BlockShape;
using dns::la::CellGraph;
using dns::la::LineSet;

// Random entries in [-1, 1], diagonal blocks made block-diagonally dominant:
// D_rr += dominance (1 + sum of |row r| over the diagonal block and every
// off-diagonal block of the row), so every smoother converges.
inline BlockMatrix randomMatrix(const CellGraph& g, BlockShape s, unsigned seed, double dominance = 1.0)
{
    BlockMatrix m = BlockMatrix::zeros(g, s);
    std::mt19937_64 rng(seed);
    std::uniform_real_distribution<double> u(-1.0, 1.0);
    for (double& v : m.diag) v = u(rng);
    for (double& v : m.off) v = u(rng);
    for (double& v : m.sc) v = u(rng);
    const int B = s.B;
    for (int c = 0; c < g.n; ++c)
        for (int r = 0; r < B; ++r) {
            double sum = 0;
            for (int q = 0; q < B; ++q) sum += std::abs(m.diagAt(c, r, q));
            for (int k = g.rowPtr[std::size_t(c)]; k < g.rowPtr[std::size_t(c) + 1]; ++k)
                for (int q = 0; q < B; ++q) sum += std::abs(m.offAt(k, r, q));
            m.diag[(std::size_t(c) * std::size_t(B) + std::size_t(r)) * std::size_t(B) + std::size_t(r)] +=
                dominance * (1.0 + sum);
        }
    return m;
}

template <class T>
BlockMatrix roundTo(BlockMatrix m)
{
    for (auto* v : {&m.diag, &m.off, &m.sc})
        for (double& x : *v) x = double(T(x));
    return m;
}

template <class T>
std::vector<double> roundVec(std::vector<double> v)
{
    for (double& x : v) x = double(T(x));
    return v;
}

inline std::vector<double> randomVec(std::size_t n, unsigned seed)
{
    std::mt19937_64 rng(seed);
    std::uniform_real_distribution<double> u(-1.0, 1.0);
    std::vector<double> v(n);
    for (double& x : v) x = u(rng);
    return v;
}

// Vectors are component-major v[q n + c].
inline double at(const std::vector<double>& v, int n, int c, int q) { return v[std::size_t(q) * std::size_t(n) + std::size_t(c)]; }

// y = A x and the row scale sum |A| |x| per (c, r).
inline void spmv(const BlockMatrix& m, const std::vector<double>& x, std::vector<double>& y,
                 std::vector<double>* scale = nullptr)
{
    const int n = m.n, B = m.shape.B;
    y.assign(std::size_t(n) * std::size_t(B), 0.0);
    if (scale) scale->assign(y.size(), 0.0);
    for (int c = 0; c < n; ++c)
        for (int r = 0; r < B; ++r) {
            double s = 0, a = 0;
            for (int q = 0; q < B; ++q) {
                s += m.diagAt(c, r, q) * at(x, n, c, q);
                a += std::abs(m.diagAt(c, r, q) * at(x, n, c, q));
            }
            for (int k = m.rowPtr[std::size_t(c)]; k < m.rowPtr[std::size_t(c) + 1]; ++k)
                for (int q = 0; q < B; ++q) {
                    const double t = m.offAt(k, r, q) * at(x, n, m.col[std::size_t(k)], q);
                    s += t;
                    a += std::abs(t);
                }
            y[std::size_t(r) * std::size_t(n) + std::size_t(c)] = s;
            if (scale) (*scale)[std::size_t(r) * std::size_t(n) + std::size_t(c)] = a;
        }
}

// Dense A x = b in place by Gaussian elimination with partial pivoting
// (A row-major N x N). False on a zero pivot.
inline bool denseSolve(std::vector<double> A, std::vector<double>& b)
{
    const std::size_t N = b.size();
    for (std::size_t k = 0; k < N; ++k) {
        std::size_t p = k;
        for (std::size_t r = k + 1; r < N; ++r)
            if (std::abs(A[r * N + k]) > std::abs(A[p * N + k])) p = r;
        if (A[p * N + k] == 0) return false;
        if (p != k) {
            for (std::size_t q = 0; q < N; ++q) std::swap(A[k * N + q], A[p * N + q]);
            std::swap(b[k], b[p]);
        }
        for (std::size_t r = k + 1; r < N; ++r) {
            const double f = A[r * N + k] / A[k * N + k];
            if (f == 0) continue;
            for (std::size_t q = k; q < N; ++q) A[r * N + q] -= f * A[k * N + q];
            b[r] -= f * b[k];
        }
    }
    for (std::size_t k = N; k-- > 0;) {
        double s = b[k];
        for (std::size_t q = k + 1; q < N; ++q) s -= A[k * N + q] * b[q];
        b[k] = s / A[k * N + k];
    }
    return true;
}

inline std::vector<double> diagDense(const BlockMatrix& m, int c)
{
    const int B = m.shape.B;
    std::vector<double> D(std::size_t(B * B));
    for (int r = 0; r < B; ++r)
        for (int q = 0; q < B; ++q) D[std::size_t(r * B + q)] = m.diagAt(c, r, q);
    return D;
}

// |D^-1| (entrywise) of row c, row-major.
inline std::vector<double> absInverse(const BlockMatrix& m, int c)
{
    const int B = m.shape.B;
    std::vector<double> X(std::size_t(B * B));
    for (int q = 0; q < B; ++q) {
        std::vector<double> e(std::size_t(B), 0.0);
        e[std::size_t(q)] = 1;
        denseSolve(diagDense(m, c), e);
        for (int r = 0; r < B; ++r) X[std::size_t(r * B + q)] = std::abs(e[std::size_t(r)]);
    }
    return X;
}

// z = D^-1 r and its scale |D^-1| |r|.
inline void jacobi(const BlockMatrix& m, const std::vector<double>& r, std::vector<double>& z,
                   std::vector<double>& scale)
{
    const int n = m.n, B = m.shape.B;
    z.assign(r.size(), 0.0);
    scale.assign(r.size(), 0.0);
    for (int c = 0; c < n; ++c) {
        std::vector<double> rc(static_cast<std::size_t>(B));
        for (int q = 0; q < B; ++q) rc[std::size_t(q)] = at(r, n, c, q);
        const std::vector<double> X = absInverse(m, c);
        for (int p = 0; p < B; ++p) {
            double s = 0;
            for (int q = 0; q < B; ++q) s += X[std::size_t(p * B + q)] * std::abs(rc[std::size_t(q)]);
            scale[std::size_t(p) * std::size_t(n) + std::size_t(c)] = s;
        }
        denseSolve(diagDense(m, c), rc);
        for (int q = 0; q < B; ++q) z[std::size_t(q) * std::size_t(n) + std::size_t(c)] = rc[std::size_t(q)];
    }
}

// One multicolour GS sweep (colours ascending or descending, rows ascending
// within a colour), x updated in place; scale gets |D^-1| (|b| + sum |A||x|)
// at each row's update.
inline void mcgsSweep(const BlockMatrix& m, const std::vector<std::uint8_t>& colour, const std::vector<double>& b,
                      std::vector<double>& x, bool forward, std::vector<double>& scale)
{
    const int n = m.n, B = m.shape.B;
    scale.resize(x.size());
    const int K = n == 0 ? 0 : int(*std::max_element(colour.begin(), colour.end())) + 1;
    for (int t = 0; t < K; ++t) {
        const int k = forward ? t : K - 1 - t;
        for (int c = 0; c < n; ++c) {
            if (colour[std::size_t(c)] != k) continue;
            std::vector<double> acc(static_cast<std::size_t>(B)), aa(static_cast<std::size_t>(B));
            for (int r = 0; r < B; ++r) {
                acc[std::size_t(r)] = at(b, n, c, r);
                aa[std::size_t(r)] = std::abs(acc[std::size_t(r)]);
                for (int s = m.rowPtr[std::size_t(c)]; s < m.rowPtr[std::size_t(c) + 1]; ++s)
                    for (int q = 0; q < B; ++q) {
                        const double v = m.offAt(s, r, q) * at(x, n, m.col[std::size_t(s)], q);
                        acc[std::size_t(r)] -= v;
                        aa[std::size_t(r)] += std::abs(v);
                    }
            }
            const std::vector<double> X = absInverse(m, c);
            for (int p = 0; p < B; ++p) {
                double s = 0;
                for (int q = 0; q < B; ++q) s += X[std::size_t(p * B + q)] * aa[std::size_t(q)];
                scale[std::size_t(p) * std::size_t(n) + std::size_t(c)] = s;
            }
            denseSolve(diagDense(m, c), acc);
            for (int q = 0; q < B; ++q) x[std::size_t(q) * std::size_t(n) + std::size_t(c)] = acc[std::size_t(q)];
        }
    }
}

// The slot of column j in row c of m, -1 when absent.
inline int slotOf(const BlockMatrix& m, int c, int j)
{
    for (int k = m.rowPtr[std::size_t(c)]; k < m.rowPtr[std::size_t(c) + 1]; ++k)
        if (m.col[std::size_t(k)] == j) return k;
    return -1;
}

// z = M^-1 r for the line block-tridiagonal M: dns::solveBlockBand (K = 1)
// when B <= 7 (useBand), else a dense solve of the line's (L B)^2 system.
// lineMax gets max |z| over each line's cells, per cell.
inline void lineSolve(const BlockMatrix& m, const LineSet& ls, const std::vector<double>& r, std::vector<double>& z,
                      std::vector<double>& lineMax, bool useBand)
{
    const int n = m.n, B = m.shape.B;
    z.assign(r.size(), 0.0);
    lineMax.assign(r.size(), 0.0);
    for (int l = 0; l < ls.count(); ++l) {
        const int* cs = ls.cells.data() + ls.start[std::size_t(l)];
        const int L = ls.start[std::size_t(l) + 1] - ls.start[std::size_t(l)];
        std::vector<double> rhs(std::size_t(L * B));
        for (int p = 0; p < L; ++p)
            for (int q = 0; q < B; ++q) rhs[std::size_t(p * B + q)] = at(r, n, cs[p], q);
        auto block = [&](int p, int off, int rr, int qq) {
            if (off == 0) return m.diagAt(cs[p], rr, qq);
            const int s = slotOf(m, cs[p], cs[p + off]);
            return m.offAt(s, rr, qq);
        };
        if (useBand) {
            dns::BlockBand A = dns::BlockBand::make(L, B, 1);
            for (int p = 0; p < L; ++p)
                for (int off = -1; off <= 1; ++off) {
                    if (p + off < 0 || p + off >= L) continue;
                    for (int rr = 0; rr < B; ++rr)
                        for (int qq = 0; qq < B; ++qq) A.at(p, off, rr, qq) = block(p, off, rr, qq);
                }
            if (!dns::solveBlockBand(std::move(A), rhs)) std::abort();
        } else {
            const std::size_t N = std::size_t(L * B);
            std::vector<double> A(N * N, 0.0);
            for (int p = 0; p < L; ++p)
                for (int off = -1; off <= 1; ++off) {
                    if (p + off < 0 || p + off >= L) continue;
                    for (int rr = 0; rr < B; ++rr)
                        for (int qq = 0; qq < B; ++qq)
                            A[std::size_t(p * B + rr) * N + std::size_t((p + off) * B + qq)] = block(p, off, rr, qq);
                }
            if (!denseSolve(std::move(A), rhs)) std::abort();
        }
        double mx = 0;
        for (const double v : rhs) mx = std::max(mx, std::abs(v));
        for (int p = 0; p < L; ++p)
            for (int q = 0; q < B; ++q) {
                z[std::size_t(q) * std::size_t(n) + std::size_t(cs[p])] = rhs[std::size_t(p * B + q)];
                lineMax[std::size_t(q) * std::size_t(n) + std::size_t(cs[p])] = mx;
            }
    }
}

// max_i |a_i - b_i| / scale_i (scale 0 entries: absolute).
inline double maxRel(const std::vector<double>& a, const std::vector<double>& b, const std::vector<double>& scale)
{
    double e = 0;
    for (std::size_t i = 0; i < a.size(); ++i) {
        const double d = std::abs(a[i] - b[i]);
        e = std::max(e, scale[i] > 0 ? d / scale[i] : d);
    }
    return e;
}


// -div grad (SPD) on an n[0] x n[1] x n[2] tensor grid (Grid::idx order, the
// structuredGraph pattern) with per-axis cell widths h[a][i]: face coefficient
// area / centre distance, Dirichlet 0 on every non-degenerate box face (half-
// cell distance). n[2] == 1: 2D (no z faces).
inline BlockMatrix tensorLaplacian(const CellGraph& g, std::array<int, 3> n, const std::array<std::vector<double>, 3>& h)
{
    BlockMatrix m = BlockMatrix::zeros(g, *dns::la::denseShape(1));
    const int dims = n[2] == 1 ? 2 : 3;
    const int stride[3] = {1, n[0], n[0] * n[1]};
    for (int c = 0; c < g.n; ++c) {
        const int ijk[3] = {c % n[0], (c / n[0]) % n[1], c / (n[0] * n[1])};
        double d = 0;
        for (int ax = 0; ax < dims; ++ax) {
            const int a1 = (ax + 1) % 3, a2 = (ax + 2) % 3;
            const double area = (a1 < dims ? h[std::size_t(a1)][std::size_t(ijk[a1])] : 1.0) *
                                (a2 < dims ? h[std::size_t(a2)][std::size_t(ijk[a2])] : 1.0);
            const double hc = h[std::size_t(ax)][std::size_t(ijk[ax])];
            for (const int s : {-1, 1}) {
                const int m2 = ijk[ax] + s;
                if (m2 < 0 || m2 >= n[std::size_t(ax)]) {
                    d += area / (0.5 * hc);
                    continue;
                }
                const double coef = area / (0.5 * (hc + h[std::size_t(ax)][std::size_t(m2)]));
                const int j = c + s * stride[ax];
                m.off[std::size_t(slotOf(m, c, j))] = -coef;
                d += coef;
            }
        }
        m.diag[std::size_t(c)] = d;
    }
    return m;
}

// OpenFOAM v2606's `Gauss linear corrected` Laplacian matrix with gamma = 1
// (the implicit part: nonOrthDeltaCoeffs), sign -laplacian (SPD): internal
// faces |S| / max(n.d, 0.05 |d|); a patch named in `dirichlet` (fixedValue 0)
// adds |S| / (n.d_b) to its cell's diagonal; every other patch is zero-gradient
// (empty patches carry nothing). g = graphOf(fm).
inline BlockMatrix foamLaplacian(const dns::FaceMesh& fm, const CellGraph& g, std::span<const std::string> dirichlet)
{
    BlockMatrix m = BlockMatrix::zeros(g, *dns::la::denseShape(1));
    auto nd = [&](int f, double& magS, double& magD) {
        magS = std::sqrt(fm.S[0][std::size_t(f)] * fm.S[0][std::size_t(f)] + fm.S[1][std::size_t(f)] * fm.S[1][std::size_t(f)] +
                         fm.S[2][std::size_t(f)] * fm.S[2][std::size_t(f)]);
        magD = std::sqrt(fm.d[0][std::size_t(f)] * fm.d[0][std::size_t(f)] + fm.d[1][std::size_t(f)] * fm.d[1][std::size_t(f)] +
                         fm.d[2][std::size_t(f)] * fm.d[2][std::size_t(f)]);
        double s = 0;
        for (int a = 0; a < 3; ++a) s += fm.S[std::size_t(a)][std::size_t(f)] / magS * fm.d[std::size_t(a)][std::size_t(f)];
        return s;
    };
    for (int f = 0; f < fm.nI; ++f) {
        double magS = 0, magD = 0;
        const double nDotD = nd(f, magS, magD);
        const double coef = magS / std::max(nDotD, 0.05 * magD);
        const int o = fm.own[std::size_t(f)], nb = fm.nei[std::size_t(f)];
        m.diag[std::size_t(o)] += coef;
        m.diag[std::size_t(nb)] += coef;
        m.off[std::size_t(g.ownSlot[std::size_t(f)])] -= coef;
        m.off[std::size_t(g.neiSlot[std::size_t(f)])] -= coef;
    }
    for (const auto& p : fm.patches) {
        if (std::ranges::find(dirichlet, p.name) == dirichlet.end()) continue;
        for (int f = fm.nI + p.first; f < fm.nI + p.first + p.count; ++f) {
            double magS = 0, magD = 0;
            const double nDotD = nd(f, magS, magD);
            // fvPatch::delta() is n (n . (Cf - C)): |delta| = |n.d|
            m.diag[std::size_t(fm.own[std::size_t(f)])] += magS / std::max(nDotD, 0.05 * std::abs(nDotD));
        }
    }
    return m;
}

} // namespace latest
