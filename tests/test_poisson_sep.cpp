// Separable Poisson, host FP64 (masterplan real_bunsen S3, dns/septransform.hpp):
//  (a) every (kind, n) basis vector is an EXACT eigenvector of the 1D operator
//      with sepEigen's eigenvalue, and the basis is orthogonal;
//  (b) the fast algorithms (Makhoul DCT-II, DST-II via the reversed DCT-II,
//      DCT-IV via one n-point FFT, DHT) equal the direct sums and invert to
//      x / sepNorm -- to 1e-14 of sum|x|, odd and even n;
//  (c) the 3D separable solve equals a dense LU solve of the 7-point operator
//      for EVERY kind triple on 6x8x10, 5x7x9, 7x6x5 and the 2D (nz == 1)
//      restriction, singular triples on the zero-mean solution.
#include "check.hpp"
#include "sep_test_util.hpp"

#include <algorithm>
#include <cstdio>
#include <random>
#include <string>

using namespace septest;
using dnstest::check;

namespace {

std::string kname(SepKind k) { return dns::sepKindName(k); }

// (a) exact eigenpairs of the 1D operator.
void eigenpairs()
{
    const int sizes[] = {2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 16, 17};
    for (SepKind kind : kAllKinds)
    {
        double worstRes = 0, worstOrth = 0;
        for (int n : sizes)
        {
            const double h = 0.7;
            Box b{{n, 1, 1}, {h, 1, 1}, {kind, SepKind::Periodic, SepKind::Periodic}};
            const double lamMax = 4.0 / (h * h);
            for (int k = 0; k < n; ++k)
            {
                std::vector<double> v(static_cast<std::size_t>(n));
                for (int j = 0; j < n; ++j) v[std::size_t(j)] = dns::sepBasis(kind, n, k, j);
                const auto Lv = laplacian(b, v);
                const double lam = dns::sepEigen(kind, n, h, k);
                double r = 0, vmax = 0;
                for (int j = 0; j < n; ++j)
                {
                    r = std::max(r, std::abs(-Lv[std::size_t(j)] - lam * v[std::size_t(j)]));
                    vmax = std::max(vmax, std::abs(v[std::size_t(j)]));
                }
                worstRes = std::max(worstRes, r / (lamMax * vmax));
                for (int k2 = 0; k2 < k; ++k2)
                {
                    double s = 0;
                    for (int j = 0; j < n; ++j)
                        s += v[std::size_t(j)] * dns::sepBasis(kind, n, k2, j);
                    worstOrth = std::max(worstOrth, std::abs(s) / n);
                }
            }
        }
        std::printf("  %-8s eigen residual %.2e (of lamMax |v|)  orthogonality %.2e\n",
                    kname(kind).c_str(), worstRes, worstOrth);
        check(worstRes <= 1e-14, kname(kind) + ": basis vectors are eigenvectors, eigenvalue sepEigen");
        check(worstOrth <= 1e-14, kname(kind) + ": basis orthogonal");
    }
}

// (b) fast algorithms vs direct sums.
void fastVsDirect()
{
    std::mt19937_64 rng(7);
    std::uniform_real_distribution<double> u(-1, 1);
    const int sizes[] = {2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 16, 17};
    for (SepKind kind : kAllKinds)
    {
        double eFwd = 0, eInv = 0, ePair = 0;
        for (int n : sizes)
            for (int rep = 0; rep < 3; ++rep)
            {
                if (kind == SepKind::ND && n % 2) continue; // the n/2-point DCT-IV
                std::vector<double> x(static_cast<std::size_t>(n));
                double l1 = 0;
                for (auto& v : x)
                {
                    v = u(rng);
                    l1 += std::abs(v);
                }
                const auto c = fwdFast(kind, x);
                for (int k = 0; k < n; ++k)
                {
                    double s = 0;
                    for (int j = 0; j < n; ++j) s += x[std::size_t(j)] * dns::sepBasis(kind, n, k, j);
                    eFwd = std::max(eFwd, std::abs(c[std::size_t(k)] - s) / l1);
                }
                const auto y = invFast(kind, c);
                const double nrm = dns::sepNorm(kind, n);
                for (int j = 0; j < n; ++j)
                    eInv = std::max(eInv, std::abs(y[std::size_t(j)] * nrm - x[std::size_t(j)]) / l1);
                // Two lines through one FFT (sepPack) == the plain passes.
                std::vector<double> x2(x.size());
                for (auto& v : x2) v = u(rng);
                for (bool inv : {false, true})
                {
                    const auto pr = pairFast(kind, inv, x, x2);
                    const auto s1 = inv ? invFast(kind, x) : fwdFast(kind, x);
                    const auto s2 = inv ? invFast(kind, x2) : fwdFast(kind, x2);
                    double l12 = 0;
                    for (int j = 0; j < n; ++j) l12 += std::abs(x[std::size_t(j)]) + std::abs(x2[std::size_t(j)]);
                    for (int j = 0; j < n; ++j)
                        ePair = std::max({ePair, std::abs(pr[0][std::size_t(j)] - s1[std::size_t(j)]) / l12,
                                          std::abs(pr[1][std::size_t(j)] - s2[std::size_t(j)]) / l12});
                }
            }
        std::printf("  %-8s fast fwd vs direct sum %.2e   inv(fwd(x)) %.2e   paired vs plain %.2e (of sum|x|)\n",
                    kname(kind).c_str(), eFwd, eInv, ePair);
        check(eFwd <= 1e-14, kname(kind) + ": fast forward == direct sum to 1e-14");
        check(eInv <= 1e-14, kname(kind) + ": inverse(forward(x)) == x / sepNorm to 1e-14");
        check(ePair <= 1e-14, kname(kind) + ": two lines per FFT (sepPack) == plain passes to 1e-14");
    }
}

// Dense LU with partial pivoting; A row-major n x n, solved in place into b.
bool luSolve(std::vector<double> A, std::vector<double>& b, int n)
{
    for (int c = 0; c < n; ++c)
    {
        int piv = c;
        for (int r = c + 1; r < n; ++r)
            if (std::abs(A[std::size_t(r) * n + c]) > std::abs(A[std::size_t(piv) * n + c])) piv = r;
        if (A[std::size_t(piv) * n + c] == 0.0) return false;
        if (piv != c)
        {
            for (int q = 0; q < n; ++q) std::swap(A[std::size_t(c) * n + q], A[std::size_t(piv) * n + q]);
            std::swap(b[std::size_t(c)], b[std::size_t(piv)]);
        }
        for (int r = c + 1; r < n; ++r)
        {
            const double f = A[std::size_t(r) * n + c] / A[std::size_t(c) * n + c];
            if (f == 0.0) continue;
            for (int q = c; q < n; ++q) A[std::size_t(r) * n + q] -= f * A[std::size_t(c) * n + q];
            b[std::size_t(r)] -= f * b[std::size_t(c)];
        }
    }
    for (int r = n - 1; r >= 0; --r)
    {
        double s = b[std::size_t(r)];
        for (int q = r + 1; q < n; ++q) s -= A[std::size_t(r) * n + q] * b[std::size_t(q)];
        b[std::size_t(r)] = s / A[std::size_t(r) * n + r];
    }
    return true;
}

// (c) separable solve vs dense LU, every kind triple.
void denseCompare(int nx, int ny, int nz)
{
    std::mt19937_64 rng(std::uint64_t(nx * 10000 + ny * 100 + nz));
    std::uniform_real_distribution<double> u(-1, 1);
    const double h[3] = {0.31, 0.27, 0.43};
    double worst = 0, worstRes = 0;
    int cases = 0;
    for (SepKind kx : kAllKinds)
        for (SepKind ky : kAllKinds)
            for (SepKind kz : kAllKinds)
            {
                if (nz == 1 && kz != SepKind::Periodic) continue; // 2D: z has no stencil
                const SepKind kk[3] = {kx, ky, kz};
                const int nn[3] = {nx, ny, nz};
                bool oddNd = false;
                for (int a = 0; a < 3; ++a) oddNd |= kk[a] == SepKind::ND && nn[a] % 2;
                if (oddNd) continue; // ND needs an even axis (the n/2-point DCT-IV)
                Box b{{nx, ny, nz}, {h[0], h[1], h[2]}, {kx, ky, kz}};
                const int N = int(b.size());
                std::vector<double> f(static_cast<std::size_t>(N));
                for (auto& v : f) v = u(rng);
                const bool sing = b.singular();
                if (sing)
                {
                    double m = 0;
                    for (double v : f) m += v;
                    m /= N;
                    for (auto& v : f) v -= m;
                }
                // Dense matrix of L by columns (+ 11^T/N on the null space if singular).
                std::vector<double> A(std::size_t(N) * N);
                std::vector<double> e(std::size_t(N), 0.0);
                for (int c = 0; c < N; ++c)
                {
                    e[std::size_t(c)] = 1.0;
                    const auto col = laplacian(b, e);
                    e[std::size_t(c)] = 0.0;
                    for (int r = 0; r < N; ++r)
                        A[std::size_t(r) * N + c] = col[std::size_t(r)] + (sing ? 1.0 / N : 0.0);
                }
                std::vector<double> pd = f;
                const bool ok = luSolve(A, pd, N);
                const auto ps = sepSolve(b, f);
                double err = 0, scale = 0, mean = 0;
                for (int c = 0; c < N; ++c)
                {
                    err = std::max(err, std::abs(ps[std::size_t(c)] - pd[std::size_t(c)]));
                    scale = std::max(scale, std::abs(pd[std::size_t(c)]));
                    mean += ps[std::size_t(c)];
                }
                const auto Lp = laplacian(b, ps);
                double res = 0, fs = 0;
                for (int c = 0; c < N; ++c)
                {
                    res = std::max(res, std::abs(Lp[std::size_t(c)] - f[std::size_t(c)]));
                    fs = std::max(fs, std::abs(f[std::size_t(c)]));
                }
                const double rel = err / scale;
                worst = std::max(worst, rel);
                worstRes = std::max(worstRes, res / fs);
                ++cases;
                if (!ok || rel > 1e-12 || res > 1e-12 * fs || (sing && std::abs(mean) > 1e-12 * scale * N))
                    check(false, "dense " + std::to_string(nx) + "x" + std::to_string(ny) + "x" +
                                     std::to_string(nz) + " " + kname(kx) + "/" + kname(ky) + "/" +
                                     kname(kz) + " rel " + std::to_string(rel));
            }
    std::printf("  %dx%dx%d: %d kind triples, sep vs dense LU max rel %.2e, |Lp - f| %.2e\n", nx, ny, nz,
                cases, worst, worstRes);
    check(worst <= 1e-12 && worstRes <= 1e-12,
          std::to_string(nx) + "x" + std::to_string(ny) + "x" + std::to_string(nz) +
              ": separable == dense LU to 1e-12, every kind triple");
}

} // namespace

int main()
{
    std::printf("(a) eigenpairs of the 1D operator\n");
    eigenpairs();
    std::printf("(b) fast algorithms vs direct sums\n");
    fastVsDirect();
    std::printf("(c) separable vs dense LU\n");
    denseCompare(6, 8, 10);
    denseCompare(5, 7, 9);
    denseCompare(7, 6, 5);
    denseCompare(9, 4, 6);
    denseCompare(6, 8, 1);
    denseCompare(9, 7, 1);
    return dnstest::finish();
}
