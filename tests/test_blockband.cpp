// dns/blockband: the block-banded FP64 solve of the implicit RANS line, against
// a banded Gaussian elimination with partial pivoting over the whole matrix
// (what a dense partial-pivot LU does on a banded matrix), on random
// diagonally dominant systems and on the real line matrix diag(1/dtau) - J at
// CFL 1e8; and its error contract (a singular or non-finite diagonal block
// is an error, never an assert or a NaN). Milliseconds.
#include "dns/blockband.hpp"
#include "dns/pseudo_time.hpp"
#include "channel_line_test_util.hpp"
#include "check.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <format>
#include <random>
#include <vector>

using namespace dns;
using dnstest::check;

namespace {

// Dense row-major copy of A (N = n B).
std::vector<double> dense(const BlockBand& A)
{
    const int N = A.n * A.B;
    std::vector<double> D(std::size_t(N) * N, 0.0);
    for (int i = 0; i < A.n; ++i)
        for (int o = -A.K; o <= A.K; ++o) {
            const int j = i + o;
            if (j < 0 || j >= A.n) continue;
            for (int r = 0; r < A.B; ++r)
                for (int c = 0; c < A.B; ++c)
                    D[std::size_t(i * A.B + r) * N + j * A.B + c] = A.at(i, o, r, c);
        }
    return D;
}

// Gaussian elimination with partial pivoting over every row below the
// diagonal that can be non-zero (scalar half-bandwidth bw), fill to 2 bw.
std::vector<double> denseSolve(std::vector<double> A, std::vector<double> b,
                               int N, int bw)
{
    for (int k = 0; k < N; ++k) {
        const int rmax = std::min(N, k + bw + 1), cmax = std::min(N, k + 2 * bw + 1);
        int piv = k;
        for (int i = k + 1; i < rmax; ++i)
            if (std::abs(A[std::size_t(i) * N + k]) > std::abs(A[std::size_t(piv) * N + k]))
                piv = i;
        if (piv != k) {
            for (int c = k; c < cmax; ++c)
                std::swap(A[std::size_t(k) * N + c], A[std::size_t(piv) * N + c]);
            std::swap(b[k], b[piv]);
        }
        for (int i = k + 1; i < rmax; ++i) {
            const double l = A[std::size_t(i) * N + k] / A[std::size_t(k) * N + k];
            for (int c = k; c < cmax; ++c)
                A[std::size_t(i) * N + c] -= l * A[std::size_t(k) * N + c];
            b[i] -= l * b[k];
        }
    }
    for (int k = N - 1; k >= 0; --k) {
        double s = b[k];
        for (int c = k + 1; c < std::min(N, k + 2 * bw + 1); ++c)
            s -= A[std::size_t(k) * N + c] * b[c];
        b[k] = s / A[std::size_t(k) * N + k];
    }
    return b;
}

double maxAbs(const std::vector<double>& v)
{
    double m = 0;
    for (double x : v) m = std::max(m, std::abs(x));
    return m;
}

// Normwise backward error ||A x - b|| / (||A|| ||x|| + ||b||), max norm.
double backwardError(const std::vector<double>& D, const std::vector<double>& x,
                     const std::vector<double>& b)
{
    const std::size_t N = b.size();
    double rmax = 0, anorm = 0;
    for (std::size_t i = 0; i < N; ++i) {
        double s = -b[i], row = 0;
        for (std::size_t c = 0; c < N; ++c) {
            s += D[i * N + c] * x[c];
            row += std::abs(D[i * N + c]);
        }
        rmax = std::max(rmax, std::abs(s));
        anorm = std::max(anorm, row);
    }
    return rmax / (anorm * maxAbs(x) + maxAbs(b));
}

// The block solve against the dense one: both backward errors <= 1e-12 and,
// when `conditioned`, the solutions agree to 1e-10 (max norm, relative).
// A nearly singular system (the line at CFL 1e8: sum vol R_rho == 0 makes J
// exactly singular, so cond ~ 1e8 ||J||) separates ANY two backward-stable
// solvers by ~cond eps; there only the backward errors are meaningful.
void compare(const BlockBand& A, const std::vector<double>& b,
             const std::string& what, bool conditioned = true)
{
    const int N = A.n * A.B;
    std::vector<double> x = b;
    const auto e = solveBlockBand(A, x);
    check(e.has_value(), what + ": solves");
    if (!e) return;
    const auto D = dense(A);
    const auto xd = denseSolve(D, b, N, (A.K + 1) * A.B - 1);
    const double back = backwardError(D, x, b), backD = backwardError(D, xd, b);
    double dmax = 0;
    for (int i = 0; i < N; ++i) dmax = std::max(dmax, std::abs(x[i] - xd[i]));
    const double rel = dmax / maxAbs(xd);
    std::printf("  %s: backward error %.2e (dense %.2e), vs dense LU %.2e\n",
                what.c_str(), back, backD, rel);
    check(back <= 1e-12 && backD <= 1e-12,
          what + ": relative residual <= 1e-12 (block and dense)");
    if (conditioned)
        check(rel <= 1e-10, what + ": agrees with dense partial-pivot LU <= 1e-10");
}

void randomSystems()
{
    std::mt19937_64 rng(7);
    std::uniform_real_distribution<double> U(-1.0, 1.0);
    for (int B : {5, 6})
        for (int K : {1, 2})
            for (int n : {8, 64, 254}) {
                BlockBand A = BlockBand::make(n, B, K);
                for (double& v : A.a) v = U(rng);
                // Diagonally dominant by rows.
                for (int i = 0; i < n; ++i)
                    for (int r = 0; r < B; ++r) {
                        double s = 0;
                        for (int o = -K; o <= K; ++o)
                            for (int c = 0; c < B; ++c) s += std::abs(A.at(i, o, r, c));
                        A.at(i, 0, r, r) = s + 1.0;
                    }
                std::vector<double> b(std::size_t(n) * B);
                for (double& v : b) v = U(rng);
                compare(A, b, std::format("random B {} K {} n {}", B, K, n));
            }
}

// The actual line matrix diag(1/dtau) - J at the IC: at CFL 100 (the ramp's
// start, conditioned) and 1e8 (the cap, nearly singular -- see compare()).
void lineSystem(RansModel model, int ny, double cfl)
{
    Channel1dCase cc;
    cc.ny = ny;
    cc.model = model;
    const auto r = linetest::makeLine(cc);
    const auto q = linetest::icColumn(r);
    BlockBand A = r.line.jacobian(q);
    std::vector<double> dtau(r.line.rows());
    r.line.pseudoStep(q, cfl, dtau);
    for (double& v : A.a) v = -v;
    for (int j = 0; j < A.n; ++j)
        for (int u = 0; u < A.B; ++u) A.at(j, 0, u, u) += 1.0 / dtau[j + 1];
    std::vector<double> R(q.size()), b(std::size_t(A.n) * A.B);
    r.line.residual(q, R);
    for (int j = 0; j < A.n; ++j)
        for (int u = 0; u < A.B; ++u)
            b[std::size_t(j) * A.B + u] =
                R[std::size_t(j + 1) * r.line.fields() + ChannelLine::field(u)];
    compare(A, b, std::format("line {} ny {} CFL {:g}", ransName(model), ny, cfl),
            cfl == kImplicitCfl0);
}

void lineSystems()
{
    for (RansModel model : {RansModel::SaNeg, RansModel::Sst2003m})
        for (int ny : {64, 254})
            for (double cfl : {kImplicitCfl0, kImplicitCflMax})
                lineSystem(model, ny, cfl);
}

void errors()
{
    BlockBand A = BlockBand::make(16, 5, 2);
    for (int i = 0; i < 16; ++i)
        for (int r = 0; r < 5; ++r) {
            A.at(i, 0, r, r) = 4.0;
            if (i > 0) A.at(i, -1, r, r) = 1.0;
        }
    std::vector<double> b(16 * 5, 1.0);
    {
        BlockBand S = A;
        for (int r = 0; r < 5; ++r)
            for (int c = 0; c < 5; ++c) S.at(7, 0, r, c) = 0.0;
        auto x = b;
        const auto e = solveBlockBand(S, x);
        check(!e.has_value(), "a singular diagonal block is an error");
        if (!e) std::printf("    -> %s\n", e.error().c_str());
    }
    {
        BlockBand S = A;
        S.at(3, 0, 1, 2) = std::nan("");
        auto x = b;
        check(!solveBlockBand(S, x).has_value(), "a NaN entry is an error");
    }
    {
        auto x = b;
        x[11] = HUGE_VAL;
        check(!solveBlockBand(A, x).has_value(), "an infinite rhs is an error");
    }
    {
        auto x = b;
        check(solveBlockBand(A, x).has_value() &&
                  std::ranges::all_of(x, [](double v) { return std::isfinite(v); }),
              "the regular system solves finite");
    }
}

} // namespace

int main()
{
    randomSystems();
    lineSystems();
    errors();
    return dnstest::finish();
}
