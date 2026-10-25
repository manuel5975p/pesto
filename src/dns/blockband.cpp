#include "dns/blockband.hpp"

#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <format>
#include <utility>

namespace dns {
namespace {

constexpr int kMaxB = 7, kMaxK = 2;
// Right-hand sides of one diagonal solve: the K upper blocks plus b.
constexpr int kMaxRhs = kMaxK * kMaxB + 1;

bool usablePivot(double p) { return std::isfinite(p) && p != 0.0; }

// Gaussian elimination with partial pivoting on the B x B block D, applied to
// the B x m right-hand side X (row-major, stride m), overwriting X with
// D^{-1} X. False on a zero or non-finite pivot.
bool solveDiagonal(int B, std::array<double, kMaxB * kMaxB> D, double* X,
                   int m)
{
    for (int k = 0; k < B; ++k) {
        int piv = k;
        for (int i = k + 1; i < B; ++i)
            if (std::abs(D[i * B + k]) > std::abs(D[piv * B + k])) piv = i;
        if (!usablePivot(D[piv * B + k])) return false;
        if (piv != k) {
            for (int c = 0; c < B; ++c) std::swap(D[k * B + c], D[piv * B + c]);
            for (int c = 0; c < m; ++c) std::swap(X[k * m + c], X[piv * m + c]);
        }
        for (int i = k + 1; i < B; ++i) {
            const double l = D[i * B + k] / D[k * B + k];
            if (l == 0.0) continue;
            for (int c = k; c < B; ++c) D[i * B + c] -= l * D[k * B + c];
            for (int c = 0; c < m; ++c) X[i * m + c] -= l * X[k * m + c];
        }
    }
    for (int k = B - 1; k >= 0; --k)
        for (int c = 0; c < m; ++c) {
            double s = X[k * m + c];
            for (int i = k + 1; i < B; ++i) s -= D[k * B + i] * X[i * m + c];
            X[k * m + c] = s / D[k * B + k];
        }
    return true;
}

} // namespace

BlockBand BlockBand::make(int n, int B, int K)
{
    assert(n >= 1 && B >= 1 && B <= kMaxB && K >= 0 && K <= kMaxK);
    BlockBand A;
    A.n = n;
    A.B = B;
    A.K = K;
    A.a.assign(std::size_t(n) * (2 * K + 1) * B * B, 0.0);
    return A;
}

std::size_t BlockBand::index(int row, int off, int r, int c) const
{
    assert(row >= 0 && row < n && off >= -K && off <= K && r >= 0 && r < B &&
           c >= 0 && c < B);
    return ((std::size_t(row) * (2 * K + 1) + std::size_t(off + K)) * B + r) * B +
           c;
}

std::expected<void, std::string> solveBlockBand(BlockBand A, std::span<double> b)
{
    const int n = A.n, B = A.B, K = A.K;
    assert(b.size() == std::size_t(n) * B && B <= kMaxB && K <= kMaxK);
    // Forward: row k's diagonal block solves its upper blocks and rhs, X =
    // D_k^{-1} [U_k,1 .. U_k,K | b_k], stored back in place; every row below
    // inside the band subtracts its sub-diagonal block times X.
    for (int k = 0; k < n; ++k) {
        const int up = std::min(K, n - 1 - k); // upper blocks that exist
        const int m = up * B + 1;
        std::array<double, kMaxB * kMaxB> D{};
        for (int r = 0; r < B; ++r)
            for (int c = 0; c < B; ++c) D[r * B + c] = A.at(k, 0, r, c);
        std::array<double, kMaxB * kMaxRhs> X{};
        for (int r = 0; r < B; ++r) {
            for (int o = 1; o <= up; ++o)
                for (int c = 0; c < B; ++c)
                    X[r * m + (o - 1) * B + c] = A.at(k, o, r, c);
            X[r * m + up * B] = b[std::size_t(k) * B + r];
        }
        if (!solveDiagonal(B, D, X.data(), m))
            return std::unexpected(std::format(
                "zero or non-finite pivot in diagonal block {} of {}", k, n));
        for (int r = 0; r < B; ++r) {
            for (int o = 1; o <= up; ++o)
                for (int c = 0; c < B; ++c)
                    A.at(k, o, r, c) = X[r * m + (o - 1) * B + c];
            b[std::size_t(k) * B + r] = X[r * m + up * B];
        }
        for (int i = k + 1; i <= k + up; ++i) {
            const int li = k - i; // block (i, k): offset -(i - k)
            for (int r = 0; r < B; ++r)
                for (int s = 0; s < B; ++s) {
                    const double l = A.at(i, li, r, s);
                    if (l == 0.0) continue;
                    // Columns k+1..k+up of row i: offsets li + o.
                    for (int o = 1; o <= up; ++o)
                        for (int c = 0; c < B; ++c)
                            A.at(i, li + o, r, c) -= l * X[s * m + (o - 1) * B + c];
                    b[std::size_t(i) * B + r] -= l * X[s * m + up * B];
                }
        }
    }
    // Backward: x_k = X_b,k - sum_o (D^{-1} U)_k,o x_{k+o}.
    for (int k = n - 1; k >= 0; --k) {
        const int up = std::min(K, n - 1 - k);
        for (int r = 0; r < B; ++r) {
            double s = b[std::size_t(k) * B + r];
            for (int o = 1; o <= up; ++o)
                for (int c = 0; c < B; ++c)
                    s -= A.at(k, o, r, c) * b[std::size_t(k + o) * B + c];
            b[std::size_t(k) * B + r] = s;
        }
        for (int r = 0; r < B; ++r)
            if (!std::isfinite(b[std::size_t(k) * B + r]))
                return std::unexpected(
                    std::format("non-finite solution in block row {} of {}", k, n));
    }
    return {};
}

} // namespace dns
