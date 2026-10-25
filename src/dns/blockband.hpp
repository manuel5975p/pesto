// Block-banded linear systems in FP64: the line solve of the implicit steady
// RANS march (docs/rans.md, "Implicit pseudo-time"). n block rows of B x B
// blocks, half-bandwidth K blocks (block (i, j) is stored iff |i - j| <= K).
// Host-only and small (n ~ 1e2-1e3, B <= 7, K <= 2), so a dense block LU is
// the whole design.
#pragma once

#include <cstddef>
#include <expected>
#include <span>
#include <string>
#include <vector>

namespace dns {

struct BlockBand
{
    int n = 0, B = 0, K = 0;
    // [row][off + K][r][c]: block (row, row + off), row-major inside; blocks
    // reaching past either end are stored and never read.
    std::vector<double> a;

    // A zero matrix. Pre: n >= 1, 1 <= B <= 7, 0 <= K <= 2.
    static BlockBand make(int n, int B, int K);

    // Entry (r, c) of block (row, row + off). Pre: |off| <= K, r, c < B.
    double& at(int row, int off, int r, int c)
    {
        return a[index(row, off, r, c)];
    }
    double at(int row, int off, int r, int c) const
    {
        return a[index(row, off, r, c)];
    }

private:
    std::size_t index(int row, int off, int r, int c) const;
};

// Solves A x = b in place (b becomes x): block LU WITHOUT inter-block
// pivoting, so fill stays inside the band; partial pivoting inside each
// diagonal block. Fixed order, deterministic.
// Pre: b.size() == A.n * A.B. Error (never an assert) on a zero or
// non-finite pivot, b then unspecified.
std::expected<void, std::string> solveBlockBand(BlockBand A, std::span<double> b);

} // namespace dns
