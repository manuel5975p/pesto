// Block matrices of the solver library (gpu/la, docs/la.md): the block shape,
// the sliced-ELL pattern every device matrix shares, and the host FP64 block
// matrix that tests, oracles and host assembly fill. Host-only, mesh-agnostic:
// a matrix sees a CellGraph (rows, ascending columns, per-face slots), never
// faces.
#pragma once

#include "dns/la/graph.hpp"

#include <cstddef>
#include <expected>
#include <span>
#include <string>
#include <vector>

namespace dns::la {

// Rows per ELL slice: the device row block, coalesced in the lane.
inline constexpr int kSliceRows = 64;

// Block shape: dense B x B diagonal blocks; off-diagonal blocks are a dense
// Bd x Bd top-left part plus, on rows Bd..B-1, a DIAGONAL of nS scalars
// (nS == 1: one shared scalar for every such row, nS == B - Bd: one per row).
// Bd == B (nS == 0) is a fully dense off-diagonal. The chemistry layout
// B = 5 + nT + nY is Bd = 5 with the scalars shared (masterplan 2.3).
struct BlockShape
{
    int B = 1, Bd = 1, nS = 0;
    int offDense() const { return Bd * Bd; }
    bool shared() const { return nS == 1 && B - Bd > 1; }
    bool operator==(const BlockShape&) const = default;
};

// The block sizes with compiled kernels: 1 (pressure), 3 (momentum), 5 (mean
// flow), 6/7 (+ SA / SST), 9/11 (+ 4/6 species).
std::span<const int> supportedBlockSizes();

// A supported shape: B in supportedBlockSizes(), Bd == B or (Bd == 5 < B);
// sharedScalar picks nS = 1 over nS = B - Bd (ignored when Bd == B).
// Error names the unsupported value.
std::expected<BlockShape, std::string> blockShape(int B, int Bd, bool sharedScalar);
// Dense shape B x B. Error as blockShape.
std::expected<BlockShape, std::string> denseShape(int B);

// Sliced ELL: row c lives in slice c / 64, lane c % 64; slice s holds
// sliceWidth[s] entries per row (its widest row), entry k of row c at
// E = sliceBase[s] + 64 k + lane, columns ascending, padding col -1 after the
// real entries. slotPos maps a CellGraph slot (index into g.col) to its E.
struct EllPattern
{
    int n = 0, nSlices = 0, nE = 0, maxWidth = 0;
    std::vector<int> sliceBase, sliceWidth; // nSlices + 1, nSlices
    std::vector<int> col;                   // nE, -1 = padding
    std::vector<int> slotPos;               // nnz(g)
    bool operator==(const EllPattern&) const = default;
};

// Pattern of g. Post: every slot maps to a distinct E with col[E] == g.col[slot].
EllPattern ellPattern(const CellGraph& g);

// Host FP64 block matrix on a CellGraph pattern (rows, slots): what device
// matrices are uploaded from and what the host oracles read.
//  diag[(c * B + r) * B + q]          block (c, c), row-major
//  off [(slot * Bd + r) * Bd + q]     dense part of block (c, col[slot])
//  sc  [slot * nS + e]                diagonal scalars of rows Bd + e
struct BlockMatrix
{
    BlockShape shape;
    int n = 0;
    std::vector<int> rowPtr, col;
    std::vector<double> diag, off, sc;

    // A zero matrix on g's pattern.
    static BlockMatrix zeros(const CellGraph& g, BlockShape s);

    // Entry (r, q) of the full logical B x B block at slot (0 where the
    // shape stores nothing). Pre: r, q < B.
    double offAt(int slot, int r, int q) const;
    double diagAt(int c, int r, int q) const
    {
        return diag[(std::size_t(c) * std::size_t(shape.B) + std::size_t(r)) * std::size_t(shape.B) +
                    std::size_t(q)];
    }
    std::size_t nnz() const { return col.size(); }
};

} // namespace dns::la
