// Aggregation AMG setup of the solver library (gpu/la/amg, docs/la.md,
// masterplan_meshsolver 2.4 / P5): pairwise aggregation (Notay type) on a
// CellGraph with per-slot strength weights, the coarse patterns, and the
// FIXED-ORDER gather maps the device refresh sums fine entries along
// (coarse = P^T A P with piecewise-constant P: no SpGEMM). Host-only, FP64,
// mesh-agnostic (rows, ascending columns, slots), deterministic: Notay's
// priority order (fewest unassigned strong couplers first, bucket queue),
// ties to the lower column, every list ascending.
#pragma once

#include "dns/la/ell.hpp"
#include "dns/la/lines.hpp"

#include <cstdint>
#include <expected>
#include <span>
#include <string>
#include <vector>

namespace dns::la {

// The largest dense coarsest system (unknowns = rows x B) the device inverts.
inline constexpr int kAmgMaxCoarseUnknowns = 1024;

struct AmgSetup
{
    double strongBeta = 0.25;  // link (i, j) strong if w_ij >= beta max_k w_ik
    int passes = 2;            // pairwise passes per level (2: aggregates <= 4)
    int coarseUnknowns = 128;  // a level with rows B <= this is the coarsest (dense)
    int maxLevels = 30;
    double minRatio = 1.2;     // coarsening stalls below rows_l / rows_{l+1}
    double lineRatio = 0;      // > 1: strongLines on every level at this ratio
};

// Level l -> l + 1. Coarse row I sums its members' rows; the coarse diagonal
// block also sums every fine slot whose two rows both lie in I; coarse slot s
// sums the fine slots offSlot[offPtr[s] .. offPtr[s + 1]). Every list ascending.
struct AmgTransfer
{
    std::vector<int> agg;                 // fine row -> coarse row
    std::vector<int> memberPtr, member;   // coarse row -> fine rows
    std::vector<int> intraPtr, intraSlot; // coarse row -> fine slots inside it
    std::vector<int> offPtr, offSlot;     // coarse slot -> fine slots
    bool operator==(const AmgTransfer&) const = default;
};

// Levels 0 .. levels() - 1; level 0 is the caller's graph (not stored).
struct AmgHierarchy
{
    int B = 1;
    std::vector<int> rows, nnz;                    // per level (nnz: graph slots)
    std::vector<CellGraph> graphs;                 // levels 1 .. (graphs[l - 1])
    std::vector<AmgTransfer> transfers;            // transfers[l]: level l -> l + 1
    std::vector<std::vector<std::uint8_t>> colours; // per level; empty when a row
                                                   // has > 254 entries (no MC-GS)
    std::vector<LineSet> lines;                    // per level iff lineRatio > 1
    int levels() const { return int(rows.size()); }
    // (sum_l rows_l) / rows_0 and (sum_l rows_l + nnz_l) / (rows_0 + nnz_0).
    double gridComplexity() const;
    double operatorComplexity() const;
    bool operator==(const AmgHierarchy&) const = default;
};

// The hierarchy of g: pairwise passes, slot weights of each coarse level the
// sums of the fine weights between its aggregates, stop at the coarse size or
// a stalled ratio. Error by name: weight count, a negative or non-finite
// weight, a bad option, a coarsest level above kAmgMaxCoarseUnknowns.
// Pre: g neighbour-unique ascending rows (graphOf, structuredGraph). B >= 1.
std::expected<AmgHierarchy, std::string> amgHierarchy(const CellGraph& g, std::span<const double> slotWeight, int B,
                                                      const AmgSetup& s);

// Strength weights from values: B == 1 max(-a_ij, 0) (Notay: negative
// couplings), B > 1 the Frobenius norm of the logical off block.
std::vector<double> amgStrength(const BlockMatrix& m);

// Host Galerkin restatement of the device refresh, same order: diag(I) =
// sum of member diagonals ascending, then the intra slots; off(s) = its fine
// slots. Pre: t is the transfer from m's graph to `coarse`.
BlockMatrix amgGalerkin(const BlockMatrix& m, const CellGraph& coarse, const AmgTransfer& t);

} // namespace dns::la
