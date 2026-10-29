// Line sets of the line smoother (gpu/la/lines, docs/la.md): chains of rows
// that the smoother inverts exactly (block-tridiagonal per line), every other
// coupling lagged. Host-only, deterministic.
#pragma once

#include "dns/la/graph.hpp"

#include <expected>
#include <span>
#include <string>
#include <vector>

namespace dns::la {

// Every row in exactly one line: line l is cells[start[l] .. start[l + 1]),
// consecutive cells adjacent in the graph. Rows on no chain are singletons.
struct LineSet
{
    int n = 0;
    std::vector<int> start, cells;
    int count() const { return int(start.size()) - 1; }
    int maxLength() const;
    bool operator==(const LineSet&) const = default;
};

// The chains in order, then every uncovered row as a singleton (ascending).
// Error by name: a row out of range, a row in two chains, an empty chain,
// consecutive rows not adjacent in g.
std::expected<LineSet, std::string> lineSet(const CellGraph& g, std::span<const std::vector<int>> chains);

// Algebraic lines (strongest-coupling chains). slotWeight[slot] >= 0 is the
// coupling of row c to g.col[slot]. A row is anisotropic if its largest
// weight >= ratio x its smallest; a link (c, j) is strong if both rows are
// anisotropic and w(c, j) >= ratio x min_row(c), w(j, c) >= ratio x min_row(j).
// Seeds in descending anisotropy (max/min, ties lower row), each grown both
// ways along its strongest unassigned strong link (ties lower column).
// Pre: slotWeight.size() == g.col.size(), ratio > 1. Deterministic.
LineSet strongLines(const CellGraph& g, std::span<const double> slotWeight, double ratio);

} // namespace dns::la
