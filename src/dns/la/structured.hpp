// Index-space graphs and lines of a structured block (Grid::idx numbering,
// i fastest): the 7-point pattern without building a FaceMesh -- the
// structured NK's graph (masterplan P6) and the solver library's test
// operators. Host-only.
#pragma once

#include "dns/la/graph.hpp"
#include "dns/la/lines.hpp"

#include <array>

namespace dns::la {

// The 7-point (5-point at n[2] == 1) graph of an n[0] x n[1] x n[2] block;
// periodic[a] wraps axis a. Neighbour-unique (a 2-cell periodic axis is one
// entry), no self entry; ownSlot/neiSlot empty (no faces).
// Post: rowPtr/col == graphOf(FaceMesh::fromCurvMesh(uniform, Source order)).
// Pre: n[a] >= 1.
CellGraph structuredGraph(std::array<int, 3> n, std::array<bool, 3> periodic);

// Lines along `axis` (open chains, a periodic wrap is lagged), ordered by the
// other two indices lexicographically (lower axis fastest). Pre: axis < 3.
LineSet gridLines(std::array<int, 3> n, int axis);

} // namespace dns::la
