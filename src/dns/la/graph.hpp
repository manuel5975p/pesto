// The matrix pattern of a FaceMesh: what every FV matrix assembly (gpu/la,
// masterplan_meshsolver P3+) scatters face blocks into. Mesh-agnostic beyond
// this header: a solver sees rows, columns and per-face slots, never faces.
#pragma once

#include "dns/facemesh.hpp"

#include <cstdint>
#include <span>
#include <utility>
#include <vector>

namespace dns::la {

// Neighbour-unique rows in ascending column order, no self entry. Per internal
// face the entry of nei in row own (ownSlot) and of own in row nei (neiSlot),
// as indices into col: assembly scatters face blocks there in face order --
// deterministic, no atomics. Two faces between one cell pair share slots.
struct CellGraph
{
    int n = 0;
    std::vector<int> rowPtr, col;      // n + 1, nnz
    std::vector<int> ownSlot, neiSlot; // per internal face
    bool operator==(const CellGraph&) const = default;
};

// The graph of fm's internal faces. Post: symmetric, slots consistent.
CellGraph graphOf(const FaceMesh& fm);

// graphOf(fm) plus the couplings (row, col) of extra (cyclicAMI: a patch face's
// cell and each cell its AMI addresses). Pre: row != col, both in [0, nC).
// Post: every (row, col) of extra is an entry (graphSlot finds it); symmetric
// only if extra is.
CellGraph graphOf(const FaceMesh& fm, std::span<const std::pair<int, int>> extra);

// The slot (index into g.col) of column col in row row, -1: no such entry.
int graphSlot(const CellGraph& g, int row, int col);

// Distance-1 greedy colouring in row order, smallest free colour.
// Pre: every row has at most 254 entries (colours fit uint8).
std::vector<std::uint8_t> greedyColour(const CellGraph& g);

// The colour-major permutation: perm[new] = old, stable (colour, then old index).
std::vector<int> colourMajorOrder(std::span<const std::uint8_t> colours);

} // namespace dns::la
