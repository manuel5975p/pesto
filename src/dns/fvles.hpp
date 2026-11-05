// The LES and thickened-flame geometry of a FaceMesh (docs/fvcomp.md "Species
// LES and TFLES"): the filter width, the TFLES sensor's face-neighbour walk and
// the efficiency's edge layers. Host-only, FP64, built once per run; FvRef and
// gpu/fv_comp read the same tables, so the oracle and the device dilate the same
// cells.
//
// The walk restates gpu/tfles's separable T-gated max filter on faces: along
// global axis a, direction +-1, a cell steps to the neighbour across its face
// whose outward unit normal is most aligned with +-e_a (ties: the first in CSR
// order); a boundary face stops the walk (the structured x clamp), a cyclic face
// is internal (the structured y, z wrap). On a uniform Cartesian mesh that is the
// structured line, cell for cell, so the filter is the structured one; on a mesh
// rotated or sheared by less than 45 deg it walks the mesh lines.
#pragma once

#include "dns/facemesh.hpp"
#include "dns/fvgrad.hpp"

#include <cstdint>
#include <expected>
#include <string>
#include <vector>

namespace dns {

// The squared filter width per cell: V^(2/3) in 3D, the cell's area V / depth on
// a 2D mesh (depth off the first `empty` face): kCompSmagNutCurv's curvDelta2.
std::vector<double> fvLesDelta2(const FaceMesh& fm);

struct FvTflesWalk
{
    int nAxes = 0;
    int axis[3] = {0, 1, 2};          // the walked global axes, x y z order, the empty axis skipped
    std::vector<int> next;            // [(slot * 2 + dir) * nC + c]: dir 0 = +e_a, 1 = -e_a; -1 stops
    std::vector<std::uint8_t> edge;   // per cell: within kTflesEdgeRows face layers of an active boundary face
    std::vector<double> lapW;         // per CSR entry of cellFace: corr . S on internal faces, 0 on boundary/empty
};

// The walk tables of fm (geom: FvRef's face geometry, for the active flags and the
// compact Laplacian's corr . S). Exactly tied faces go to the lowest srcFace, so the
// walk is numbering-invariant. Errors: a cell with no face toward some walked
// direction (a degenerate cell).
std::expected<FvTflesWalk, std::string> fvTflesWalk(const FaceMesh& fm, const FvFaceGeom& geom);

} // namespace dns
