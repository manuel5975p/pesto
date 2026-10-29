// Wall distance of the incompressible FV path's RAS models (phase BL,
// docs/fv.md), OpenFOAM v2606 verbatim, host, FP64, on the polyMesh's own
// numbering. Two distances, as OpenFOAM keeps them:
//  y      the model's wallDist (F1/F2 of kOmegaSST, d of SpalartAllmaras):
//         `meshWave` = FaceCellWave of wallPoint (nearest wall-face CENTRE,
//         propagationTol 0.01, serial order) + correctBoundaryCells
//         (useCombinedWallPatch: wall-face cells and wall-point cells take the
//         exact distance to the faces around them); `exactDistance` = nearest
//         point on the wall faces for every cell (the knob of masterplan U9).
//  yWall  nearWallDist (the wall functions' y): per wall face, the cell
//         centre's distance to the face and its point neighbours on the
//         combined wall patch.
// "Wall" = every polyMesh patch of type wall. Face distance = face::nearestPoint
// (fan of triangles (p_i, p_i+1, face::centre), Ericson's closest point).
#pragma once

#include "dns/facemesh.hpp"
#include "dns/polymesh.hpp"

#include <expected>
#include <string>
#include <vector>

namespace dns {

struct FvWallDist
{
    std::vector<double> y;     // [fm.nC] FaceMesh cell order
    std::vector<double> yWall; // [fm.nB] FaceMesh boundary face fb (face fm.nI + fb); 0 off the walls
    long nUnset = 0;           // meshWave: cells the wave never reached (no wall in their region)
};

// Pre: fm = FaceMesh::fromPolyMesh(pm, CellOrder::Source). Error (exit 3): no
// wall patch; meshWave on a mesh with cyclic patches (OpenFOAM's cyclic
// transfer of the wave is not reproduced); a cell the wave never reached.
std::expected<FvWallDist, std::string> fvWallDistance(const PolyMesh& pm, const FaceMesh& fm, bool exact,
                                                      bool correctWalls);

} // namespace dns
