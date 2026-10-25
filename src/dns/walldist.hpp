// Wall distance for the RANS closures: d at every cell of a mesh, evaluated
// at the NODE-MEAN centroid (the centroid CurvMetrics and setMrfCurv use).
// Ghost cells take the d of the cell they mirror, so every cell that
// evaluates a model source has d > 0 (their outputs are overwritten anyway).
// Host-side, build time only.
#pragma once

#include "dns/curvmesh.hpp"

#include <vector>

namespace dns {

// Distance to the channel plates y = 0 and y = 2 delta at the node-mean
// centroid of every cell of m, Grid::idx layout; ghost rows (0, ny-1) take the
// d of the row they mirror (1, ny-2). Exact (the plates are planes).
// Pre: m from CurvMesh::channelY(g, beta, delta), delta > 0.
std::vector<double> channelWallDistance(const CurvMesh& m, double delta);

} // namespace dns
