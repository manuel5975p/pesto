// Wall distance for the RANS closures: d at every cell of a mesh, evaluated
// at the NODE-MEAN centroid (the centroid CurvMetrics and setMrfCurv use).
// Ghost cells take the d of the cell they mirror, so every cell that
// evaluates a model source has d > 0 (their outputs are overwritten anyway).
// Host-side, build time only.
#pragma once

#include "dns/curvmesh.hpp"
#include "dns/remap.hpp"
#include "dns/sdf.hpp"

#include <vector>

namespace dns {

// Distance to the channel plates y = 0 and y = 2 delta at the node-mean
// centroid of every cell of m, Grid::idx layout; ghost rows (0, ny-1) take the
// d of the row they mirror (1, ny-2). Exact (the plates are planes).
// Pre: m from CurvMesh::channelY(g, beta, delta), delta > 0.
std::vector<double> channelWallDistance(const CurvMesh& m, double delta);

// Exact distance to the DISCRETE wall of a z-extruded O-grid -- the closed
// polyline of node row j = 1 in the k = 0 plane -- at the in-plane node-mean
// centroid of every cell, Grid::idx layout. Ghost rows take their source
// row's d (0 -> 1, ny-1 -> ny-2); every k carries the k = 0 value (the mesh
// is an extrusion). `threads` (0 = hardware) write disjoint rows, so the
// result is bitwise independent of it. Pre: m from buildOGrid.
std::vector<double> ogridWallDistance(const CurvMesh& m, int threads = 0);

// Exact distance to the DISCRETE wall of a REVOLVED O-grid (buildRevolved-
// OGrid): every real wall quad (node row j = 1, real columns; wedge: plane
// k = 1 only) split into 4 triangles through its centroid, an unsigned BVH
// closest-triangle query (MeshSdf::distance) at the node-mean centroid of
// every real cell. Ghosts take their source cell's d: rows 0 / ny-1 rows
// 1 / ny-2, the axis columns the cell they continue (wedge: the same plane;
// full: across the axis), the wedge planes plane 1's. `threads` (0 =
// hardware) write disjoint rows: bitwise independent of it.
// Pre: m from buildRevolvedOGrid with nTheta real columns, layout r.
std::vector<double> revolvedWallDistance(const CurvMesh& m, int nTheta,
                                         bool wedge, int threads = 0);

// The immersed path's d: |sdf| at the PHYSICAL cell centre -- the remap of
// the reference centre, the point the ghost wall classifies -- every cell,
// Grid::idx layout, floored at 1e-9 of the smallest reference spacing (a
// centre exactly on the surface). Exact where the sdf is a true distance
// (sphere, cylinder, cube, a mesh BVH); a lens or a scene union is not, and
// the validator warns. `threads` (0 = hardware) write disjoint rows: bitwise
// independent of it. Pre: sdf callable concurrently.
std::vector<double> immersedWallDistance(const Grid& g, const Sdf& sdf,
                                         const AxisRemap3& remap,
                                         int threads = 0);

// d1(i, k) = d(i, 1, k): the first-cell wall distance per wall face, nx*nz,
// i fastest. Pre: d from a wall-distance function of m's grid.
std::vector<double> wallFirstCellDistance(const Grid& g,
                                          const std::vector<double>& d);

} // namespace dns
