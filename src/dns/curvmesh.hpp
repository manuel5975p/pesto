// Distorted regular (structured curvilinear, fully periodic) mesh for the
// compressible solver: logically the Grid, one node per cell (its low
// corner), periodic images offset by the axis-aligned box (lx, ly, lz).
// CurvMetrics turns nodes into the finite-volume tables the scheme consumes.
#pragma once

#include "dns/grid.hpp"

#include <array>
#include <expected>
#include <string>
#include <vector>

namespace dns {

struct CurvMesh
{
    Grid g;                      // logical grid; periods are nx*hx, ny*hy, nz*hz
    std::vector<double> x, y, z; // node coordinates, Grid::idx layout

    // Undistorted mesh: node (i,j,k) = (i hx, j hy, k hz).
    static CurvMesh uniform(const Grid& g);

    // Smooth fixed mapping x = X + amp*L*sin(2 pi X_other/L): skewness stays
    // finite under refinement. |amp| <= 0.06 keeps cells valid; nz == 1
    // stays planar (z = 0) so the exact 2D restriction survives.
    static CurvMesh sinusoidal(const Grid& g, double amp);

    // Node position for any logical index; periodic images add the box.
    std::array<double, 3> node(int i, int j, int k) const;
};

// Precomputed finite-volume metrics; build() is the only validator (sizes,
// positive volumes, invertible face Jacobians, planar layer at nz == 1).
struct CurvMetrics
{
    Grid g;
    // S[a][c]: component c of the +a-face area vector of each cell. The face
    // is the exact bilinear quad of its 4 nodes, so per-cell closure
    // sum_faces S = 0 holds identically (freestream preservation).
    std::array<std::array<std::vector<double>, 3>, 3> S;
    // Ji[a][d*3+c]: d xi_d / d x_c at the +a face (index space has dxi = 1).
    std::array<std::array<std::vector<double>, 9>, 3> Ji;
    // Exact hex volumes via consistently-diagonalized face triangles: cells
    // tile the box, sum vol == lx ly lz.
    std::vector<double> vol;
    // Per-axis mean cross-face area / volume (acoustic CFL width 1/aov).
    std::array<std::vector<double>, 3> aov;
    double hmin = 0.0; // min cell width over cells and non-degenerate axes

    // nx, ny >= 2; nz == 1 or nz >= 2; node arrays sized g.size().
    static std::expected<CurvMetrics, std::string> build(const CurvMesh& m);
};

} // namespace dns
