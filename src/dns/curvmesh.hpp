// Distorted regular (structured curvilinear) mesh for the compressible
// solver: logically the Grid, one node per cell (its low corner). Each axis
// continues past its ends by its own AxisWrap -- by default the translated
// periodic image offset by the axis-aligned box (lx, ly, lz), which is what
// every Cartesian-topology mesh here wants; an O-grid needs a closed angular
// axis and a linearly extended radial one instead.
// CurvMetrics turns nodes into the finite-volume tables the scheme consumes.
#pragma once

#include "dns/grid.hpp"
#include "dns/remap.hpp"

#include <array>
#include <expected>
#include <string>
#include <vector>

namespace dns {

// How node() continues past either end of one axis. The mesh stores exactly
// one node per cell, so every stencil that reaches index n needs a rule --
// CurvMetrics::build reaches one node past the top of every axis, which is
// the only reason this enum has to exist.
enum class AxisWrap
{
    // Periodic image offset by the box: node(i + n) = node(i) + L_axis. Every
    // distorted, stretched and channel mesh in the tree; the default, so a
    // mesh that says nothing keeps the historical behaviour bit-for-bit.
    Translate,
    // Closed axis: node(i + n) = node(i) EXACTLY, no offset. The angular axis
    // of an O-grid, whose last cell's hi face IS the first cell's lo face.
    Cyclic,
    // Not periodic at all: linear extension off each end,
    //   node(-1) = 2 node(0) - node(1),  node(n) = 2 node(n-1) - node(n-2).
    // The radial axis of an O-grid: it never wraps the far boundary back onto
    // the wall, and it supplies the outer ghost row's top nodes for free.
    // Needs n >= 2 (build() rejects a degenerate Extend axis).
    Extend,
    // The axisymmetric WEDGE's azimuthal axis (z only, n == 3 exactly): node
    // planes 0..2 are stored and node(i, j, 3) is node(i, j, 0) mirrored in
    // z about CurvMesh::mirrorZ0 -- so the two side faces of the real cell
    // plane k = 1 are exact z-mirrors and every metric of the ghost planes is
    // the mirror of its partner's. Any other out-of-range k is a programmer
    // error (asserted): build() reaches only one node past the top.
    MirrorZ,
};

struct CurvMesh
{
    Grid g;                      // logical grid; periods are nx*hx, ny*hy, nz*hz
    std::vector<double> x, y, z; // node coordinates, Grid::idx layout
    // Per-axis continuation for node(); all-Translate is the historical mesh.
    std::array<AxisWrap, 3> wrap = {AxisWrap::Translate, AxisWrap::Translate,
                                    AxisWrap::Translate};
    // The z of the MirrorZ plane (wrap[2] == MirrorZ only).
    double mirrorZ0 = 0;
    // Degenerate AXIS faces of a revolved mesh: the +i face of every cell
    // in columns axisFaces[0] and axisFaces[1] (-1 = none, today's mesh)
    // lies on the axis of revolution, S == 0 EXACTLY. build() requires that,
    // skips their Jacobian inversion and stores the analytic limit metric:
    // rows j and k of dXi/dx are 0 (only ever read multiplied by S = 0) and
    // row i is the across-axis central difference's, (t_j x e_phi)/(c_i .
    // (t_j x e_phi)) with c_i = +-2 rho_F rho^_F the vector from the fluid
    // cell F's centroid to its across-axis continuation, oriented +i.
    // The columns PAST the axis (axisFaces[0] itself and axisFaces[1] + 1)
    // are placeholders whose node geometry has the wrong handedness (an
    // across-axis continuation runs its azimuth backwards in index space):
    // build() gives them their real neighbour's volume and j-/k-face tables.
    // No real cell reads them; the ghost's own rhs is discarded.
    std::array<int, 2> axisFaces = {-1, -1};

    // Undistorted mesh: node (i,j,k) = (i hx, j hy, k hz).
    static CurvMesh uniform(const Grid& g);

    // Nodes from any PointRemap (dns/remap.hpp): node (i,j,k) =
    // remap(i hx, j hy, k hz). The remap is the mathematical source of truth;
    // CurvMetrics::build still derives the finite-volume tables from the
    // nodes so the discrete face closure survives.
    template <class Remap>
    static CurvMesh fromRemap(const Grid& g, const Remap& r)
    {
        CurvMesh m = uniform(g);
        for (std::size_t c = 0; c < m.x.size(); ++c) {
            const double X[3] = {m.x[c], m.y[c], m.z[c]};
            double x[3];
            r.map(X, x);
            m.x[c] = x[0];
            m.y[c] = x[1];
            m.z[c] = x[2];
        }
        return m;
    }

    // Smooth fixed mapping x = X + amp*L*sin(2 pi X_other/L): skewness stays
    // finite under refinement. |amp| <= 0.06 keeps cells valid; nz == 1
    // stays planar (z = 0) so the exact 2D restriction survives.
    // Equivalent to fromRemap(g, SinusoidalRemap{...}).
    static CurvMesh sinusoidal(const Grid& g, double amp);

    // Two-sided tanh wall-normal clustering along y (TanhStretchY): cells
    // shrink toward both y extremes, x and z stay uniform, nz == 1 stays
    // planar. beta == 0 reproduces uniform() bit-for-bit.
    static CurvMesh stretchedY(const Grid& g, double beta);

    // Channel mesh (ChannelStretchY): rows 0 and ny-1 are wall-mirror ghost
    // rows, exact geometric reflections of the first fluid rows; the wall
    // faces sit ON nodes 1 and ny-1 at y = 0 and 2*delta. The caller must
    // have sized the grid with ly = 2*delta + 2*ChannelStretchY::firstCell.
    static CurvMesh channelY(const Grid& g, double beta, double delta);

    // Node position for any logical index, continued per axis by wrap[a].
    // Pre: an Extend axis needs n >= 2; Translate/Cyclic accept any index.
    std::array<double, 3> node(int i, int j, int k) const;
};

// Precomputed finite-volume metrics; build() is the only validator (sizes,
// positive volumes, invertible face Jacobians, planar layer at nz == 1).
struct CurvMetrics
{
    Grid g;
    std::array<AxisWrap, 3> wrap = {AxisWrap::Translate, AxisWrap::Translate,
                                    AxisWrap::Translate}; // copied from the mesh
    // S[a][c]: component c of the +a-face area vector of each cell. The face
    // is the exact bilinear quad of its 4 nodes, so per-cell closure
    // sum_faces S = 0 holds identically (freestream preservation) -- the lo
    // face of cell i is read as S[a](i-1). On an Extend axis cell 0 has no lo
    // neighbour, so S[a](n-1) is NOT its lo face: cell 0 is a ghost whose
    // state the BC overwrites, and no flux divergence may be taken there.
    std::array<std::array<std::vector<double>, 3>, 3> S;
    // Ji[a][d*3+c]: d xi_d / d x_c at the +a face (index space has dxi = 1).
    std::array<std::array<std::vector<double>, 9>, 3> Ji;
    // Exact hex volumes via consistently-diagonalized face triangles: cells
    // tile the box, sum vol == lx ly lz.
    std::vector<double> vol;
    // Per-axis mean cross-face area / volume (acoustic CFL width 1/aov); at
    // index 0 of an Extend axis the missing lo neighbour is clamped to the
    // cell's own face. On a MirrorZ (wedge) axis the width is rho, not
    // rho dphi: aov[2] carries the factor hz = dphi (no azimuthal mode) for
    // the time march.
    std::array<std::vector<double>, 3> aov;
    // MirrorZ only (else empty): the azimuthal aov WITHOUT that factor,
    // 1/(rho dphi) -- the width of the steady local dt (DeviceComp::localDt):
    // the cold-start RANS march needs it (docs/rans.md).
    std::vector<double> aovSteadyZ;
    double hmin = 0.0; // min cell width over cells and non-degenerate axes

    // nx, ny >= 2; nz == 1 or nz >= 2; node arrays sized g.size().
    // threads <= 0: hardware concurrency. Contiguous (j, k)-row blocks, each
    // cell by the serial expression: bit-identical for ANY thread count,
    // and the error returned is the serial loop's first (host_threads).
    static std::expected<CurvMetrics, std::string> build(const CurvMesh& m,
                                                         int threads = 0);
};

} // namespace dns
