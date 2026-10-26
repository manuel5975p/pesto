// Body-fitted O-grid: an annular structured mesh whose INNER boundary IS the
// body surface, for the compressible curvilinear solver. Host-side, build time
// only -- this header owns the MESH, nothing about the boundary conditions,
// the runner or the CLI.
//
// TOPOLOGY (the channel's ghost-row convention, wrapped into an annulus):
//   i  angular,   nx cells, AxisWrap::Cyclic,   theta_i = -2 pi i / nx at node i
//   j  radial,    ny cells, AxisWrap::Extend,   INCLUDING two ghost rows
//   k  spanwise,  nz cells, AxisWrap::Translate, z = cz - span/2 + k hz
// The angular index runs CLOCKWISE on purpose: e_theta x e_r = -z, so an
// (increasing theta, outward radius, +z) index triple is LEFT-handed and every
// cell would come out with a negative volume and a flipped face Jacobian,
// which CurvMetrics::build rejects. Negating theta is the one free sign.
// Node row j = 1 is the WALL (on the body surface), node row j = ny-1 is the
// FAR boundary, so cell 0 is the inner ghost (inside the solid), cells
// 1..ny-2 are fluid, cell ny-1 is the outer ghost. Exactly like the channel:
// the ghost cells' state is overwritten by the BC every RK stage and the wall
// face is a real mesh face. The outer ghost's top node row costs nothing --
// it is the Extend continuation of the last two fluid rows.
//
// GEOMETRY. C(theta) is the wall point, n(theta) the unit outward wall normal,
// F(theta) = centre + rFar e_r(theta) the far-boundary point on the exact
// circle, and t in [0,1] the normalized radial coordinate (t = 0 wall, t = 1
// far). The node is the transfinite blend of the wall-normal ray with the
// radial ray, scaled by one radial distribution D:
//
//   P(theta, t) = C + D(t) [ (1 - w(t)) Ln(theta) n(theta) + w(t) (F - C) ]
//
//   w(t)  = 3t^2 - 2t^3     smoothstep, w(0)=0, w'(0)=0, w(1)=1, w'(1)=0
//   D(t)  = radial distribution, D(0)=0, D(1)=1, D' > 0 (ogridRadialDist)
//   Ln    = distance from C along n to the circle of radius rFar
//
// WHY Ln, and why the far term is a RAY and not the fixed point F. The naive
// blend (1-w)[C + D n] + w F degenerates: F does not depend on t, so at t = 1
// both (1-w) and w' vanish and dP/dt = 0 -- the outermost cell collapses to
// O(1/ny^2) thickness whatever D is. Blending two rays that both start at C
// and both reach the far circle at t = 1 fixes it: dP/dt(1) = D'(1) (F - C),
// strictly positive and exactly radial (that is what w'(1) = 0 buys), while
// dP/dt(0) = D'(0) Ln n is exactly wall-normal and w'(0) = 0 makes the
// deviation from normal O(t^2). Ln is the unique scale that makes the
// wall-normal ray land ON the far circle at t = 1, so the two blended curves
// share both endpoints and the blend cannot pull a node off the circle:
//   |C + Ln n - centre| = rFar,  C - centre = rw e_r
//   => Ln^2 + 2 Ln rw (e_r . n) + rw^2 - rFar^2 = 0
//   => Ln = -rw mu + sqrt(rw^2 mu^2 - rw^2 + rFar^2),  mu = e_r . n > 0,
// whose discriminant is rFar^2 - rw^2 (1 - mu^2) > 0 whenever rFar > rw.
// For a circular contour n == e_r and Ln == rFar - rw, the bracket collapses
// to (rFar - rw) e_r for every w, and the mesh is the EXACT polar annulus.
#pragma once

#include "dns/curvmesh.hpp"
#include "dns/grid.hpp"
#include "dns/sdf.hpp"

#include <array>
#include <expected>
#include <string>
#include <vector>

namespace dns {

struct OGridSpec
{
    double cx = 0, cy = 0, cz = 0; // body centre; contour must be star-shaped about it
    double rFar = 0;               // far-boundary radius
    double beta = 0;               // radial tanh clustering toward the wall; 0 = uniform
    double span = 1;               // spanwise physical extent (nz == 1: the unit depth)
};

// The azimuthal layout of a REVOLVED O-grid (a body of revolution about the
// x axis through the centre): the axisymmetric WEDGE -- one real cell plane
// of angle dphi between two ghost planes, AxisWrap::MirrorZ -- or the FULL
// revolution of nphi cells (Cyclic, nphi even: every azimuthal cell has its
// partner across the axis).
struct RevolvedSpec
{
    bool wedge = true;
    double dphi = 0;   // wedge angle, radians, in (0, 10 deg]; full: 2 pi/nphi
                       // (read only by the mesh export there)
    int nphi = 0;      // full revolution: even, >= 4
};

// The full revolution's default odd-mode trip amplitude (/ U; --ogrid-trip,
// the solenoidal vortex of runFlowOGrid). The mesh and an axisymmetric start
// are exactly pi-symmetric, so without a seed the m = 1 mode never arises.
// Bracket (64 x 48 x 32, M 0.2, TOL 1e-4, measured 2026-10-26): laminar Re
// 250 reaches the same steady m = 1 wake (cd 0.70690, cl -0.0647) from 0.005,
// 0.01, 0.02 and 0.05, |cl| passing 1e-2 at 29400 / 27000 / 24600 / 21400
// iterations -- against the untripped run's certification of the symmetric
// state at 36200, so 0.05 leaves the widest margin (the stop does not watch
// cl). Where m = 1 is stable it decays and costs nothing measurable: Re 100
// cd 1.08930 both ways (cl 1e-6, 23400 vs 22200 iterations), Re 200 cl 4e-5,
// SST-2003m Re 1e3 cd 0.49511 both ways (95000 vs 95200).
inline constexpr double kOgridTripDefault = 0.05;

struct OGrid
{
    CurvMesh mesh;
    // Revolved meshes only (buildRevolvedOGrid): i = theta from the
    // UPSTREAM axis, nTheta real columns between ghost columns 0 and
    // nTheta + 1, j = r, k = phi; the axis faces are mesh.axisFaces.
    bool revolved = false;
    RevolvedSpec rev{};
    int nTheta = 0;
    std::vector<double> rWall;                    // nx entries, r_w(theta_i)
    std::vector<std::array<double, 3>> wallNormal; // nx entries, unit outward, z == 0
    double h1 = 0;      // first fluid cell height, min over theta (the CFL one)
    double h1Max = 0;   // first fluid cell height, max over theta
    double growth = 1;  // max adjacent radial growth ratio over all theta
    double rWallMin = 0, rWallMax = 0;
    // Max angle (rad) of a first radial edge off the wall normal. Floor ~2e-8:
    // it is an acos of a dot product, which loses half the mantissa near 1.
    double normalDevMax = 0;
};

// Normalized radial distribution, one-sided tanh clustered toward the wall:
//   D(t) = 1 + tanh(beta (t - 1)) / tanh(beta),  D(0) = 0, D(1) = 1 exactly.
// beta <= 0 returns t verbatim -- the EXACT uniform limit, bit-for-bit.
// Pre: t in [0, 1].
double ogridRadialDist(double t, double beta);

// First fluid cell height of the O-grid around a CIRCLE of radius rWall, the
// flags-only number the validator can quote (no sdf): (rFar - rWall) D(1/nf),
// nf = ny - 2. Exact for a cylinder, where n == e_r. Pre: ny >= 4, 0 < rWall < rFar.
double ogridNominalH1(int ny, double rWall, double rFar, double beta);

// Radial FLUID cells spanned by a wall-normal length `len` on that circle:
// nf t* with D(t*) = len / (rFar - rWall), by bisection (D is monotone); nf when
// len reaches the far circle. Pre: as ogridNominalH1, len >= 0.
double ogridRadialCells(double len, int ny, double rWall, double rFar,
                        double beta);

// Reference box for an O-grid, so the validator can size the Grid without an
// sdf: hx = 2 pi / nx, hy = (rFar - rWall) / (ny - 2), hz = span / nz.
// hy is NOMINAL -- the radial axis is Extend, so no node ever picks up a y
// image and hy enters no metric; it exists so a reported spacing is
// meaningful and so hy > 0 pins rWall < rFar.
// Pre: nx >= 3, ny >= 4, nz >= 1, 0 < rWall < rFar, span > 0.
Grid ogridGrid(int nx, int ny, int nz, double rWall, double rFar, double span);

// Build the annular mesh whose inner boundary is the zero level set of sdf.
// g must be the ogridGrid() box (hx and hz are checked; hy is ignored).
// Errors (never asserts) on: a centre outside the solid, a ray with no root
// inside rFar (body larger than rFar), a ray with more than one sign change
// (not star-shaped about the centre), a contour that moves with z, and a
// degenerate wall normal -- every message names the failing angle.
// The scan resolves radial features to rFar/1024 and samples 5 z planes over
// [cz - span/2, cz + span/2]; z-invariance is required to 1e-9 rFar.
// Cell validity (positive volumes, invertible Jacobians) stays CurvMetrics::
// build's job; call it on the result.
std::expected<OGrid, std::string> buildOGrid(const Grid& g, const Sdf& sdf,
                                             const OGridSpec& spec);

// Reference box of a revolved O-grid: nx = nTheta + 2 (a ghost column past
// each axis), ny radial INCLUDING two ghost rows, nz = 3 (wedge: one real
// plane, two ghost planes) or nphi; hx = pi / nTheta, hz = dphi or
// 2 pi / nphi, hy nominal as ogridGrid's.
// Pre: nTheta >= 4, ny >= 4, 0 < rWall < rFar, r valid (RevolvedSpec).
Grid ogridRevolvedGrid(int nTheta, int ny, const RevolvedSpec& r, double rWall,
                       double rFar);

// The revolved O-grid around a body of revolution about the x axis through
// (cx, cy, cz) -- masterplan_flowrans section 4.5. Node i = 1..nTheta+1 at
// theta = pi (i - 1)/nTheta from the UPSTREAM axis (node 1 and node
// nTheta+1 ON the axis, rho == 0 exactly), the meridional nodes the
// buildOGrid blend in the (x, rho) half-plane, revolved to (x, cy + rho cos
// phi, cz + rho sin phi). Wedge node planes at phi = -3/2, -1/2, +1/2 dphi
// (written as exact z-mirrors; MirrorZ supplies +3/2), full revolution at
// phi = 2 pi k / nphi with the planes k and k + nphi/2 exactly opposite.
// Ghost column nodes 0 (and nTheta+2 by Extend) are the point reflections
// 2 node(1) - node(2): a positive-volume placeholder no real cell reads.
// Errors (never asserts) as buildOGrid, plus an sdf that is not rotationally
// invariant (5 azimuths, 1e-9 rFar). Requires spec.span == 1 (unused).
std::expected<OGrid, std::string> buildRevolvedOGrid(const Grid& g,
                                                     const Sdf& sdf,
                                                     const OGridSpec& spec,
                                                     const RevolvedSpec& r);

// In-plane midpoints of the nx wall faces (the +j face of cell (i, 0):
// nodes (i, 1) and (i+1, 1) averaged, angular wrap included) -- the lever
// arms and mirror points of a rigidly rotating wall (gpu/ogrid_bc).
// Pre: og from buildOGrid.
std::vector<std::array<double, 2>> ogridWallMidpoints(const OGrid& og);

// In-plane vector from the inner ghost cell (i, 0) to the first fluid cell
// (i, 1), centres taken as their four-node averages: the span the wall row's
// pressure image covers (p0 = p1 - grad p . span). The ghost row is the
// reflection of row 1 about the wall, so this lies along the wall normal and
// is twice the first centre's wall distance. Pre: og from buildOGrid.
std::vector<std::array<double, 2>> ogridWallSpans(const OGrid& og);

} // namespace dns
