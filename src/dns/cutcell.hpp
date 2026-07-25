// Cut-cell embedded boundary on the MAC grid: a sharp, geometry-conforming
// alternative to the staircase mask. The solid contributes cell fluid-volume
// fractions and per-face open-area apertures; the finite-volume divergence,
// pressure Poisson and gradient are weighted by them so the no-through-flow
// wall condition is enforced exactly on the cut faces (not staircased).
//
// Small cells: the scheme never divides by the fluid fraction (momentum is
// staircase-masked -- though the GPU apps sharpen the viscous wall with mirror
// ghost values, see gpu/ghostfill.hpp -- the projection is aperture-weighted
// only), so tiny cells cannot destabilize the update. What they can do is degrade the pressure solve:
// a nearly closed cell couples to the fluid through vanishing apertures and
// contributes near-zero eigenvalues (a fully isolated one contributes an exact
// extra null mode) that stall the FP32 PCG. sanitizeCut() removes both by
// thresholding, at a geometric perturbation of O(lambdaMin * h) -- below the
// first-order boundary discretization error.
#pragma once

#include "dns/bodies.hpp"   // RotatedCube / Sphere / CylinderZ
#include "dns/field.hpp"
#include "dns/grid.hpp"

#include <expected>
#include <string>
#include <vector>

namespace dns {

// Per-cell fluid fraction and per-face open apertures, all in [0,1] and indexed
// by Grid::idx. ax[idx(i,j,k)] is the u-face (left, x=i*hx), ay the v-face
// (y=j*hy), az the w-face (z=k*hz). lambda is the cell fluid-volume fraction.
struct CutGeometry
{
    Scalar lambda, ax, ay, az;

    // Exact geometry for a rotated cube by polygon/segment clipping.
    static CutGeometry make(const Grid& g, const RotatedCube& c);

    // Sphere geometry: apertures by exact circle/rectangle overlap, fluid
    // fractions by adaptive integration of the overlap area (abs err ~1e-12).
    // Arbitrarily small cut cells result; run sanitizeCut before solving.
    static CutGeometry make(const Grid& g, const Sphere& s);

    // z-cylinder geometry, exact: lambda and az by circle/rectangle overlap,
    // ax/ay by the chord the circle cuts from the face; z-invariant.
    static CutGeometry make(const Grid& g, const CylinderZ& c);

    // A face carrying fluid; below this it is treated as a solid wall.
    static constexpr double openTol = 1e-9;
};

// Cut-region dispatch mask for the apertured momentum kernel (--cut-flux):
// 1 on every cell within DISPATCH RADIUS 2 (the widest read footprint of the
// apertured stencils: the skew correction at face i reaches cellDiv(i-1),
// whose apertures sit one more cell out) of any cell with lambda < 1 or any
// non-unity incident face aperture; 0 elsewhere. Cells at 0 take the plain
// interior stencil VERBATIM, which is what makes the interior bit-identical
// to the uniform kernel. Deterministic fixed-order dilation; periodic wrap.
std::vector<unsigned char> buildCutFlag(const Grid& g, const CutGeometry& cut);

// Clear the cut flag within `width` cells of every domain boundary. The ring
// closure (closeBoxBoundary) makes the whole boundary ring non-unity, but the
// far-field machinery (tunnel rows, NSCBC/sponge strips) OWNS those cells --
// aperture-weighting their fluxes fights the boundary scheme (measured: the
// compressible probe diverged through the NSCBC rows). The cut treatment is
// for the body, which the validator keeps away from the boundary. NB floor/
// fence bodies touch the boundary and lose the cut treatment there.
void clearBoundaryFlag(const Grid& g, std::vector<unsigned char>& flag,
                       int width);

// What sanitizeCut changed; min* are the post-sanitize conditioning bounds.
struct CutSanitizeStats
{
    std::size_t cellsSolidified; // fluid fraction < lambdaMin -> solid
    std::size_t facesClosed;     // aperture < apertureMin or solid-adjacent -> wall
    std::size_t pocketsClosed;   // fluid cells left with no open face -> solid
    double minLambda;            // min fluid-cell fraction remaining
    double minAperture;          // min open aperture remaining
};

// Small-cell treatment: solidify cells with lambda < lambdaMin, close faces
// with aperture < apertureMin or a solid cell on either side, then solidify
// fluid cells left without any open face (isolated pockets are exact null
// modes of the apertured Poisson operator). Runs to a fixed point. Post:
// every open face joins two fluid cells; every fluid cell has an open face.
// Pre: thresholds in [0,1); geometry perturbation is O(lambdaMin * h).
CutSanitizeStats sanitizeCut(const Grid& g, CutGeometry& cut,
                             double lambdaMin = 1e-2, double apertureMin = 1e-2);

// Close a one-cell ghost ring for confined-channel (non-periodic) runs: ghost cells
// become solid and every face on or outside the interior-box boundary gets a
// zero aperture, making the apertured operator homogeneous-Neumann there.
// Prescribed-flux faces (inlet/outlet) stay closed too; their fluxes enter the
// projection through explicit RHS corrections. Pre: nx,ny >= 4; nz >= 4 or
// nz == 1 (2D: the z axis stays periodic/untouched).
void closeBoxBoundary(const Grid& g, CutGeometry& cut);

// Finite-volume divergence weighted by face apertures: out_c = sum of apertured
// face fluxes / V. Reduces to the plain divergence when all apertures are 1.
void divergenceCut(const Grid& g, const CutGeometry& cut, const VectorField& U,
                   Scalar& out);

// U -= scale * grad(p) applied only on open faces (aperture > openTol); wall
// faces keep their (zero) normal velocity.
void subtractGradientCut(const Grid& g, const CutGeometry& cut, const Scalar& p,
                         double scale, VectorField& U);

// Zero every face whose aperture is closed: enforces u.n = 0 on the solid.
void maskSolidFaces(const Grid& g, const CutGeometry& cut, VectorField& U);

// max |apertured div| over fluid cells (lambda > 0); the cut-cell continuity
// residual, the diagnostic the projection drives to zero.
double maxAbsDivergenceCut(const Grid& g, const CutGeometry& cut,
                           const VectorField& U);

// Solves the aperture-weighted pressure Poisson M(p) = rhs by conjugate
// gradients on the symmetric operator
//   M(p)_c = sum_f a_f (p_nbr - p_c) / h_f^2      (a_f the incident aperture)
// over the fluid region, with the (single) constant null mode projected out.
// Neumann/no-flux on the embedded boundary is implicit in the zero apertures.
class CutPoissonSolver
{
public:
    // An all-solid geometry (no fluid cell) is a USER-caused input, not a
    // programmer error -- a body that swallows the box, or a mesh that failed
    // to sample. It used to be an assert, so in Release projectMean divided by
    // zero and poisoned every subsequent solve. Reported instead.
    static std::expected<CutPoissonSolver, std::string>
    make(const Grid& g, const CutGeometry& cut, int maxIters, double tol);

    CutPoissonSolver(CutPoissonSolver&&) = default;

    // Solve M(p) = rhs; rhs mean over fluid cells is projected out. p has zero
    // mean over fluid cells on return (zero in solid cells). warmStart reuses
    // the incoming p as the initial guess (cheap for slowly varying pressure).
    // Returns the iterations taken, or maxIters (non-convergence) if the
    // residual is non-finite -- a NaN residual is DIVERGENCE, never a solve
    // that converged instantly. Same contract as the GPU CG.
    int solve(const Scalar& rhs, Scalar& p, bool warmStart = false);

private:
    CutPoissonSolver(const Grid& g, const CutGeometry& cut, int maxIters,
                     double tol, std::size_t nFluid);

    void apply(const Scalar& x, Scalar& y) const; // y = -M(x), SPD on fluid
    double fluidDot(const Scalar& a, const Scalar& b) const;
    void projectMean(Scalar& x) const;

    Grid g_;
    const CutGeometry& cut_;
    int maxIters_;
    double tol_;
    std::size_t nFluid_;
    Scalar r_, d_, q_; // CG workspace
};

// One cut-cell projection: solve M(phi) = divCut(U)/dt, U -= dt grad(phi) on
// open faces, then re-mask the solid. Post: apertured divergence ~ 0.
struct CutProjectionScratch
{
    Scalar div, phi;
    static CutProjectionScratch make(const Grid& g) { return {zeros(g), zeros(g)}; }
};

void projectCut(const Grid& g, const CutGeometry& cut, CutPoissonSolver& poisson,
                double dt, VectorField& U, CutProjectionScratch& s);

} // namespace dns
